"""ARMT transformer layer with pluggable recurrent memory backends."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayer

from .monitoring import build_mean_metric, build_ratio_of_means_metric, merge_metric_primitives
from .recurrent_memory import build_recurrent_memory_backend

_MEM_TOKEN_COSINE_HIGH_THRESHOLD = 0.8
_SUPPORTED_MEMORY_WRITE_SOURCES = (
    "mem_tokens",
    "post_mlp_context",
    "post_attn_context",
    "pre_attn_context",
)
_LOGGER = logging.getLogger(__name__)


class ARMTLayer(TransformerLayer):
    """TransformerLayer with recurrent memory retrieval and update."""

    def __init__(
        self,
        config,
        submodules,
        layer_number: int = 1,
        num_mem_tokens: int = 16,
        d_mem: Optional[int] = None,
        armt_n_heads: int = 1,
        armt_head_dim: Optional[int] = None,
        nu: int = 3,
        use_denom: bool = True,
        gating: bool = False,
        correction: bool = True,
        tbptt_mode: bool = True,
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
        armt_windowed_full_attn_backend: str = "native",
        armt_equal_window_full_attn_path: str = "legacy",
        recurrent_memory_backend: str = "associative",
        recurrent_gdn_use_fla_kernel: bool = True,
        recurrent_gdn_use_causal_conv1d: bool = True,
        recurrent_gdn_conv_kernel_size: int = 4,
        recurrent_gdn_key_head_dim: Optional[int] = None,
        recurrent_gdn_value_head_dim: Optional[int] = None,
        recurrent_gdn_num_key_heads: Optional[int] = None,
        recurrent_gdn_num_value_heads: Optional[int] = None,
        recurrent_slot_num_slots: Optional[int] = None,
        recurrent_slot_num_heads: Optional[int] = None,
        recurrent_slot_head_dim: Optional[int] = None,
        recurrent_slot_read_attn_backend: str = "flash",
        recurrent_mem_qk_norm: bool = False,
        recurrent_memory_input_pre_norm: bool = False,
        armt_memory_write_source: str = "mem_tokens",
        **kwargs,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        self.num_mem_tokens = num_mem_tokens
        self.recurrent_chunk_size = recurrent_chunk_size
        self.full_attn_window_size = (
            full_attn_window_size if full_attn_window_size is not None else recurrent_chunk_size
        )
        self.armt_windowed_full_attn_backend = armt_windowed_full_attn_backend
        self.armt_equal_window_full_attn_path = armt_equal_window_full_attn_path
        self.recurrent_memory_backend = recurrent_memory_backend
        self.recurrent_memory_input_pre_norm = recurrent_memory_input_pre_norm
        if armt_memory_write_source not in _SUPPORTED_MEMORY_WRITE_SOURCES:
            raise ValueError(
                "armt_memory_write_source must be one of "
                f"{_SUPPORTED_MEMORY_WRITE_SOURCES}, got {armt_memory_write_source!r}"
            )
        self.armt_memory_write_source = armt_memory_write_source
        self.recurrent_memory_layer = None

        common_kwargs = {
            "d_model": config.hidden_size,
            "num_mem_tokens": num_mem_tokens,
            "dtype": getattr(config, "params_dtype", torch.bfloat16),
            "tbptt_mode": tbptt_mode,
            "normalization": getattr(config, "normalization", "LayerNorm"),
            "norm_epsilon": getattr(config, "layernorm_epsilon", 1e-5),
            "use_input_pre_norm": (
                recurrent_memory_input_pre_norm and armt_memory_write_source != "pre_attn_context"
            ),
        }

        if recurrent_memory_backend == "associative":
            self.recurrent_memory_layer = build_recurrent_memory_backend(
                recurrent_memory_backend,
                d_mem=d_mem or config.hidden_size,
                n_heads=armt_n_heads,
                head_dim=armt_head_dim,
                use_denom=use_denom,
                gating=gating,
                correction=correction,
                nu=nu,
                use_qk_norm=recurrent_mem_qk_norm,
                **common_kwargs,
            )
        elif recurrent_memory_backend == "cross_attn_slots":
            self.recurrent_memory_layer = build_recurrent_memory_backend(
                recurrent_memory_backend,
                num_slots=recurrent_slot_num_slots or num_mem_tokens,
                num_heads=recurrent_slot_num_heads or 1,
                head_dim=recurrent_slot_head_dim,
                read_attn_backend=recurrent_slot_read_attn_backend,
                use_qk_norm=recurrent_mem_qk_norm,
                **common_kwargs,
            )
        else:
            self.recurrent_memory_layer = build_recurrent_memory_backend(
                recurrent_memory_backend,
                conv_kernel_size=recurrent_gdn_conv_kernel_size,
                key_head_dim=recurrent_gdn_key_head_dim,
                value_head_dim=recurrent_gdn_value_head_dim,
                num_key_heads=recurrent_gdn_num_key_heads,
                num_value_heads=recurrent_gdn_num_value_heads,
                use_fla_kernel=recurrent_gdn_use_fla_kernel,
                use_causal_conv1d=recurrent_gdn_use_causal_conv1d,
                use_qk_l2norm=recurrent_mem_qk_norm,
                **common_kwargs,
            )
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        self._captured_input_layernorm_output = None
        self._pre_attn_capture_fallback_warned = False
        self._input_layernorm_capture_handle = None
        input_layernorm = getattr(self, "input_layernorm", None)
        if hasattr(input_layernorm, "register_forward_hook"):
            self._input_layernorm_capture_handle = input_layernorm.register_forward_hook(
                self._capture_input_layernorm_output
            )
        self.reset_monitoring_stats()

    def _get_memory_layer(self):
        if self.recurrent_memory_layer is not None:
            return self.recurrent_memory_layer
        raise RuntimeError("ARMTLayer has no recurrent memory layer configured.")

    def set_skip_read_memory_for_current_chunk(self, enabled: bool):
        self._skip_read_memory_for_current_chunk = bool(enabled)

    def set_current_chunk_is_first(self, enabled: bool):
        self._current_chunk_is_first = bool(enabled)

    def set_current_chunk_start_position(self, position: int):
        self._current_chunk_start_position = int(position)
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "set_current_chunk_start_position"):
            self_attention.set_current_chunk_start_position(position)

    def reset_monitoring_stats(self):
        self._monitoring_stats = {}
        self._get_memory_layer().reset_monitoring_stats()

    def _capture_input_layernorm_output(self, module, inputs, output):
        del module, inputs
        self._captured_input_layernorm_output = output

    def _uses_mem_tokens(self) -> bool:
        return self.armt_memory_write_source == "mem_tokens" and self.num_mem_tokens > 0

    def _accumulate_monitoring_stat(self, name: str, value: torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(
                f"ARMTLayer monitoring expects scalars, got {name}={tuple(value.shape)}"
            )
        value = value.reshape(()).to(dtype=torch.float32)
        current = self._monitoring_stats.get(name)
        if current is None:
            self._monitoring_stats[name] = value
        else:
            self._monitoring_stats[name] = current + value

    @staticmethod
    def _count_tensor(count: int, device: torch.device) -> torch.Tensor:
        return torch.tensor(float(count), device=device, dtype=torch.float32)

    def _gather_hidden_for_monitoring(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] == self.config.hidden_size:
            return hidden_states
        if parallel_state.model_parallel_is_initialized():
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                return tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)
        return hidden_states

    def _prepare_hidden_states_for_memory_ops(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
        return self._gather_hidden_for_monitoring(hidden_states)

    def _empty_like_sequence(self, hidden_states: torch.Tensor, *, input_is_sbh: bool) -> torch.Tensor:
        if input_is_sbh:
            return hidden_states[:0, :, :]
        return hidden_states[:, :0, :]

    def _split_context_and_mem(
        self,
        hidden_states: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._uses_mem_tokens():
            return hidden_states, self._empty_like_sequence(hidden_states, input_is_sbh=input_is_sbh)
        if input_is_sbh:
            return (
                hidden_states[:-self.num_mem_tokens, :, :],
                hidden_states[-self.num_mem_tokens :, :, :],
            )
        return (
            hidden_states[:, :-self.num_mem_tokens, :],
            hidden_states[:, -self.num_mem_tokens :, :],
        )

    def _update_token_monitoring_stats(
        self,
        context_part: torch.Tensor,
        mem_part: torch.Tensor,
        *,
        input_is_sbh: bool,
    ):
        if context_part.numel() > 0:
            context_norms = torch.linalg.vector_norm(context_part.float(), dim=-1)
            self._accumulate_monitoring_stat("context_token_norm_sum", context_norms.sum())
            self._accumulate_monitoring_stat(
                "context_token_count",
                self._count_tensor(context_norms.numel(), context_part.device),
            )

        if mem_part.numel() == 0:
            return

        mem_norms = torch.linalg.vector_norm(mem_part.float(), dim=-1)
        self._accumulate_monitoring_stat("mem_token_norm_sum", mem_norms.sum())
        self._accumulate_monitoring_stat(
            "mem_token_count",
            self._count_tensor(mem_norms.numel(), mem_part.device),
        )

        mem_tokens = mem_part.transpose(0, 1) if input_is_sbh else mem_part
        if mem_tokens.shape[1] < 2:
            return

        normalized = F.normalize(mem_tokens.float(), dim=-1, p=2.0)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal_mask = ~torch.eye(
            mem_tokens.shape[1],
            device=cosine.device,
            dtype=torch.bool,
        )
        cosine = cosine.masked_select(off_diagonal_mask.unsqueeze(0)).view(mem_tokens.shape[0], -1)
        self._accumulate_monitoring_stat("mem_token_cosine_sum", cosine.mean(dim=-1).sum())
        self._accumulate_monitoring_stat("mem_token_cosine_max_sum", cosine.max(dim=-1).values.sum())
        self._accumulate_monitoring_stat("mem_token_cosine_min_sum", cosine.min(dim=-1).values.sum())
        self._accumulate_monitoring_stat(
            "mem_token_cosine_gt_0p8_ratio_sum",
            (cosine > _MEM_TOKEN_COSINE_HIGH_THRESHOLD).float().mean(dim=-1).sum(),
        )
        self._accumulate_monitoring_stat(
            "mem_token_cosine_count",
            self._count_tensor(mem_tokens.shape[0], cosine.device),
        )

    def consume_monitoring_primitives(self):
        primitives = {}
        stats = self._monitoring_stats

        if "mem_token_count" in stats:
            primitives["armt/token/mem_token_norm_mean"] = build_mean_metric(
                stats["mem_token_norm_sum"],
                stats["mem_token_count"],
            )

        if "context_token_count" in stats:
            primitives["armt/token/context_token_norm_mean"] = build_mean_metric(
                stats["context_token_norm_sum"],
                stats["context_token_count"],
            )

        if "mem_token_count" in stats and "context_token_count" in stats:
            primitives["armt/token/mem_ctx_norm_ratio"] = build_ratio_of_means_metric(
                stats["mem_token_norm_sum"],
                stats["mem_token_count"],
                stats["context_token_norm_sum"],
                stats["context_token_count"],
            )

        if "mem_token_cosine_count" in stats:
            primitives["armt/token/mem_token_cosine_mean"] = build_mean_metric(
                stats["mem_token_cosine_sum"],
                stats["mem_token_cosine_count"],
            )
            primitives["armt/token/mem_token_cosine_max_mean"] = build_mean_metric(
                stats["mem_token_cosine_max_sum"],
                stats["mem_token_cosine_count"],
            )
            primitives["armt/token/mem_token_cosine_min_mean"] = build_mean_metric(
                stats["mem_token_cosine_min_sum"],
                stats["mem_token_cosine_count"],
            )
            primitives["armt/token/mem_token_cosine_gt_0p8_ratio_mean"] = build_mean_metric(
                stats["mem_token_cosine_gt_0p8_ratio_sum"],
                stats["mem_token_cosine_count"],
            )

        self._monitoring_stats = {}
        merge_metric_primitives(primitives, self._get_memory_layer().consume_monitoring_primitives())
        return primitives

    def _resolve_write_source(
        self,
        *,
        memory_layer,
        pre_attn_hidden_states: torch.Tensor,
        attn_hidden_states: torch.Tensor,
        post_mlp_hidden_states: torch.Tensor,
        input_is_sbh: bool,
    ) -> tuple[torch.Tensor, bool]:
        if self.armt_memory_write_source == "mem_tokens":
            prepared = self._prepare_hidden_states_for_memory_ops(post_mlp_hidden_states)
            _, mem_part = self._split_context_and_mem(prepared, input_is_sbh=input_is_sbh)
            return mem_part, False

        if self.armt_memory_write_source == "post_attn_context":
            prepared = self._prepare_hidden_states_for_memory_ops(attn_hidden_states)
            context_part, _ = self._split_context_and_mem(prepared, input_is_sbh=input_is_sbh)
            return context_part, False

        if self.armt_memory_write_source == "post_mlp_context":
            prepared = self._prepare_hidden_states_for_memory_ops(post_mlp_hidden_states)
            context_part, _ = self._split_context_and_mem(prepared, input_is_sbh=input_is_sbh)
            return context_part, False

        if self.armt_memory_write_source == "pre_attn_context":
            prepared = pre_attn_hidden_states
            input_already_pre_normed = False
            if self.recurrent_memory_input_pre_norm:
                captured = self._captured_input_layernorm_output
                if captured is not None:
                    prepared = captured
                else:
                    if not self._pre_attn_capture_fallback_warned:
                        _LOGGER.warning(
                            "ARMTLayer pre-attn write source fell back to recomputing "
                            "input_layernorm because _captured_input_layernorm_output is None."
                        )
                        self._pre_attn_capture_fallback_warned = True
                    input_layernorm = getattr(self, "input_layernorm", None)
                    if input_layernorm is None:
                        raise RuntimeError(
                            "pre_attn_context with recurrent_memory_input_pre_norm=True "
                            "requires TransformerLayer.input_layernorm"
                        )
                    prepared = input_layernorm(prepared)
                input_already_pre_normed = True
            prepared = self._prepare_hidden_states_for_memory_ops(prepared)
            context_part, _ = self._split_context_and_mem(prepared, input_is_sbh=input_is_sbh)
            return context_part, input_already_pre_normed

        raise RuntimeError(
            f"Unsupported armt_memory_write_source: {self.armt_memory_write_source!r}"
        )

    def forward(self, *args, **kwargs):
        # Keep TransformerLayer.forward compatibility for local cudagraph inference.
        kwargs.pop("dynamic_inference_decode_only", None)

        if args:
            hidden_states = args[0]
            attention_args = args[1:]
        else:
            if "hidden_states" not in kwargs:
                raise TypeError("ARMTLayer.forward missing required argument: 'hidden_states'")
            hidden_states = kwargs.pop("hidden_states")
            attention_args = ()

        # Project convention: ARMT only supports Megatron's SBH layout.
        input_is_sbh = True
        memory_layer = self._get_memory_layer()
        self._captured_input_layernorm_output = None
        pre_attn_hidden_states = hidden_states

        # Step 1: Associate (Memory Retrieval)
        if not self._skip_read_memory_for_current_chunk:
            retrieved = memory_layer.associate(hidden_states, input_is_sbh=input_is_sbh)
            hidden_states = hidden_states + retrieved

        pre_attn_hidden_states = hidden_states

        # Step 2 & 3: Attention + MLP
        attn_hidden_states, context = self._forward_attention(
            hidden_states,
            *attention_args,
            **kwargs,
        )
        hidden_states = self._forward_mlp(
            attn_hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )

        # Step 4: Update Memory and collect token monitoring stats.
        monitoring_hidden_states = self._prepare_hidden_states_for_memory_ops(hidden_states)
        context_part, mem_part = self._split_context_and_mem(
            monitoring_hidden_states,
            input_is_sbh=input_is_sbh,
        )

        self._update_token_monitoring_stats(
            context_part,
            mem_part,
            input_is_sbh=input_is_sbh,
        )
        write_part, input_already_pre_normed = self._resolve_write_source(
            memory_layer=memory_layer,
            pre_attn_hidden_states=pre_attn_hidden_states,
            attn_hidden_states=attn_hidden_states,
            post_mlp_hidden_states=hidden_states,
            input_is_sbh=input_is_sbh,
        )
        if write_part.numel() != 0:
            memory_layer.update_mem(
                write_part,
                input_is_sbh=input_is_sbh,
                input_already_pre_normed=input_already_pre_normed,
            )

        return hidden_states, context

    def reset_memory(self):
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        self._captured_input_layernorm_output = None
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "reset_window_kv_cache"):
            self_attention.reset_window_kv_cache()
        self._get_memory_layer().reset_memory()
