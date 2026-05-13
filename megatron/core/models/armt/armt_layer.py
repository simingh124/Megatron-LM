"""ARMT transformer layer with pluggable recurrent memory backends."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayer

from .monitoring import (
    build_mean_metric,
    build_ratio_metric,
    build_ratio_of_means_metric,
    merge_metric_primitives,
)
from .recurrent_memory import build_recurrent_memory_backend

_MEM_TOKEN_COSINE_HIGH_THRESHOLD = 0.8
_SUPPORTED_MEMORY_WRITE_SOURCES = (
    "mem_tokens",
    "post_mlp_context",
    "post_attn_context",
    "pre_attn_context",
)
_SUPPORTED_READ_MEMORY_MODES = ("none", "per_layer", "shared", "initial")
_LOGGER = logging.getLogger(__name__)


class ARMTLayer(TransformerLayer):
    """TransformerLayer with recurrent memory retrieval and update."""

    def __init__(
        self,
        config,
        submodules,
        layer_number: int = 1,
        num_mem_tokens: int = 16,
        num_read_mem_tokens: int = 0,
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
        recurrent_gdn_read_mode: str = "normal",
        recurrent_slot_num_slots: Optional[int] = None,
        recurrent_slot_num_heads: Optional[int] = None,
        recurrent_slot_head_dim: Optional[int] = None,
        recurrent_slot_read_attn_backend: str = "flash",
        recurrent_mem_qk_norm: bool = False,
        recurrent_memory_input_pre_norm: bool = False,
        armt_memory_write_source: str = "mem_tokens",
        armt_read_memory_mode: str = "none",
        armt_read_memory_residual: bool = False,
        log_read_position_metrics_to_tensorboard: bool = False,
        log_read_prefix_attn_mass_to_tensorboard: bool = False,
        **kwargs,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        self.num_mem_tokens = num_mem_tokens
        self.num_read_mem_tokens = num_read_mem_tokens
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
        if armt_read_memory_mode not in _SUPPORTED_READ_MEMORY_MODES:
            raise ValueError(
                "armt_read_memory_mode must be one of "
                f"{_SUPPORTED_READ_MEMORY_MODES}, got {armt_read_memory_mode!r}"
            )
        self.armt_memory_write_source = armt_memory_write_source
        self.armt_read_memory_mode = armt_read_memory_mode
        self.armt_read_memory_residual = bool(armt_read_memory_residual)
        self.log_read_prefix_attn_mass_to_tensorboard = bool(
            log_read_prefix_attn_mass_to_tensorboard
        )
        self.recurrent_memory_layer = None
        params_dtype = getattr(config, "params_dtype", torch.bfloat16)

        common_kwargs = {
            "config": config,
            "d_model": config.hidden_size,
            "num_mem_tokens": num_mem_tokens,
            "tbptt_mode": tbptt_mode,
            "normalization": getattr(config, "normalization", "LayerNorm"),
            "norm_epsilon": getattr(config, "layernorm_epsilon", 1e-5),
            "use_input_pre_norm": (
                recurrent_memory_input_pre_norm and armt_memory_write_source != "pre_attn_context"
            ),
            "log_read_position_metrics_to_tensorboard": (
                log_read_position_metrics_to_tensorboard
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
                dtype=params_dtype,
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
                dtype=params_dtype,
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
                read_mode=recurrent_gdn_read_mode,
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
        self._current_layer_read_memory_embeddings: Optional[torch.Tensor] = None
        input_layernorm = getattr(self, "input_layernorm", None)
        if hasattr(input_layernorm, "register_forward_hook"):
            self._input_layernorm_capture_handle = input_layernorm.register_forward_hook(
                self._capture_input_layernorm_output
            )
        self._collect_monitoring_for_current_iteration = True
        read_prefix_mode_setter = getattr(self.recurrent_memory_layer, "set_read_prefix_mode", None)
        if callable(read_prefix_mode_setter):
            read_prefix_mode_setter(self._uses_read_prefix_mode())
        self.reset_monitoring_stats()

    def _get_memory_layer(self):
        if self.recurrent_memory_layer is not None:
            return self.recurrent_memory_layer
        raise RuntimeError("ARMTLayer has no recurrent memory layer configured.")

    def _uses_mem_tokens(self) -> bool:
        return self.armt_memory_write_source == "mem_tokens" and self.num_mem_tokens > 0

    def _uses_read_prefix_mode(self) -> bool:
        return self.armt_read_memory_mode != "none" and self.num_read_mem_tokens > 0

    def _current_read_mem_tokens(self) -> int:
        if not self._uses_read_prefix_mode() or self._skip_read_memory_for_current_chunk:
            return 0
        return self.num_read_mem_tokens

    def _current_write_mem_tokens(self) -> int:
        if not self._uses_mem_tokens():
            return 0
        return self.num_mem_tokens

    def set_skip_read_memory_for_current_chunk(self, enabled: bool):
        self._skip_read_memory_for_current_chunk = bool(enabled)
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "set_skip_read_memory_for_current_chunk"):
            self_attention.set_skip_read_memory_for_current_chunk(enabled)

    def set_current_chunk_is_first(self, enabled: bool):
        self._current_chunk_is_first = bool(enabled)

    def set_current_chunk_start_position(self, position: int):
        self._current_chunk_start_position = int(position)
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "set_current_chunk_start_position"):
            self_attention.set_current_chunk_start_position(position)

    def set_current_layer_read_memory_embeddings(self, embeddings: Optional[torch.Tensor]):
        self._current_layer_read_memory_embeddings = embeddings

    def set_collect_monitoring_for_current_iteration(self, enabled: bool):
        self._collect_monitoring_for_current_iteration = bool(enabled)
        memory_layer = self._get_memory_layer()
        setter = getattr(memory_layer, "set_collect_monitoring_for_current_iteration", None)
        if callable(setter):
            setter(enabled)
        self_attention = getattr(self, "self_attention", None)
        setter = getattr(self_attention, "set_collect_monitoring_for_current_iteration", None)
        if callable(setter):
            setter(enabled)

    def reset_monitoring_stats(self):
        self._monitoring_stats = {}
        self._get_memory_layer().reset_monitoring_stats()
        self_attention = getattr(self, "self_attention", None)
        resetter = getattr(self_attention, "reset_monitoring_stats", None)
        if callable(resetter):
            resetter()

    def _capture_input_layernorm_output(self, module, inputs, output):
        del module, inputs
        self._captured_input_layernorm_output = output

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

    def _restore_hidden_states_after_memory_ops(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                hidden_states, group=tp_group
            )
        return hidden_states

    def _empty_like_sequence(self, hidden_states: torch.Tensor, *, input_is_sbh: bool) -> torch.Tensor:
        if input_is_sbh:
            return hidden_states[:0, :, :]
        return hidden_states[:, :0, :]

    def _split_read_context_and_write(
        self,
        hidden_states: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        read_tokens = self._current_read_mem_tokens()
        write_tokens = self._current_write_mem_tokens()
        if input_is_sbh:
            seq_len = hidden_states.shape[0]
            read_end = read_tokens
            write_start = seq_len - write_tokens
            return (
                hidden_states[:read_end, :, :],
                hidden_states[read_end:write_start, :, :],
                hidden_states[write_start:, :, :],
            )
        seq_len = hidden_states.shape[1]
        read_end = read_tokens
        write_start = seq_len - write_tokens
        return (
            hidden_states[:, :read_end, :],
            hidden_states[:, read_end:write_start, :],
            hidden_states[:, write_start:, :],
        )

    def _concat_read_context_and_write(
        self,
        read_part: torch.Tensor,
        context_part: torch.Tensor,
        write_part: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> torch.Tensor:
        parts = [part for part in (read_part, context_part, write_part) if part.numel() > 0]
        if not parts:
            raise RuntimeError("ARMTLayer expected at least one non-empty hidden-state partition")
        return torch.cat(parts, dim=0 if input_is_sbh else 1)

    def _split_context_and_mem(
        self,
        hidden_states: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, context_part, write_part = self._split_read_context_and_write(
            hidden_states,
            input_is_sbh=input_is_sbh,
        )
        return context_part, write_part

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

    def _resolve_read_query(
        self,
        read_part: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> torch.Tensor:
        if self.armt_read_memory_mode == "initial":
            return read_part

        embeddings = self._current_layer_read_memory_embeddings
        if embeddings is None:
            raise RuntimeError(
                "ARMTLayer missing current-layer read-memory embeddings for "
                f"armt_read_memory_mode={self.armt_read_memory_mode!r}"
            )
        if embeddings.shape[0] != self._current_read_mem_tokens():
            raise ValueError(
                "Current-layer read-memory embedding count does not match current read token count: "
                f"{embeddings.shape[0]} vs {self._current_read_mem_tokens()}"
            )

        if input_is_sbh:
            batch_size = read_part.shape[1]
            expanded = embeddings.unsqueeze(1).expand(-1, batch_size, -1)
        else:
            batch_size = read_part.shape[0]
            expanded = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return expanded.to(dtype=read_part.dtype, device=read_part.device)

    def _update_read_prefix_monitoring_stats(
        self,
        query: torch.Tensor,
        retrieved: torch.Tensor,
        output: torch.Tensor,
        context_part: torch.Tensor,
        *,
        input_is_sbh: bool,
    ) -> None:
        query_tokens = query.transpose(0, 1) if input_is_sbh else query
        retrieved_tokens = retrieved.transpose(0, 1) if input_is_sbh else retrieved
        output_tokens = output.transpose(0, 1) if input_is_sbh else output
        if output_tokens.numel() == 0:
            return

        query_norms = torch.linalg.vector_norm(query_tokens.float(), dim=-1)
        retrieved_norms = torch.linalg.vector_norm(retrieved_tokens.float(), dim=-1)
        output_norms = torch.linalg.vector_norm(output_tokens.float(), dim=-1)
        self._accumulate_monitoring_stat("read_prefix_query_norm_sum", query_norms.sum())
        self._accumulate_monitoring_stat(
            "read_prefix_query_count",
            self._count_tensor(query_norms.numel(), query_tokens.device),
        )
        self._accumulate_monitoring_stat("read_prefix_retrieved_norm_sum", retrieved_norms.sum())
        self._accumulate_monitoring_stat(
            "read_prefix_retrieved_count",
            self._count_tensor(retrieved_norms.numel(), retrieved_tokens.device),
        )
        self._accumulate_monitoring_stat("read_prefix_norm_sum", output_norms.sum())
        self._accumulate_monitoring_stat(
            "read_prefix_count",
            self._count_tensor(output_norms.numel(), output_tokens.device),
        )

        if output_tokens.shape[1] >= 2:
            normalized = F.normalize(output_tokens.float(), dim=-1, p=2.0)
            cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
            off_diagonal_mask = ~torch.eye(
                output_tokens.shape[1],
                device=cosine.device,
                dtype=torch.bool,
            )
            cosine = cosine.masked_select(off_diagonal_mask.unsqueeze(0)).view(
                output_tokens.shape[0], -1
            )
            self._accumulate_monitoring_stat("read_prefix_cosine_sum", cosine.mean(dim=-1).sum())
            self._accumulate_monitoring_stat(
                "read_prefix_cosine_max_sum", cosine.max(dim=-1).values.sum()
            )
            self._accumulate_monitoring_stat(
                "read_prefix_cosine_min_sum", cosine.min(dim=-1).values.sum()
            )
            self._accumulate_monitoring_stat(
                "read_prefix_cosine_gt_0p8_ratio_sum",
                (cosine > _MEM_TOKEN_COSINE_HIGH_THRESHOLD).float().mean(dim=-1).sum(),
            )
            self._accumulate_monitoring_stat(
                "read_prefix_cosine_count",
                self._count_tensor(output_tokens.shape[0], cosine.device),
            )

        if context_part.numel() > 0:
            context_tokens = context_part.transpose(0, 1) if input_is_sbh else context_part
            context_norms = torch.linalg.vector_norm(context_tokens.float(), dim=-1)
            self._accumulate_monitoring_stat("read_prefix_context_norm_sum", context_norms.sum())
            self._accumulate_monitoring_stat(
                "read_prefix_context_count",
                self._count_tensor(context_norms.numel(), context_tokens.device),
            )

    def consume_monitoring_primitives(self):
        if not self._collect_monitoring_for_current_iteration:
            self.reset_monitoring_stats()
            return {}

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

        if "read_prefix_count" in stats:
            primitives["armt/read_prefix/norm_mean"] = build_mean_metric(
                stats["read_prefix_norm_sum"],
                stats["read_prefix_count"],
            )

        if "read_prefix_cosine_count" in stats:
            primitives["armt/read_prefix/cosine_mean"] = build_mean_metric(
                stats["read_prefix_cosine_sum"],
                stats["read_prefix_cosine_count"],
            )
            primitives["armt/read_prefix/cosine_max_mean"] = build_mean_metric(
                stats["read_prefix_cosine_max_sum"],
                stats["read_prefix_cosine_count"],
            )
            primitives["armt/read_prefix/cosine_min_mean"] = build_mean_metric(
                stats["read_prefix_cosine_min_sum"],
                stats["read_prefix_cosine_count"],
            )
            primitives["armt/read_prefix/cosine_gt_0p8_ratio_mean"] = build_mean_metric(
                stats["read_prefix_cosine_gt_0p8_ratio_sum"],
                stats["read_prefix_cosine_count"],
            )

        if "read_prefix_query_count" in stats:
            primitives["armt/read_prefix/query_norm_mean"] = build_mean_metric(
                stats["read_prefix_query_norm_sum"],
                stats["read_prefix_query_count"],
            )
            primitives["armt/read_prefix/retrieved_norm_mean"] = build_mean_metric(
                stats["read_prefix_retrieved_norm_sum"],
                stats["read_prefix_retrieved_count"],
            )
            primitives["armt/read_prefix/retrieved_to_query_norm_ratio"] = build_ratio_metric(
                stats["read_prefix_retrieved_norm_sum"],
                stats["read_prefix_query_norm_sum"],
            )

        if "read_prefix_context_count" in stats:
            primitives["armt/read_prefix/read_to_context_norm_ratio"] = (
                build_ratio_of_means_metric(
                    stats["read_prefix_norm_sum"],
                    stats["read_prefix_count"],
                    stats["read_prefix_context_norm_sum"],
                    stats["read_prefix_context_count"],
                )
            )

        self._monitoring_stats = {}
        self_attention = getattr(self, "self_attention", None)
        attention_consumer = getattr(self_attention, "consume_monitoring_primitives", None)
        if callable(attention_consumer):
            merge_metric_primitives(primitives, attention_consumer())
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
        del memory_layer
        if self.armt_memory_write_source == "mem_tokens":
            prepared = self._prepare_hidden_states_for_memory_ops(post_mlp_hidden_states)
            _, _, write_part = self._split_read_context_and_write(
                prepared, input_is_sbh=input_is_sbh
            )
            return write_part, False

        if self.armt_memory_write_source == "post_attn_context":
            prepared = self._prepare_hidden_states_for_memory_ops(attn_hidden_states)
            _, context_part, _ = self._split_read_context_and_write(
                prepared, input_is_sbh=input_is_sbh
            )
            return context_part, False

        if self.armt_memory_write_source == "post_mlp_context":
            prepared = self._prepare_hidden_states_for_memory_ops(post_mlp_hidden_states)
            _, context_part, _ = self._split_read_context_and_write(
                prepared, input_is_sbh=input_is_sbh
            )
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
            _, context_part, _ = self._split_read_context_and_write(
                prepared, input_is_sbh=input_is_sbh
            )
            return context_part, input_already_pre_normed

        raise RuntimeError(
            f"Unsupported armt_memory_write_source: {self.armt_memory_write_source!r}"
        )

    def forward(self, *args, **kwargs):
        kwargs.pop("dynamic_inference_decode_only", None)

        if args:
            hidden_states = args[0]
            attention_args = args[1:]
        else:
            if "hidden_states" not in kwargs:
                raise TypeError("ARMTLayer.forward missing required argument: 'hidden_states'")
            hidden_states = kwargs.pop("hidden_states")
            attention_args = ()

        input_is_sbh = True
        attention_mask = attention_args[0] if attention_args else kwargs.get("attention_mask", None)
        memory_layer = self._get_memory_layer()
        self._captured_input_layernorm_output = None

        if self._uses_read_prefix_mode() and self._current_read_mem_tokens() > 0:
            full_hidden_states = self._prepare_hidden_states_for_memory_ops(hidden_states)
            read_part, context_part, write_part = self._split_read_context_and_write(
                full_hidden_states,
                input_is_sbh=input_is_sbh,
            )
            query = self._resolve_read_query(read_part, input_is_sbh=input_is_sbh)
            retrieved = memory_layer.associate(query, input_is_sbh=input_is_sbh)
            read_output = query + retrieved if self.armt_read_memory_residual else retrieved
            if self._collect_monitoring_for_current_iteration:
                self._update_read_prefix_monitoring_stats(
                    query,
                    retrieved,
                    read_output,
                    context_part,
                    input_is_sbh=input_is_sbh,
                )
            full_hidden_states = self._concat_read_context_and_write(
                read_output,
                context_part,
                write_part,
                input_is_sbh=input_is_sbh,
            )
            if self._collect_monitoring_for_current_iteration and self.log_read_prefix_attn_mass_to_tensorboard:
                input_layernorm = getattr(self, "input_layernorm", None)
                self_attention = getattr(self, "self_attention", None)
                tracker = getattr(self_attention, "track_read_prefix_attention_mass", None)
                if input_layernorm is not None and callable(tracker):
                    tracker(
                        input_layernorm(full_hidden_states),
                        attention_mask=attention_mask,
                        rotary_pos_emb=kwargs.get("rotary_pos_emb", None),
                    )
            hidden_states = full_hidden_states
            hidden_states = self._restore_hidden_states_after_memory_ops(hidden_states)
        elif self.armt_read_memory_mode == "none" and not self._skip_read_memory_for_current_chunk:
            retrieved = memory_layer.associate(hidden_states, input_is_sbh=input_is_sbh)
            hidden_states = hidden_states + retrieved

        pre_attn_hidden_states = hidden_states

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

        if self._collect_monitoring_for_current_iteration:
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
        self._current_layer_read_memory_embeddings = None
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "reset_window_kv_cache"):
            self_attention.reset_window_kv_cache()
        self._get_memory_layer().reset_memory()
