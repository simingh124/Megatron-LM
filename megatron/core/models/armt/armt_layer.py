"""ARMT transformer layer with pluggable recurrent memory backends."""

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayer

from .monitoring import build_mean_metric, build_ratio_of_means_metric, merge_metric_primitives
from .recurrent_memory import build_recurrent_memory_backend

_MEM_TOKEN_COSINE_HIGH_THRESHOLD = 0.8


class ARMTLayer(TransformerLayer):
    """TransformerLayer with associative memory retrieval and update."""

    def __init__(
        self,
        config,
        submodules,
        layer_number: int = 1,
        num_mem_tokens: int = 16,
        d_mem: Optional[int] = None,
        armt_n_heads: int = 1,
        nu: int = 3,
        use_denom: bool = True,
        gating: bool = False,
        correction: bool = True,
        tbptt_mode: bool = True,
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
        recurrent_memory_backend: str = "associative",
        recurrent_gdn_use_fla_kernel: bool = True,
        recurrent_gdn_use_causal_conv1d: bool = True,
        recurrent_gdn_conv_kernel_size: int = 4,
        recurrent_gdn_key_head_dim: Optional[int] = None,
        recurrent_gdn_value_head_dim: Optional[int] = None,
        recurrent_gdn_num_key_heads: Optional[int] = None,
        recurrent_gdn_num_value_heads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        self.num_mem_tokens = num_mem_tokens
        self.recurrent_chunk_size = recurrent_chunk_size
        self.full_attn_window_size = (
            full_attn_window_size if full_attn_window_size is not None else recurrent_chunk_size
        )
        self.recurrent_memory_backend = recurrent_memory_backend
        self.associative_layer = None
        self.recurrent_memory_layer = None

        common_kwargs = {
            "d_model": config.hidden_size,
            "num_mem_tokens": num_mem_tokens,
            "dtype": getattr(config, "params_dtype", torch.bfloat16),
            "tbptt_mode": tbptt_mode,
        }

        if recurrent_memory_backend == "associative":
            self.associative_layer = build_recurrent_memory_backend(
                recurrent_memory_backend,
                d_mem=d_mem or config.hidden_size,
                n_heads=armt_n_heads,
                use_denom=use_denom,
                gating=gating,
                correction=correction,
                nu=nu,
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
                **common_kwargs,
            )
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        self.reset_monitoring_stats()

    def _get_memory_layer(self):
        if self.recurrent_memory_layer is not None:
            return self.recurrent_memory_layer
        if self.associative_layer is not None:
            return self.associative_layer
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Project convention: ARMT only supports Megatron's SBH layout.
        input_is_sbh = True
        memory_layer = self._get_memory_layer()

        # Step 1: Associate (Memory Retrieval)
        if not self._skip_read_memory_for_current_chunk:
            retrieved = memory_layer.associate(hidden_states, input_is_sbh=input_is_sbh)
            hidden_states = hidden_states + retrieved

        # Step 2 & 3: Attention + MLP
        hidden_states, context = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Step 4: Update Memory and collect token monitoring stats.
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            monitoring_hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
        else:
            monitoring_hidden_states = hidden_states

        monitoring_hidden_states = self._gather_hidden_for_monitoring(monitoring_hidden_states)

        if input_is_sbh:
            context_part = monitoring_hidden_states[:-self.num_mem_tokens, :, :]
            mem_part = monitoring_hidden_states[-self.num_mem_tokens :, :, :]
        else:
            context_part = monitoring_hidden_states[:, :-self.num_mem_tokens, :]
            mem_part = monitoring_hidden_states[:, -self.num_mem_tokens :, :]

        self._update_token_monitoring_stats(
            context_part,
            mem_part,
            input_is_sbh=input_is_sbh,
        )
        memory_layer.update_mem(mem_part, input_is_sbh=input_is_sbh)

        return hidden_states, context

    def reset_memory(self):
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        self_attention = getattr(self, "self_attention", None)
        if hasattr(self_attention, "reset_window_kv_cache"):
            self_attention.reset_window_kv_cache()
        self._get_memory_layer().reset_memory()
