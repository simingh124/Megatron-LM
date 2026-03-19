"""ARMT transformer layer with associative memory."""

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayer

from .associative_layer import AssociativeLayer
from .monitoring import build_mean_metric, build_ratio_of_means_metric, merge_metric_primitives

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
        **kwargs,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        self.num_mem_tokens = num_mem_tokens

        self.associative_layer = AssociativeLayer(
            d_model=config.hidden_size,
            num_mem_tokens=num_mem_tokens,
            d_mem=d_mem or config.hidden_size,
            n_heads=armt_n_heads,
            use_denom=use_denom,
            gating=gating,
            correction=correction,
            nu=nu,
            dtype=getattr(config, "params_dtype", torch.bfloat16),
            tbptt_mode=tbptt_mode,
        )
        self.reset_monitoring_stats()

    def reset_monitoring_stats(self):
        self._monitoring_stats = {}
        self.associative_layer.reset_monitoring_stats()

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
        merge_metric_primitives(primitives, self.associative_layer.consume_monitoring_primitives())
        return primitives

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Project convention: ARMT only supports Megatron's SBH layout.
        input_is_sbh = True

        # Step 1: Associate (Memory Retrieval)
        retrieved = self.associative_layer.associate(hidden_states, input_is_sbh=input_is_sbh)
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
        self.associative_layer.update_mem(mem_part, input_is_sbh=input_is_sbh)

        return hidden_states, context

    def reset_memory(self):
        self.associative_layer.reset_memory()
