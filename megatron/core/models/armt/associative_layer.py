"""Associative memory components for ARMT."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel

from .monitoring import build_mean_metric, build_ratio_metric, build_rms_metric
from .norm_utils import build_recurrent_norm


class DPFP(nn.Module):
    """Deterministic Projected Feature Pairs."""

    def __init__(self, nu: int = 3):
        super().__init__()
        self.nu = nu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, heads, seq, dim] -> [B, heads, seq, 2 * nu * dim]
        x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu + 1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


class AssociativeLayer(nn.Module):
    """Associative memory layer with TBPTT-friendly state updates."""

    def __init__(
        self,
        d_model: Optional[int] = None,
        num_mem_tokens: int = 16,
        d_mem: Optional[int] = None,
        n_heads: int = 1,
        head_dim: Optional[int] = None,
        use_denom: bool = True,
        gating: bool = False,
        correction: bool = True,
        nu: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        tbptt_mode: bool = True,
        use_qk_norm: bool = False,
        use_input_pre_norm: bool = False,
        normalization: str = "LayerNorm",
        norm_epsilon: float = 1e-5,
        *,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()

        if d_model is None:
            d_model = hidden_size
        if d_model is None:
            raise ValueError("d_model (or hidden_size) must be provided for AssociativeLayer")

        if d_mem is None:
            d_mem = d_model

        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if d_mem % n_heads != 0:
            raise ValueError("d_mem must be divisible by n_heads")
        if head_dim is None:
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads when head_dim is omitted")
            head_dim = d_model // n_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")

        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_denom = use_denom
        self.gating = gating
        self.correction = correction
        self.nu = nu
        self.tbptt_mode = tbptt_mode
        self.use_qk_norm = use_qk_norm
        self.use_input_pre_norm = use_input_pre_norm

        self.d_key = 2 * nu * d_mem
        self.value_dim = self.n_heads * self.head_dim

        self.input_pre_norm = None
        if self.use_input_pre_norm:
            self.input_pre_norm = build_recurrent_norm(
                d_model,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )

        self.phi = DPFP(nu)
        self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=dtype)
        self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=dtype)
        self.W_mv = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype)
        self.W_mo = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype)

        # NOTE: Do not init to exact zeros. `W_mv` participates only through the
        # chunk-to-chunk memory state; if it's initialized to zeros, the memory update
        # becomes identically zero -> retrieval is identically zero -> gradients to
        # memory-write parameters are also identically zero.
        nn.init.normal_(self.W_mv.weight, mean=0.0, std=1e-3)

        if gating:
            self.W_mb = nn.Linear(d_model, self.value_dim, dtype=dtype)
        else:
            self.W_mb = nn.Linear(d_model, n_heads, dtype=dtype)

        self.register_buffer("W_mem", torch.empty(0), persistent=False)
        if use_denom:
            self.register_buffer("z", torch.empty(0), persistent=False)

        self._first_chunk = True
        self._pending_reset = True
        self.reset_monitoring_stats()

    def set_tbptt_mode(self, enabled: bool):
        self.tbptt_mode = enabled

    def reset_monitoring_stats(self):
        self._monitoring_stats: Dict[str, torch.Tensor] = {}

    def _accumulate_monitoring_stat(self, name: str, value: torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(
                f"AssociativeLayer monitoring expects scalars, got {name}={tuple(value.shape)}"
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

    def get_memory_state_breakdown(self, batch_size: int = 1) -> list[tuple[str, int]]:
        # Report the logical single-sample memory store size before DPFP expansion.
        w_mem_numel = batch_size * self.n_heads * (self.d_mem // self.n_heads) * self.head_dim
        return [("W_mem", w_mem_numel)]

    def consume_monitoring_primitives(self):
        stats = self._monitoring_stats
        primitives = {}

        if "retrieved_norm_count" in stats:
            primitives["armt/read/retrieved_norm_mean"] = build_mean_metric(
                stats["retrieved_norm_sum"],
                stats["retrieved_norm_count"],
            )
            primitives["armt/read/retrieved_to_hidden_ratio"] = build_ratio_metric(
                stats["retrieved_norm_sum"],
                stats["hidden_norm_sum"],
            )

        if "delta_mem_elem_count" in stats:
            primitives["armt/write/delta_mem_norm"] = build_rms_metric(
                stats["delta_mem_sq_sum"],
                stats["delta_mem_elem_count"],
            )

        if "write_gate_elem_count" in stats:
            primitives["armt/write/write_gate_mean"] = build_mean_metric(
                stats["write_gate_sum"],
                stats["write_gate_elem_count"],
            )

        if "W_mem_elem_count" in stats:
            primitives["armt/state/W_mem_norm"] = build_rms_metric(
                stats["W_mem_sq_sum"],
                stats["W_mem_elem_count"],
            )

        if self.use_denom and "z_elem_count" in stats:
            primitives["armt/state/z_norm"] = build_rms_metric(
                stats["z_sq_sum"],
                stats["z_elem_count"],
            )

        self.reset_monitoring_stats()
        return primitives

    def reset_memory(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """Mark memory state for reset; optionally initialize immediately."""
        # A reset starts a fresh sequence. If the previous microbatch left graph-bearing
        # tensors in the recurrent state, detach them here so the next microbatch does
        # not accidentally traverse a graph that has already been backpropagated.
        if self.W_mem.numel() != 0:
            self.W_mem = self.W_mem.detach()
        if self.use_denom and self.z.numel() != 0:
            self.z = self.z.detach()

        self._first_chunk = True
        self._pending_reset = True

        if batch_size is not None:
            if device is None:
                device = self.W_mq.weight.device
            self._maybe_initialize_memory(batch_size=batch_size, device=device)

    def _maybe_initialize_memory(self, batch_size: int, device: torch.device):
        dtype = self.W_mq.weight.dtype
        expected_mem_shape = (
            batch_size,
            self.n_heads,
            self.d_key // self.n_heads,
            self.head_dim,
        )
        expected_z_shape = (
            batch_size,
            self.n_heads,
            self.d_key // self.n_heads,
        )

        if self.W_mem.device != device or self.W_mem.dtype != dtype:
            self.W_mem = self.W_mem.to(device=device, dtype=dtype)
        if self.W_mem.numel() == 0 or self.W_mem.shape != expected_mem_shape:
            self.W_mem.resize_(expected_mem_shape)

        if self._pending_reset:
            self.W_mem.zero_()

        if self.use_denom:
            if self.z.device != device or self.z.dtype != dtype:
                self.z = self.z.to(device=device, dtype=dtype)
            if self.z.numel() == 0 or self.z.shape != expected_z_shape:
                self.z.resize_(expected_z_shape)
            if self._pending_reset:
                self.z.zero_()

        self._pending_reset = False

    def _to_batch_first(
        self, hidden_states: torch.Tensor, input_is_sbh: bool
    ) -> Tuple[torch.Tensor, bool]:
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {hidden_states.dim()}D")
        if not isinstance(input_is_sbh, bool):
            raise TypeError("input_is_sbh must be a bool")
        if input_is_sbh:
            return hidden_states.transpose(0, 1).contiguous(), True
        return hidden_states, False

    def _from_batch_first(self, hidden_states: torch.Tensor, input_is_sbh: bool) -> torch.Tensor:
        if input_is_sbh:
            return hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, dim // self.n_heads)
        return x.permute(0, 2, 1, 3).contiguous()

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, d_head = x.shape
        return x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * d_head)

    def _gather_if_tp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] == self.d_model:
            return hidden_states
        if parallel_state.model_parallel_is_initialized():
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                return tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)
        return hidden_states

    def _scatter_if_tp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] == self.d_model:
            return hidden_states
        if parallel_state.model_parallel_is_initialized():
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                return tensor_parallel.scatter_to_tensor_model_parallel_region(hidden_states)
        return hidden_states

    def associate(
        self, hidden_states: torch.Tensor, input_is_sbh: bool
    ) -> torch.Tensor:
        hidden_states, input_is_sbh = self._to_batch_first(
            hidden_states, input_is_sbh=input_is_sbh
        )
        hidden_states = self._gather_if_tp(hidden_states)
        if hidden_states.dtype != self.W_mq.weight.dtype:
            hidden_states = hidden_states.to(dtype=self.W_mq.weight.dtype)
        if self.input_pre_norm is not None:
            hidden_states = self.input_pre_norm(hidden_states)

        self._maybe_initialize_memory(
            batch_size=hidden_states.shape[0], device=hidden_states.device
        )
        should_track_read_metrics = not self._first_chunk

        if self._first_chunk:
            result = torch.zeros_like(hidden_states)
        else:
            q = self._to_heads(self.W_mq(hidden_states))
            mq = self.phi(q)
            if self.use_qk_norm:
                mq = F.normalize(mq, dim=-1, p=2.0)

            num = torch.einsum("bhsk,bhkd->bhsd", mq, self.W_mem)
            if self.use_denom:
                denom = torch.einsum("bhk,bhsk->bhs", self.z, mq)[..., None] + 1e-5
                result = num / denom
            else:
                result = num

            result = self.W_mo(self._from_heads(result))

        if should_track_read_metrics:
            hidden_norms = torch.linalg.vector_norm(hidden_states.float(), dim=-1)
            retrieved_norms = torch.linalg.vector_norm(result.float(), dim=-1)
            token_count = self._count_tensor(hidden_norms.numel(), hidden_states.device)
            self._accumulate_monitoring_stat("hidden_norm_sum", hidden_norms.sum())
            self._accumulate_monitoring_stat("retrieved_norm_sum", retrieved_norms.sum())
            self._accumulate_monitoring_stat("retrieved_norm_count", token_count)

        result = self._scatter_if_tp(result)
        return self._from_batch_first(result, input_is_sbh)

    def update_mem(
        self,
        mem_tokens: torch.Tensor,
        input_is_sbh: bool,
        input_already_pre_normed: bool = False,
    ):
        mem_tokens, input_is_sbh = self._to_batch_first(
            mem_tokens, input_is_sbh=input_is_sbh
        )
        mem_tokens = self._gather_if_tp(mem_tokens)
        if mem_tokens.dtype != self.W_mq.weight.dtype:
            mem_tokens = mem_tokens.to(dtype=self.W_mq.weight.dtype)
        if self.input_pre_norm is not None and not input_already_pre_normed:
            mem_tokens = self.input_pre_norm(mem_tokens)

        self._maybe_initialize_memory(batch_size=mem_tokens.shape[0], device=mem_tokens.device)

        # TBPTT: keep cross-chunk credit assignment to memory-write parameters while
        # preventing gradients from flowing into previous-chunk transformer activations.
        # This also avoids requiring `retain_graph=True` across chunk-wise backward.
        if self.tbptt_mode:
            mem_tokens = mem_tokens.detach()

        k = self._to_heads(self.W_mk(mem_tokens))
        mk = self.phi(k)
        if self.use_qk_norm:
            mk = F.normalize(mk, dim=-1, p=2.0)

        new_mv = self._to_heads(self.W_mv(mem_tokens))

        if not self._first_chunk:
            prev_W_mem = self.W_mem.detach() if self.tbptt_mode else self.W_mem
            num = torch.einsum("bhsk,bhkd->bhsd", mk, prev_W_mem)
            if self.use_denom:
                prev_z = self.z.detach() if self.tbptt_mode else self.z
                denom = torch.einsum("bhk,bhsk->bhs", prev_z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    mk_norm_sq = torch.linalg.norm(mk, dim=-1) ** 2
                    new_info_coef = 1 - (denom / mk_norm_sq[..., None])
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1.0
            else:
                prev_mv = num
                new_info_coef = 1.0
        else:
            prev_mv = torch.zeros_like(new_mv)
            new_info_coef = 1.0

        mv = new_mv - prev_mv

        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))
        if self.gating:
            associations = torch.einsum("bhsk,bhsd,bhsd->bhkd", mk, mv, mb)
        else:
            associations = torch.einsum("bhsk,bhsd,bhsx->bhkd", mk, mv, mb)

        self._accumulate_monitoring_stat("delta_mem_sq_sum", associations.float().square().sum())
        self._accumulate_monitoring_stat(
            "delta_mem_elem_count",
            self._count_tensor(associations.numel(), associations.device),
        )
        self._accumulate_monitoring_stat("write_gate_sum", mb.float().sum())
        self._accumulate_monitoring_stat(
            "write_gate_elem_count",
            self._count_tensor(mb.numel(), mb.device),
        )

        if self.tbptt_mode:
            # Avoid in-place updates on buffers that were used earlier in the forward,
            # which can invalidate autograd saved tensors (version counter mismatch).
            self.W_mem = self.W_mem.detach() + associations
            if self.use_denom:
                self.z = self.z.detach() + (new_info_coef * mk).sum(dim=-2)
        else:
            self.W_mem = self.W_mem + associations
            if self.use_denom:
                self.z = self.z + (new_info_coef * mk).sum(dim=-2)

        self._accumulate_monitoring_stat("W_mem_sq_sum", self.W_mem.float().square().sum())
        self._accumulate_monitoring_stat(
            "W_mem_elem_count",
            self._count_tensor(self.W_mem.numel(), self.W_mem.device),
        )
        if self.use_denom:
            self._accumulate_monitoring_stat("z_sq_sum", self.z.float().square().sum())
            self._accumulate_monitoring_stat(
                "z_elem_count",
                self._count_tensor(self.z.numel(), self.z.device),
            )

        self._first_chunk = False
