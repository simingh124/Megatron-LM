"""Cross-attention slot-based recurrent memory backend for ARMT."""

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends.cuda import SDPAParams
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core import parallel_state, tensor_parallel

from .monitoring import build_mean_metric, build_ratio_metric, build_rms_metric
from .norm_utils import build_recurrent_norm

_ATTN_EPSILON = 1e-5
_ENTROPY_EPSILON = 1e-8
_READ_ATTN_BACKENDS = ("sdpa", "flash")


class CrossAttentionSlotMemory(nn.Module):
    """ARMT-compatible recurrent memory backend with persistent memory slots."""

    def __init__(
        self,
        d_model: Optional[int] = None,
        num_mem_tokens: int = 16,
        num_slots: int = 16,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        tbptt_mode: bool = True,
        read_attn_backend: str = "flash",
        use_qk_norm: bool = False,
        use_input_pre_norm: bool = False,
        normalization: str = "LayerNorm",
        norm_epsilon: float = 1e-5,
        *,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()

        if hidden_size is not None:
            d_model = hidden_size
        if d_model is None:
            raise ValueError("d_model (or hidden_size) must be provided for CrossAttentionSlotMemory")
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads when head_dim is omitted")
            head_dim = d_model // num_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if read_attn_backend not in _READ_ATTN_BACKENDS:
            raise ValueError(
                f"read_attn_backend must be one of {_READ_ATTN_BACKENDS}, got {read_attn_backend}"
            )

        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tbptt_mode = tbptt_mode
        self.read_attn_backend = read_attn_backend
        self.use_qk_norm = use_qk_norm
        self.use_input_pre_norm = use_input_pre_norm

        projection_size = self.num_heads * self.head_dim
        self.W_read_q = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_read_k = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_read_v = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_read_o = nn.Linear(projection_size, d_model, bias=False, dtype=dtype)

        self.W_write_q = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_write_k = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_write_v = nn.Linear(d_model, projection_size, bias=False, dtype=dtype)
        self.W_write_o = nn.Linear(projection_size, d_model, bias=False, dtype=dtype)
        self.W_gate = nn.Linear(2 * d_model, d_model, bias=True, dtype=dtype)
        self.slot_norm = nn.LayerNorm(d_model, eps=1e-5, dtype=dtype)

        self.input_pre_norm = None
        if self.use_input_pre_norm:
            self.input_pre_norm = build_recurrent_norm(
                d_model,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )

        self.read_q_norm = None
        self.read_k_norm = None
        self.write_q_norm = None
        self.write_k_norm = None
        if self.use_qk_norm:
            self.read_q_norm = build_recurrent_norm(
                self.head_dim,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )
            self.read_k_norm = build_recurrent_norm(
                self.head_dim,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )
            self.write_q_norm = build_recurrent_norm(
                self.head_dim,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )
            self.write_k_norm = build_recurrent_norm(
                self.head_dim,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=dtype,
            )

        self.initial_slots = nn.Parameter(torch.empty(num_slots, d_model, dtype=dtype))
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)

        self.register_buffer("mem_slots", torch.empty(0), persistent=False)
        self._first_chunk = True
        self._pending_reset = True
        self.reset_monitoring_stats()

    def get_memory_state_breakdown(self, batch_size: int = 1) -> list[tuple[str, int]]:
        del batch_size
        return [("initial_slots", self.initial_slots.numel())]

    def set_tbptt_mode(self, enabled: bool):
        self.tbptt_mode = enabled

    def reset_monitoring_stats(self):
        self._monitoring_stats: Dict[str, torch.Tensor] = {}

    def _accumulate_monitoring_stat(self, name: str, value: torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(
                f"CrossAttentionSlotMemory monitoring expects scalars, got {name}={tuple(value.shape)}"
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

        if "slot_elem_count" in stats:
            primitives["armt/state/slot_norm"] = build_rms_metric(
                stats["slot_sq_sum"],
                stats["slot_elem_count"],
            )

        if "slot_usage_entropy_count" in stats:
            primitives["armt/state/slot_usage_entropy"] = build_mean_metric(
                stats["slot_usage_entropy_sum"],
                stats["slot_usage_entropy_count"],
            )

        if "max_slot_mass_ratio_count" in stats:
            primitives["armt/state/max_slot_mass_ratio"] = build_mean_metric(
                stats["max_slot_mass_ratio_sum"],
                stats["max_slot_mass_ratio_count"],
            )

        self.reset_monitoring_stats()
        return primitives

    def reset_memory(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        if self.mem_slots.numel() != 0:
            self.mem_slots = self.mem_slots.detach()

        self._first_chunk = True
        self._pending_reset = True

        if batch_size is not None:
            if device is None:
                device = self.W_read_q.weight.device
            self._maybe_initialize_memory(batch_size=batch_size, device=device)

    def _maybe_initialize_memory(self, batch_size: int, device: torch.device):
        dtype = self.W_read_q.weight.dtype
        expected_shape = (batch_size, self.num_slots, self.d_model)

        if self.mem_slots.numel() == 0 or self.mem_slots.shape != expected_shape or self._pending_reset:
            self.mem_slots = (
                self.initial_slots.to(device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .clone()
            )
            self._pending_reset = False
            return

        if self.mem_slots.device != device or self.mem_slots.dtype != dtype:
            self.mem_slots = self.mem_slots.to(device=device, dtype=dtype)

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

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)

    @staticmethod
    def _apply_norm(x: torch.Tensor, norm: Optional[nn.Module]) -> torch.Tensor:
        if norm is None:
            return x
        return norm(x)

    def _flash_read_attention_context(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        if query.device.type != "cuda":
            raise RuntimeError("cross_attn_slots read_attn_backend=flash requires CUDA inputs")

        allowed_dtypes = (torch.float16, torch.bfloat16)
        if query.dtype not in allowed_dtypes:
            raise RuntimeError(
                "cross_attn_slots read_attn_backend=flash requires fp16 or bf16 query/key/value"
            )

        params = SDPAParams(query, key, value, None, 0.0, False, False)
        if not torch.backends.cuda.can_use_flash_attention(params, debug=False):
            raise RuntimeError(
                "cross_attn_slots read_attn_backend=flash is unavailable for the current "
                "device/input shape; use read_attn_backend=sdpa instead"
            )

        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

    def _read_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        context = nullcontext()
        if self.read_attn_backend == "flash":
            context = self._flash_read_attention_context(query, key, value)

        with context:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

    def associate(self, hidden_states: torch.Tensor, input_is_sbh: bool) -> torch.Tensor:
        hidden_states, input_is_sbh = self._to_batch_first(hidden_states, input_is_sbh=input_is_sbh)
        hidden_states = self._gather_if_tp(hidden_states)
        if hidden_states.dtype != self.W_read_q.weight.dtype:
            hidden_states = hidden_states.to(dtype=self.W_read_q.weight.dtype)
        if self.input_pre_norm is not None:
            hidden_states = self.input_pre_norm(hidden_states)

        self._maybe_initialize_memory(batch_size=hidden_states.shape[0], device=hidden_states.device)
        should_track_read_metrics = not self._first_chunk

        if self._first_chunk:
            result = torch.zeros_like(hidden_states)
        else:
            query = self._apply_norm(self._to_heads(self.W_read_q(hidden_states)), self.read_q_norm)
            key = self._apply_norm(self._to_heads(self.W_read_k(self.mem_slots)), self.read_k_norm)
            value = self._to_heads(self.W_read_v(self.mem_slots))

            retrieved = self._read_attention(query, key, value)
            result = self.W_read_o(self._from_heads(retrieved))

        if should_track_read_metrics:
            hidden_norms = torch.linalg.vector_norm(hidden_states.float(), dim=-1)
            retrieved_norms = torch.linalg.vector_norm(result.float(), dim=-1)
            token_count = self._count_tensor(hidden_norms.numel(), hidden_states.device)
            self._accumulate_monitoring_stat("hidden_norm_sum", hidden_norms.sum())
            self._accumulate_monitoring_stat("retrieved_norm_sum", retrieved_norms.sum())
            self._accumulate_monitoring_stat("retrieved_norm_count", token_count)

        result = self._scatter_if_tp(result)
        return self._from_batch_first(result, input_is_sbh)

    def update_mem(self, mem_tokens: torch.Tensor, input_is_sbh: bool):
        mem_tokens, input_is_sbh = self._to_batch_first(mem_tokens, input_is_sbh=input_is_sbh)
        mem_tokens = self._gather_if_tp(mem_tokens)
        if mem_tokens.dtype != self.W_write_q.weight.dtype:
            mem_tokens = mem_tokens.to(dtype=self.W_write_q.weight.dtype)
        if self.input_pre_norm is not None:
            mem_tokens = self.input_pre_norm(mem_tokens)

        self._maybe_initialize_memory(batch_size=mem_tokens.shape[0], device=mem_tokens.device)

        if self.tbptt_mode:
            mem_tokens = mem_tokens.detach()
            slot_state = self.mem_slots.detach()
        else:
            slot_state = self.mem_slots

        query = self._apply_norm(self._to_heads(self.W_write_q(mem_tokens)), self.write_q_norm)
        key = self._apply_norm(self._to_heads(self.W_write_k(slot_state)), self.write_k_norm)
        value = self._to_heads(self.W_write_v(mem_tokens))

        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim**0.5)
        write_weights = torch.softmax(scores, dim=-1)

        numerators = torch.einsum("bhtn,bhtd->bhnd", write_weights, value)
        denominators = write_weights.sum(dim=-2).unsqueeze(-1)
        delta_heads = numerators / (denominators + _ATTN_EPSILON)
        delta_slots = self.W_write_o(self._from_heads(delta_heads))

        gate_inputs = torch.cat([slot_state, delta_slots], dim=-1)
        write_gate = torch.sigmoid(self.W_gate(gate_inputs))
        updated_slots = (1.0 - write_gate) * slot_state + write_gate * delta_slots
        self.mem_slots = self.slot_norm(updated_slots)

        slot_delta = self.mem_slots - slot_state
        self._accumulate_monitoring_stat("delta_mem_sq_sum", slot_delta.float().square().sum())
        self._accumulate_monitoring_stat(
            "delta_mem_elem_count",
            self._count_tensor(slot_delta.numel(), slot_delta.device),
        )
        self._accumulate_monitoring_stat("write_gate_sum", write_gate.float().sum())
        self._accumulate_monitoring_stat(
            "write_gate_elem_count",
            self._count_tensor(write_gate.numel(), write_gate.device),
        )
        self._accumulate_monitoring_stat("slot_sq_sum", self.mem_slots.float().square().sum())
        self._accumulate_monitoring_stat(
            "slot_elem_count",
            self._count_tensor(self.mem_slots.numel(), self.mem_slots.device),
        )

        slot_mass = write_weights.sum(dim=-2)
        slot_prob = slot_mass / torch.clamp(slot_mass.sum(dim=-1, keepdim=True), min=_ENTROPY_EPSILON)
        slot_entropy = -(slot_prob * torch.log(slot_prob + _ENTROPY_EPSILON)).sum(dim=-1)
        max_slot_mass_ratio = slot_mass.max(dim=-1).values / torch.clamp(
            slot_mass.sum(dim=-1), min=_ENTROPY_EPSILON
        )
        self._accumulate_monitoring_stat("slot_usage_entropy_sum", slot_entropy.sum())
        self._accumulate_monitoring_stat(
            "slot_usage_entropy_count",
            self._count_tensor(slot_entropy.numel(), slot_entropy.device),
        )
        self._accumulate_monitoring_stat("max_slot_mass_ratio_sum", max_slot_mass_ratio.sum())
        self._accumulate_monitoring_stat(
            "max_slot_mass_ratio_count",
            self._count_tensor(max_slot_mass_ratio.numel(), max_slot_mass_ratio.device),
        )

        self._first_chunk = False
