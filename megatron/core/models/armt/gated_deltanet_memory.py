"""Gated DeltaNet-style recurrent memory backend for ARMT."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
from megatron.core.transformer.transformer_config import TransformerConfig

from .monitoring import build_mean_metric, build_ratio_metric, build_rms_metric
from .norm_utils import build_recurrent_norm

try:
    from fla.modules.l2norm import l2norm as fla_l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    fla_l2norm = None
    chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def is_fla_available() -> bool:
    return chunk_gated_delta_rule is not None


def is_causal_conv1d_available() -> bool:
    return causal_conv1d_fn is not None


_SUPPORTED_GDN_READ_MODES = ("buggy", "normal")
_READ_POSITION_MONITORING_PREFIX = "read_position"


class GatedDeltaNetMemory(nn.Module):
    """ARMT-compatible recurrent memory backend backed by gated delta rule."""

    def __init__(
        self,
        d_model: Optional[int] = None,
        num_mem_tokens: int = 16,
        conv_kernel_size: int = 4,
        key_head_dim: int = 64,
        value_head_dim: int = 64,
        num_key_heads: int = 16,
        num_value_heads: int = 16,
        use_fla_kernel: bool = True,
        use_causal_conv1d: bool = True,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        config: Optional[TransformerConfig] = None,
        tbptt_mode: bool = True,
        use_input_pre_norm: bool = True,
        log_read_position_metrics_to_tensorboard: bool = False,
        normalization: str = "LayerNorm",
        norm_epsilon: float = 1e-5,
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        *,
        hidden_size: Optional[int] = None,
        read_mode: str = "normal",
    ):
        super().__init__()

        if config is None:
            raise ValueError("config must be provided for GatedDeltaNetMemory")
        self.config = config

        if hidden_size is not None:
            d_model = hidden_size
        if d_model is None:
            d_model = self.config.hidden_size
        elif d_model != self.config.hidden_size:
            raise ValueError("d_model must match config.hidden_size for GatedDeltaNetMemory")
        if d_model is None:
            raise ValueError("d_model (or hidden_size) must be provided for GatedDeltaNetMemory")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if key_head_dim <= 0:
            raise ValueError("key_head_dim must be > 0")
        if value_head_dim <= 0:
            raise ValueError("value_head_dim must be > 0")
        if num_key_heads <= 0:
            raise ValueError("num_key_heads must be > 0")
        if num_value_heads <= 0:
            raise ValueError("num_value_heads must be > 0")
        if num_value_heads % num_key_heads != 0:
            raise ValueError("num_value_heads must be a multiple of num_key_heads")
        if A_init_range[0] < 0 or A_init_range[1] < A_init_range[0]:
            raise ValueError("A_init_range must satisfy 0 <= low <= high")
        if read_mode not in _SUPPORTED_GDN_READ_MODES:
            supported_modes = ", ".join(repr(mode) for mode in _SUPPORTED_GDN_READ_MODES)
            raise ValueError(
                f"read_mode must be one of {supported_modes}"
            )
        if use_fla_kernel and not is_fla_available():
            raise ImportError(
                "FLA is required when recurrent_gdn_use_fla_kernel=True, but it is not installed."
            )
        if use_causal_conv1d and not is_causal_conv1d_available():
            raise ImportError(
                "causal_conv1d is required when recurrent_gdn_use_causal_conv1d=True."
            )

        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.conv_kernel_size = conv_kernel_size
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.use_fla_kernel = use_fla_kernel
        self.use_causal_conv1d = use_causal_conv1d
        self.conv_init = conv_init
        self.use_qk_l2norm = use_qk_l2norm
        self.tbptt_mode = tbptt_mode
        self.use_input_pre_norm = use_input_pre_norm
        self.log_read_position_metrics_to_tensorboard = bool(
            log_read_position_metrics_to_tensorboard
        )
        self._collect_monitoring_for_current_iteration = True
        self.A_init_range = A_init_range
        self.read_mode = read_mode

        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.in_proj_dim = self.conv_dim + self.v_dim + self.num_value_heads * 2

        self.in_proj = nn.Linear(
            d_model,
            self.in_proj_dim,
            bias=False,
            dtype=self.config.params_dtype,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
            dtype=self.config.params_dtype,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, dtype=self.config.params_dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, dtype=self.config.params_dtype))
        self.out_norm = nn.RMSNorm(self.value_head_dim, eps=1e-6, dtype=self.config.params_dtype)
        self.out_proj = nn.Linear(self.v_dim, d_model, bias=False, dtype=self.config.params_dtype)

        self.input_pre_norm = None
        if self.use_input_pre_norm:
            self.input_pre_norm = build_recurrent_norm(
                d_model,
                normalization=normalization,
                eps=norm_epsilon,
                dtype=self.config.params_dtype,
            )

        self.reset_parameters()

        self.register_buffer("recurrent_state", torch.empty(0), persistent=False)
        self._first_chunk = True
        self._pending_reset = True
        self.reset_monitoring_stats()

    def reset_parameters(self):
        if self.config.perform_initialization:
            with torch.no_grad():
                self.config.init_method(self.in_proj.weight)
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                torch.ones(
                    self.num_value_heads,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=self.dt_bias.device,
                )
                A = torch.empty(
                    self.num_value_heads,
                    dtype=self.config.params_dtype,
                    device=self.A_log.device,
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))
                self.config.output_layer_init_method(self.out_proj.weight)

    def set_tbptt_mode(self, enabled: bool):
        self.tbptt_mode = enabled

    def set_collect_monitoring_for_current_iteration(self, enabled: bool):
        self._collect_monitoring_for_current_iteration = bool(enabled)

    def reset_monitoring_stats(self):
        self._monitoring_stats: Dict[str, torch.Tensor] = {}

    def _accumulate_monitoring_stat(self, name: str, value: torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(
                f"GatedDeltaNetMemory monitoring expects scalars, got {name}={tuple(value.shape)}"
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

    @staticmethod
    def _position_metric_tag(position: int) -> str:
        return f"pos_{position:04d}"

    def _accumulate_read_position_monitoring_stats(
        self,
        hidden_states: torch.Tensor,
        retrieved_states: torch.Tensor,
    ) -> None:
        if hidden_states.shape != retrieved_states.shape:
            raise ValueError(
                "GatedDeltaNetMemory read position monitoring expects hidden/retrieved "
                f"tensors with matching shapes, got {tuple(hidden_states.shape)} and "
                f"{tuple(retrieved_states.shape)}"
            )

        hidden_norms = torch.linalg.vector_norm(hidden_states.float(), dim=-1)
        retrieved_norms = torch.linalg.vector_norm(retrieved_states.float(), dim=-1)
        token_count = self._count_tensor(hidden_states.shape[0], hidden_states.device)

        for position in range(hidden_states.shape[1]):
            position_tag = self._position_metric_tag(position)
            stats_prefix = f"{_READ_POSITION_MONITORING_PREFIX}/{position_tag}"
            self._accumulate_monitoring_stat(
                f"{stats_prefix}/hidden_norm_sum",
                hidden_norms[:, position].sum(),
            )
            self._accumulate_monitoring_stat(
                f"{stats_prefix}/retrieved_norm_sum",
                retrieved_norms[:, position].sum(),
            )
            self._accumulate_monitoring_stat(f"{stats_prefix}/count", token_count)

    def _iter_read_monitoring_parts(
        self,
        hidden_states: torch.Tensor,
        retrieved_states: torch.Tensor,
    ) -> Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...]:
        if hidden_states.shape != retrieved_states.shape:
            raise ValueError(
                "GatedDeltaNetMemory read monitoring expects hidden/retrieved tensors "
                f"with matching shapes, got {tuple(hidden_states.shape)} and "
                f"{tuple(retrieved_states.shape)}"
            )

        memory_tokens = min(self.num_mem_tokens, hidden_states.shape[1])
        context_tokens = hidden_states.shape[1] - memory_tokens
        partitions = []
        if context_tokens > 0:
            partitions.append(
                (
                    "context",
                    hidden_states[:, :context_tokens, :],
                    retrieved_states[:, :context_tokens, :],
                )
            )
        if memory_tokens > 0:
            partitions.append(
                (
                    "memory",
                    hidden_states[:, context_tokens:, :],
                    retrieved_states[:, context_tokens:, :],
                )
            )
        return tuple(partitions)

    def _accumulate_read_monitoring_stats(
        self,
        hidden_states: torch.Tensor,
        retrieved_states: torch.Tensor,
    ) -> None:
        if self.log_read_position_metrics_to_tensorboard:
            self._accumulate_read_position_monitoring_stats(hidden_states, retrieved_states)

        for (
            partition_name,
            hidden_part,
            retrieved_part,
        ) in self._iter_read_monitoring_parts(
            hidden_states,
            retrieved_states,
        ):
            hidden_norms = torch.linalg.vector_norm(hidden_part.float(), dim=-1)
            retrieved_norms = torch.linalg.vector_norm(retrieved_part.float(), dim=-1)
            token_count = self._count_tensor(hidden_norms.numel(), hidden_part.device)
            self._accumulate_monitoring_stat(
                f"{partition_name}_hidden_norm_sum", hidden_norms.sum()
            )
            self._accumulate_monitoring_stat(
                f"{partition_name}_retrieved_norm_sum", retrieved_norms.sum()
            )
            self._accumulate_monitoring_stat(
                f"{partition_name}_retrieved_norm_count", token_count
            )

    def _prepare_memory_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.input_pre_norm is not None:
            return self.input_pre_norm(hidden_states)
        return hidden_states

    def get_memory_state_breakdown(self, batch_size: int = 1) -> list[tuple[str, int]]:
        recurrent_state_numel = (
            batch_size
            * self.num_value_heads
            * self.key_head_dim
            * self.value_head_dim
        )
        return [("W_mem", recurrent_state_numel)]

    def consume_monitoring_primitives(self):
        if not self._collect_monitoring_for_current_iteration:
            self.reset_monitoring_stats()
            return {}

        stats = self._monitoring_stats
        primitives = {}

        for partition_name in ("context", "memory"):
            retrieved_count_key = f"{partition_name}_retrieved_norm_count"
            if retrieved_count_key not in stats:
                continue
            primitives[f"armt/read/{partition_name}_retrieved_norm_mean"] = build_mean_metric(
                stats[f"{partition_name}_retrieved_norm_sum"],
                stats[retrieved_count_key],
            )
            primitives[
                f"armt/read/retrieved_to_{partition_name}_hidden_ratio"
            ] = build_ratio_metric(
                stats[f"{partition_name}_retrieved_norm_sum"],
                stats[f"{partition_name}_hidden_norm_sum"],
            )

        position_count_keys = sorted(
            key
            for key in stats
            if key.startswith(f"{_READ_POSITION_MONITORING_PREFIX}/") and key.endswith("/count")
        )
        for count_key in position_count_keys:
            position_tag = count_key.split("/")[1]
            stats_prefix = f"{_READ_POSITION_MONITORING_PREFIX}/{position_tag}"
            primitives[f"armt/read/retrieved_norm_mean/{position_tag}"] = build_mean_metric(
                stats[f"{stats_prefix}/retrieved_norm_sum"],
                stats[count_key],
            )
            primitives[f"armt/read/retrieved_to_hidden_ratio/{position_tag}"] = (
                build_ratio_metric(
                    stats[f"{stats_prefix}/retrieved_norm_sum"],
                    stats[f"{stats_prefix}/hidden_norm_sum"],
                )
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

        if "log_decay_elem_count" in stats:
            primitives["armt/state/log_decay_norm"] = build_rms_metric(
                stats["log_decay_sq_sum"],
                stats["log_decay_elem_count"],
            )

        self.reset_monitoring_stats()
        return primitives

    def reset_memory(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        if self.recurrent_state.numel() != 0:
            self.recurrent_state = self.recurrent_state.detach()

        self._first_chunk = True
        self._pending_reset = True

        if batch_size is not None:
            if device is None:
                device = self.in_proj.weight.device
            self._maybe_initialize_memory(batch_size=batch_size, device=device)

    def _maybe_initialize_memory(self, batch_size: int, device: torch.device):
        dtype = self.in_proj.weight.dtype
        expected_state_shape = (
            batch_size,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
        )

        needs_rebuild = (
            self.recurrent_state.numel() == 0
            or self.recurrent_state.shape != expected_state_shape
        )

        if needs_rebuild:
            self.recurrent_state = torch.zeros(expected_state_shape, device=device, dtype=dtype)
            self._pending_reset = False
            return

        if self.recurrent_state.device != device:
            # The gated delta rule may return a higher-precision recurrent state.
            # Preserve the accumulated memory when only the placement changes.
            self.recurrent_state = self.recurrent_state.to(device=device)

        if self._pending_reset:
            self.recurrent_state = self.recurrent_state.detach().zero_()
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

    def _apply_conv(self, qkv: torch.Tensor, *, allow_causal_kernel: bool) -> torch.Tensor:
        seq_len = qkv.shape[1]
        qkv = qkv.transpose(1, 2).contiguous()

        if self.use_causal_conv1d and allow_causal_kernel:
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            )
        else:
            qkv = F.silu(self.conv1d(qkv)[..., :seq_len])

        return qkv.transpose(1, 2).contiguous()

    def _project_hidden_states(
        self,
        hidden_states: torch.Tensor,
        *,
        allow_causal_kernel: bool,
        memory_input_states: Optional[torch.Tensor] = None,
    ):
        if memory_input_states is None:
            memory_input_states = self._prepare_memory_input(hidden_states)
        projected = self.in_proj(memory_input_states)
        qkv, gate, beta, alpha = torch.split(
            projected,
            [self.conv_dim, self.v_dim, self.num_value_heads, self.num_value_heads],
            dim=-1,
        )
        qkv = self._apply_conv(qkv, allow_causal_kernel=allow_causal_kernel)

        query, key, value = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        batch_size, seq_len = memory_input_states.shape[:2]

        query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        gate = gate.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        beta = beta.reshape(batch_size, seq_len, self.num_value_heads)
        alpha = alpha.reshape(batch_size, seq_len, self.num_value_heads)

        if self.use_qk_l2norm:
            if self.use_fla_kernel:
                query = fla_l2norm(query.contiguous())
                key = fla_l2norm(key.contiguous())
            else:
                query = F.normalize(query.float(), dim=-1, p=2.0).to(value.dtype)
                key = F.normalize(key.float(), dim=-1, p=2.0).to(value.dtype)

        if self.num_value_heads // self.num_key_heads > 1:
            repeat = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            alpha.contiguous(),
        )

    def _compute_decay(self, alpha: torch.Tensor) -> torch.Tensor:
        return -self.A_log.exp().float().view(1, 1, -1) * F.softplus(
            alpha.float() + self.dt_bias.float().view(1, 1, -1)
        )

    def _compute_g_and_beta(self, alpha: torch.Tensor, beta: torch.Tensor):
        decay = self._compute_decay(alpha)
        write_gate = beta.sigmoid()
        return decay.contiguous(), write_gate.contiguous()

    def _run_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
    ):
        if self.use_fla_kernel:
            if not query.is_cuda:
                raise RuntimeError("FLA gated delta rule requires CUDA tensors.")
            return chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
            )

        chunk_size = max(1, min(64, query.shape[1]))
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )

    def _apply_output_gate(self, core_attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = core_attn_out.dtype
        normalized = self.out_norm(core_attn_out.reshape(-1, core_attn_out.shape[-1]))
        gate = F.silu(gate.float().reshape(-1, gate.shape[-1])).to(dtype=normalized.dtype)
        normalized = normalized * gate
        normalized = normalized.to(dtype=x_dtype)
        normalized = normalized.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        return self.out_proj(normalized)

    def _associate_via_buggy_delta_rule(
        self,
        query: torch.Tensor,
        gate: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        g = self._compute_decay(alpha).contiguous()
        beta = torch.zeros_like(alpha, dtype=query.dtype).contiguous()
        zero_key = torch.zeros_like(query)
        zero_value = torch.zeros(
            query.shape[0],
            query.shape[1],
            self.num_value_heads,
            self.value_head_dim,
            device=query.device,
            dtype=query.dtype,
        )
        core_attn_out, _ = self._run_gated_delta_rule(
            query,
            zero_key,
            zero_value,
            g=g,
            beta=beta,
            initial_state=self.recurrent_state,
            output_final_state=False,
        )
        return self._apply_output_gate(core_attn_out, gate)

    def _associate_via_direct_state(
        self,
        query: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        read_query = query
        if read_query.dtype != self.recurrent_state.dtype:
            read_query = read_query.to(dtype=self.recurrent_state.dtype)
        core_attn_out = torch.einsum("bshk,bhkv->bshv", read_query, self.recurrent_state)
        core_attn_out = core_attn_out * (self.key_head_dim ** -0.5)
        core_attn_out = core_attn_out.to(dtype=query.dtype)
        return self._apply_output_gate(core_attn_out, gate)

    def associate(self, hidden_states: torch.Tensor, input_is_sbh: bool) -> torch.Tensor:
        hidden_states, input_is_sbh = self._to_batch_first(
            hidden_states, input_is_sbh=input_is_sbh
        )
        hidden_states = self._gather_if_tp(hidden_states)
        if hidden_states.dtype != self.in_proj.weight.dtype:
            hidden_states = hidden_states.to(dtype=self.in_proj.weight.dtype)
        memory_input_states = self._prepare_memory_input(hidden_states)

        self._maybe_initialize_memory(
            batch_size=hidden_states.shape[0], device=hidden_states.device
        )
        should_track_read_metrics = not self._first_chunk

        if self._first_chunk:
            result = torch.zeros_like(hidden_states)
        else:
            query, _, _, gate, _, alpha = self._project_hidden_states(
                hidden_states,
                allow_causal_kernel=True,
                memory_input_states=memory_input_states,
            )
            if self.read_mode == "buggy":
                result = self._associate_via_buggy_delta_rule(query, gate, alpha)
            else:
                result = self._associate_via_direct_state(query, gate)

        if should_track_read_metrics and self._collect_monitoring_for_current_iteration:
            self._accumulate_read_monitoring_stats(
                hidden_states,
                result,
            )

        result = self._scatter_if_tp(result)
        return self._from_batch_first(result, input_is_sbh)

    def update_mem(
        self,
        mem_tokens: torch.Tensor,
        input_is_sbh: bool,
        input_already_pre_normed: bool = False,
    ):
        mem_tokens, input_is_sbh = self._to_batch_first(mem_tokens, input_is_sbh=input_is_sbh)
        mem_tokens = self._gather_if_tp(mem_tokens)
        if mem_tokens.dtype != self.in_proj.weight.dtype:
            mem_tokens = mem_tokens.to(dtype=self.in_proj.weight.dtype)
        if input_already_pre_normed:
            memory_input_states = mem_tokens
        else:
            memory_input_states = self._prepare_memory_input(mem_tokens)

        self._maybe_initialize_memory(batch_size=mem_tokens.shape[0], device=mem_tokens.device)

        if self.tbptt_mode:
            memory_input_states = memory_input_states.detach()
            initial_state = self.recurrent_state.detach()
        else:
            initial_state = self.recurrent_state

        query, key, value, _, beta, alpha = self._project_hidden_states(
            mem_tokens,
            allow_causal_kernel=not self.tbptt_mode,
            memory_input_states=memory_input_states,
        )
        g, beta = self._compute_g_and_beta(alpha, beta)
        _, final_state = self._run_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
        )

        delta_state = final_state - initial_state
        self.recurrent_state = final_state

        if self._collect_monitoring_for_current_iteration:
            self._accumulate_monitoring_stat("delta_mem_sq_sum", delta_state.float().square().sum())
            self._accumulate_monitoring_stat(
                "delta_mem_elem_count",
                self._count_tensor(delta_state.numel(), delta_state.device),
            )
            self._accumulate_monitoring_stat("write_gate_sum", beta.float().sum())
            self._accumulate_monitoring_stat(
                "write_gate_elem_count",
                self._count_tensor(beta.numel(), beta.device),
            )
            self._accumulate_monitoring_stat(
                "W_mem_sq_sum", self.recurrent_state.float().square().sum()
            )
            self._accumulate_monitoring_stat(
                "W_mem_elem_count",
                self._count_tensor(self.recurrent_state.numel(), self.recurrent_state.device),
            )
            self._accumulate_monitoring_stat("log_decay_sq_sum", g.float().square().sum())
            self._accumulate_monitoring_stat(
                "log_decay_elem_count",
                self._count_tensor(g.numel(), g.device),
            )

        self._first_chunk = False
