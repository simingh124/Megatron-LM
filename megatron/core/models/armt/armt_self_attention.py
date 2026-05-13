"""ARMT self-attention with optional recurrent-window KV reuse."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.fusions.fused_softmax import SoftmaxOne
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType

from .monitoring import build_mean_metric
from .windowed_attention_utils import should_use_windowed_full_attention


def _get_flash_attn_func():
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "ARMT windowed full-attention backend 'flash_attn' requires flash-attn to be "
            "installed. Set --armt-windowed-full-attn-backend native to use the native path."
        ) from exc
    return flash_attn_func


class ARMTSelfAttention(SelfAttention):
    """Self-attention with a differentiable real-token KV cache for ARMT."""

    def __init__(
        self,
        config,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        num_mem_tokens: int = 16,
        num_read_mem_tokens: int = 0,
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
        armt_windowed_full_attn_backend: str = "native",
        armt_equal_window_full_attn_path: str = "legacy",
        armt_read_memory_mode: str = "none",
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.num_mem_tokens = num_mem_tokens
        self.num_read_mem_tokens = num_read_mem_tokens
        self.recurrent_chunk_size = recurrent_chunk_size
        self.full_attn_window_size = (
            full_attn_window_size if full_attn_window_size is not None else recurrent_chunk_size
        )
        self.armt_windowed_full_attn_backend = armt_windowed_full_attn_backend
        self.armt_equal_window_full_attn_path = armt_equal_window_full_attn_path
        self.armt_read_memory_mode = armt_read_memory_mode
        self._skip_read_memory_for_current_chunk = False
        self._collect_monitoring_for_current_iteration = True
        self._monitoring_stats = {}
        self._configure_windowed_full_attention_mode()
        self._current_chunk_start_position = 0
        self.reset_window_kv_cache()

    def set_current_chunk_start_position(self, position: int):
        self._current_chunk_start_position = int(position)

    def set_skip_read_memory_for_current_chunk(self, enabled: bool):
        self._skip_read_memory_for_current_chunk = bool(enabled)

    def set_collect_monitoring_for_current_iteration(self, enabled: bool):
        self._collect_monitoring_for_current_iteration = bool(enabled)

    def reset_monitoring_stats(self):
        self._monitoring_stats = {}

    def _configure_windowed_full_attention_mode(self):
        self._use_windowed_full_attention = should_use_windowed_full_attention(
            recurrent_chunk_size=self.recurrent_chunk_size,
            full_attn_window_size=self.full_attn_window_size,
            equal_window_full_attn_path=self.armt_equal_window_full_attn_path,
        )

    def reset_window_kv_cache(self):
        self._history_key_cache: List[Tensor] = []
        self._history_value_cache: List[Tensor] = []
        self._history_cache_num_tokens = 0
        self._current_chunk_start_position = 0

    def _current_read_mem_tokens(self) -> int:
        read_memory_mode = getattr(self, "armt_read_memory_mode", "none")
        num_read_mem_tokens = int(getattr(self, "num_read_mem_tokens", 0))
        skip_read_memory = bool(getattr(self, "_skip_read_memory_for_current_chunk", False))
        if read_memory_mode == "none" or num_read_mem_tokens == 0:
            return 0
        if skip_read_memory:
            return 0
        return num_read_mem_tokens

    def _accumulate_monitoring_stat(self, name: str, value: Tensor) -> None:
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(
                f"ARMTSelfAttention monitoring expects scalars, got {name}={tuple(value.shape)}"
            )
        value = value.reshape(()).to(dtype=torch.float32)
        current = self._monitoring_stats.get(name)
        if current is None:
            self._monitoring_stats[name] = value
        else:
            self._monitoring_stats[name] = current + value

    @staticmethod
    def _count_tensor(count: int, device: torch.device) -> Tensor:
        return torch.tensor(float(count), device=device, dtype=torch.float32)

    def _split_read_real_and_memory_tokens(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        read_tokens = self._current_read_mem_tokens()
        write_tokens = self.num_mem_tokens
        minimum_expected_tokens = read_tokens + write_tokens
        if tensor.size(0) < minimum_expected_tokens:
            raise ValueError(
                "ARMT windowed self-attention expects hidden states to contain memory tokens."
            )
        if write_tokens == 0:
            return tensor[:read_tokens], tensor[read_tokens:], tensor[:0]
        return (
            tensor[:read_tokens],
            tensor[read_tokens:-write_tokens],
            tensor[-write_tokens:],
        )

    def _select_history_kv(self, current_real_seq_len: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        self._ensure_history_cache_lists()

        if self._history_cache_num_tokens == 0:
            return None, None

        history_len = max(self.full_attn_window_size - current_real_seq_len, 0)
        if history_len == 0:
            return None, None

        remaining = min(history_len, self._history_cache_num_tokens)
        selected_keys = []
        selected_values = []
        for key_chunk, value_chunk in zip(
            reversed(self._history_key_cache),
            reversed(self._history_value_cache),
        ):
            if remaining <= 0:
                break
            if key_chunk.size(0) <= remaining:
                selected_keys.append(key_chunk)
                selected_values.append(value_chunk)
                remaining -= key_chunk.size(0)
                continue

            selected_keys.append(key_chunk[-remaining:])
            selected_values.append(value_chunk[-remaining:])
            remaining = 0

        selected_keys.reverse()
        selected_values.reverse()
        if len(selected_keys) == 1:
            return selected_keys[0], selected_values[0]

        return torch.cat(selected_keys, dim=0), torch.cat(selected_values, dim=0)

    def _update_history_kv_cache(self, current_real_key: Tensor, current_real_value: Tensor) -> None:
        if current_real_key.size(0) == 0:
            return

        self._ensure_history_cache_lists()
        self._history_key_cache.append(current_real_key)
        self._history_value_cache.append(current_real_value)
        self._history_cache_num_tokens += current_real_key.size(0)
        self._trim_history_cache(self.full_attn_window_size)

    def _ensure_history_cache_lists(self) -> None:
        if self._history_key_cache is None or self._history_value_cache is None:
            self._history_key_cache = []
            self._history_value_cache = []
            self._history_cache_num_tokens = 0
            return

        if isinstance(self._history_key_cache, torch.Tensor):
            self._history_key_cache = [self._history_key_cache]
        if isinstance(self._history_value_cache, torch.Tensor):
            self._history_value_cache = [self._history_value_cache]
        if not hasattr(self, "_history_cache_num_tokens"):
            self._history_cache_num_tokens = sum(
                chunk.size(0) for chunk in self._history_key_cache
            )

    def _trim_history_cache(self, max_tokens: Optional[int]) -> None:
        if max_tokens is None:
            return

        if max_tokens <= 0:
            self._history_key_cache.clear()
            self._history_value_cache.clear()
            self._history_cache_num_tokens = 0
            return

        while self._history_cache_num_tokens > max_tokens and self._history_key_cache:
            overflow = self._history_cache_num_tokens - max_tokens
            first_key_chunk = self._history_key_cache[0]
            first_value_chunk = self._history_value_cache[0]
            first_chunk_tokens = first_key_chunk.size(0)

            if overflow >= first_chunk_tokens:
                self._history_key_cache.pop(0)
                self._history_value_cache.pop(0)
                self._history_cache_num_tokens -= first_chunk_tokens
                continue

            self._history_key_cache[0] = first_key_chunk[overflow:]
            self._history_value_cache[0] = first_value_chunk[overflow:]
            self._history_cache_num_tokens -= overflow
            break

    def _build_rectangular_causal_mask(self, query_len: int, key_len: int, device) -> Tensor:
        history_len = key_len - query_len
        return torch.triu(
            torch.ones(query_len, key_len, device=device, dtype=torch.bool),
            diagonal=history_len + 1,
        )

    def _repeat_kv_for_gqa(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeats = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
        return key, value

    def _get_softmax_scale(self) -> float:
        if self.config.softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            softmax_scale = self.config.softmax_scale

        if self.config.apply_query_key_layer_scaling:
            softmax_scale /= self.layer_number

        return softmax_scale

    @staticmethod
    def _slice_attention_mask_for_query_range(
        attention_mask: Optional[Tensor],
        *,
        query_start: int,
        query_len: int,
    ) -> Optional[Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 4:
            raise ValueError(
                "ARMT read-prefix attention monitoring expects a 4D attention mask, "
                f"got shape {tuple(attention_mask.shape)}"
            )
        return attention_mask[:, :, query_start : query_start + query_len, :]

    @staticmethod
    def _build_context_query_causal_mask(
        *,
        query_start_position: int,
        query_len: int,
        key_len: int,
        history_len: int,
        device: torch.device,
    ) -> Tensor:
        query_positions = torch.arange(query_len, device=device) + query_start_position
        key_positions = torch.arange(key_len, device=device)
        max_visible_key = history_len + query_positions.unsqueeze(1)
        return key_positions.unsqueeze(0) > max_visible_key

    def _apply_rotary_pos_emb_to_query_key(
        self,
        query: Tensor,
        key: Tensor,
        *,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        if rotary_pos_emb is None:
            return query, key

        if not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        q_pos_emb, k_pos_emb = rotary_pos_emb
        if q_pos_emb is not None:
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=None,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        if k_pos_emb is not None:
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=None,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        return query, key

    def _build_attention_probs(
        self,
        query: Tensor,
        key: Tensor,
        *,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeats = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = key.repeat_interleave(repeats, dim=2)

        attention_scores = torch.einsum("tbhd,sbhd->bhts", query.float(), key.float())
        attention_scores = attention_scores * self._get_softmax_scale()

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.to(device=attention_scores.device, dtype=torch.bool),
                -10000.0,
            )

        softmax_offset = getattr(self.core_attention, "softmax_offset", None)
        if softmax_offset is None:
            return torch.softmax(attention_scores, dim=-1, dtype=torch.float32)

        return SoftmaxOne(
            dim=-1,
            denominator_offset=softmax_offset.to(attention_scores.device),
        )(attention_scores)

    def track_read_prefix_attention_mass(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        if not self._collect_monitoring_for_current_iteration:
            return

        read_tokens = self._current_read_mem_tokens()
        if read_tokens == 0:
            return

        qkv_output = self.get_query_key_value_tensors(
            hidden_states,
            None,
            output_gate=self.config.attention_output_gate,
            split_qkv=True,
        )
        if self.config.attention_output_gate:
            query, key, _value, _gate = qkv_output
        else:
            query, key, _value = qkv_output

        query, key = self._apply_rotary_pos_emb_to_query_key(
            query,
            key,
            rotary_pos_emb=rotary_pos_emb,
        )

        _, current_real_key, _ = self._split_read_real_and_memory_tokens(key)
        context_len = current_real_key.size(0)
        if context_len == 0:
            return

        history_len = 0
        key_for_monitoring = key
        if self._use_windowed_full_attention:
            history_key, _history_value = self._select_history_kv(current_real_key.size(0))
            if history_key is not None:
                history_len = history_key.size(0)
                key_for_monitoring = torch.cat([history_key, key], dim=0)
            monitor_mask = self._build_context_query_causal_mask(
                query_start_position=read_tokens,
                query_len=context_len,
                key_len=key_for_monitoring.size(0),
                history_len=history_len,
                device=query.device,
            ).unsqueeze(0).unsqueeze(0)
        else:
            monitor_mask = self._slice_attention_mask_for_query_range(
                attention_mask,
                query_start=read_tokens,
                query_len=context_len,
            )
            if monitor_mask is None:
                monitor_mask = self._build_context_query_causal_mask(
                    query_start_position=read_tokens,
                    query_len=context_len,
                    key_len=key_for_monitoring.size(0),
                    history_len=0,
                    device=query.device,
                ).unsqueeze(0).unsqueeze(0)

        context_queries = query[read_tokens : read_tokens + context_len]
        attention_probs = self._build_attention_probs(
            context_queries,
            key_for_monitoring,
            attention_mask=monitor_mask,
        )
        read_prefix_mass = attention_probs[..., history_len : history_len + read_tokens].sum(dim=-1)
        self._accumulate_monitoring_stat("read_prefix_attn_mass_sum", read_prefix_mass.sum())
        self._accumulate_monitoring_stat(
            "read_prefix_attn_mass_count",
            self._count_tensor(read_prefix_mass.numel(), read_prefix_mass.device),
        )

    @staticmethod
    def _reduce_monitoring_scalar_across_tp(value: Tensor) -> Tensor:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and parallel_state.model_parallel_is_initialized()
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            return value

        reduced = value.clone()
        torch.distributed.all_reduce(
            reduced,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        return reduced

    def consume_monitoring_primitives(self):
        if not self._collect_monitoring_for_current_iteration:
            self.reset_monitoring_stats()
            return {}

        stats = self._monitoring_stats
        primitives = {}
        if "read_prefix_attn_mass_count" in stats:
            attn_mass_sum = self._reduce_monitoring_scalar_across_tp(
                stats["read_prefix_attn_mass_sum"]
            )
            attn_mass_count = self._reduce_monitoring_scalar_across_tp(
                stats["read_prefix_attn_mass_count"]
            )
            primitives["armt/read_prefix/attn_mass_from_context_mean"] = build_mean_metric(
                attn_mass_sum,
                attn_mass_count,
            )

        self.reset_monitoring_stats()
        return primitives

    def _compute_windowed_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size = query.size(1)
        query_len = query.size(0)
        key_len = key.size(0)
        softmax_offset = getattr(self.core_attention, "softmax_offset", None)

        if self.armt_windowed_full_attn_backend == "flash_attn":
            if softmax_offset is not None:
                raise RuntimeError(
                    "ARMT windowed full-attention backend 'flash_attn' does not support "
                    "softmax_offset. Set --armt-windowed-full-attn-backend native to use "
                    "the native path."
                )
            if not query.is_cuda or not key.is_cuda or not value.is_cuda:
                raise RuntimeError(
                    "ARMT windowed full-attention backend 'flash_attn' requires CUDA tensors. "
                    "Set --armt-windowed-full-attn-backend native to use the native path."
                )
            supported_dtypes = (torch.float16, torch.bfloat16)
            if (
                query.dtype not in supported_dtypes
                or key.dtype not in supported_dtypes
                or value.dtype not in supported_dtypes
            ):
                raise RuntimeError(
                    "ARMT windowed full-attention backend 'flash_attn' requires fp16/bf16 "
                    "query, key, and value tensors. Set --armt-windowed-full-attn-backend "
                    "native to use the native path."
                )
            flash_attn_func = _get_flash_attn_func()

            # flash-attn supports the same bottom-right causal layout here without an explicit
            # rectangular mask, unlike PyTorch SDPA's flash backend for this seqlen_q != seqlen_k path.
            context = flash_attn_func(
                query.permute(1, 0, 2, 3).contiguous(),
                key.permute(1, 0, 2, 3).contiguous(),
                value.permute(1, 0, 2, 3).contiguous(),
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                softmax_scale=self._get_softmax_scale(),
                causal=True,
            )
            context = context.permute(1, 0, 2, 3).contiguous()
            return context.view(
                context.size(0),
                context.size(1),
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
            )
        if self.armt_windowed_full_attn_backend != "native":
            raise RuntimeError(
                "ARMT windowed full-attention backend must be either 'flash_attn' or 'native'."
            )

        key, value = self._repeat_kv_for_gqa(key, value)
        attn_mask = self._build_rectangular_causal_mask(query_len, key_len, query.device)

        if softmax_offset is None:
            context = F.scaled_dot_product_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                attn_mask=~attn_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self._get_softmax_scale(),
            )
            context = context.permute(2, 0, 1, 3).contiguous()
            return context.view(
                context.size(0),
                context.size(1),
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
            )

        query = query.reshape(query_len, batch_size * self.num_attention_heads_per_partition, -1)
        key = key.view(key_len, batch_size * self.num_attention_heads_per_partition, -1)

        attention_scores = torch.bmm(
            query.transpose(0, 1),
            key.transpose(0, 1).transpose(1, 2),
        )
        attention_scores = attention_scores.view(
            batch_size,
            self.num_attention_heads_per_partition,
            query_len,
            key_len,
        )
        attention_scores = attention_scores * self._get_softmax_scale()

        attention_scores = attention_scores.float().masked_fill(attn_mask, -10000.0)
        attention_probs = SoftmaxOne(
            dim=-1,
            denominator_offset=softmax_offset.to(attention_scores.device),
        )(attention_scores)

        attention_probs = attention_probs.to(dtype=query.dtype)
        attention_probs = F.dropout(
            attention_probs,
            p=self.config.attention_dropout,
            training=self.training,
        )

        output_size = (
            value.size(1),
            value.size(2),
            query_len,
            value.size(3),
        )
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context = torch.bmm(attention_probs, value.transpose(0, 1))
        context = context.view(*output_size)
        context = context.permute(2, 0, 1, 3).contiguous()
        return context.view(
            context.size(0),
            context.size(1),
            self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor]:
        if not self._use_windowed_full_attention:
            return super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        if inference_context is not None:
            raise NotImplementedError(
                "ARMT windowed full-attention v1 does not support inference KV caching."
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "ARMT windowed full-attention v1 does not support packed sequences."
            )
        if attention_bias is not None:
            raise NotImplementedError(
                "ARMT windowed full-attention v1 does not support attention bias."
            )

        del attention_mask, key_value_states, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin
        del packed_seq_params, sequence_len_offset, inference_params

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        qkv_output = self.get_query_key_value_tensors(
            hidden_states,
            None,
            output_gate=self.config.attention_output_gate,
            split_qkv=True,
        )
        gate = None
        if self.config.attention_output_gate:
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                    cu_seqlens=None,
                    mscale=_yarn_get_concentration_factor_from_config(self.config),
                    cp_group=self.pg_collection.cp,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=None,
                    mscale=_yarn_get_concentration_factor_from_config(self.config),
                    cp_group=self.pg_collection.cp,
                )

        _, current_real_key, _ = self._split_read_real_and_memory_tokens(key)
        _, current_real_value, _ = self._split_read_real_and_memory_tokens(value)
        history_key, history_value = self._select_history_kv(current_real_key.size(0))

        if history_key is not None and history_value is not None:
            key = torch.cat([history_key, key], dim=0)
            value = torch.cat([history_value, value], dim=0)

        core_attn_out = self._compute_windowed_attention(query, key, value)

        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        output, bias = self.linear_proj(core_attn_out)
        self._update_history_kv_cache(current_real_key, current_real_value)
        return output, bias
