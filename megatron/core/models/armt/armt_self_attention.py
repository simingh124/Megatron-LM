"""ARMT self-attention with optional recurrent-window KV reuse."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

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
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
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
        self.recurrent_chunk_size = recurrent_chunk_size
        self.full_attn_window_size = (
            full_attn_window_size if full_attn_window_size is not None else recurrent_chunk_size
        )
        self._use_windowed_full_attention = bool(
            self.recurrent_chunk_size is not None
            and self.full_attn_window_size is not None
            and self.full_attn_window_size > self.recurrent_chunk_size
        )
        self._current_chunk_start_position = 0
        self.reset_window_kv_cache()

    def set_current_chunk_start_position(self, position: int):
        self._current_chunk_start_position = int(position)

    def reset_window_kv_cache(self):
        self._history_key_cache: List[Tensor] = []
        self._history_value_cache: List[Tensor] = []
        self._history_cache_num_tokens = 0
        self._current_chunk_start_position = 0

    def _split_real_and_memory_tokens(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        if self.num_mem_tokens == 0:
            return tensor, tensor[:0]
        if tensor.size(0) < self.num_mem_tokens:
            raise ValueError(
                "ARMT windowed self-attention expects hidden states to contain memory tokens."
            )
        return tensor[:-self.num_mem_tokens], tensor[-self.num_mem_tokens :]

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

    def _compute_windowed_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        key, value = self._repeat_kv_for_gqa(key, value)

        batch_size = query.size(1)
        query_len = query.size(0)
        key_len = key.size(0)
        attn_mask = self._build_rectangular_causal_mask(query_len, key_len, query.device)
        softmax_offset = getattr(self.core_attention, "softmax_offset", None)

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

        current_real_key, _ = self._split_real_and_memory_tokens(key)
        current_real_value, _ = self._split_real_and_memory_tokens(value)
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
