"""RMT model built on top of GPTModel."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.armt.monitoring import (
    build_mean_metric,
    build_ratio_of_means_metric,
    finalize_metric_primitives,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams

_MEM_TOKEN_COSINE_HIGH_THRESHOLD = 0.8


class RMTModel(GPTModel):
    """GPTModel with model-level recurrent memory tokens."""

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        num_mem_tokens: int = 16,
        tbptt_mode: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        self.num_mem_tokens = num_mem_tokens
        self.tbptt_mode = tbptt_mode
        self.memory_state: Optional[torch.Tensor] = None
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._monitoring_stats: Dict[str, torch.Tensor] = {}

        init_std = getattr(config, "init_method_std", 0.02)
        self.memory_embeddings = nn.Parameter(
            torch.randn(num_mem_tokens, config.hidden_size) * init_std
        )

    def reset_all_memory(self):
        self.memory_state = None
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False

    def reset_all_monitoring_stats(self):
        self._monitoring_stats = {}

    def set_skip_read_memory_for_current_chunk(self, enabled: bool):
        self._skip_read_memory_for_current_chunk = bool(enabled)

    def set_current_chunk_is_first(self, enabled: bool):
        self._current_chunk_is_first = bool(enabled)

    @staticmethod
    def _count_tensor(count: int, device: torch.device) -> torch.Tensor:
        return torch.tensor(float(count), device=device, dtype=torch.float32)

    def _accumulate_monitoring_stat(self, name: str, value: torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            raise ValueError(f"RMTModel monitoring expects scalars, got {name}={tuple(value.shape)}")

        value = value.reshape(()).to(dtype=torch.float32)
        current = self._monitoring_stats.get(name)
        if current is None:
            self._monitoring_stats[name] = value
        else:
            self._monitoring_stats[name] = current + value

    def _summarize_token_norms(
        self, hidden_states: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.numel() == 0:
            return None

        norms = torch.linalg.vector_norm(hidden_states.float(), dim=-1)
        return norms.sum(), self._count_tensor(norms.numel(), hidden_states.device)

    def _accumulate_norm_summary(
        self, prefix: str, summary: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        if summary is None:
            return

        sum_value, count_value = summary
        self._accumulate_monitoring_stat(f"{prefix}_sum", sum_value)
        self._accumulate_monitoring_stat(f"{prefix}_count", count_value)

    def _summarize_mem_token_cosine(
        self, hidden_states: torch.Tensor
    ) -> Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if hidden_states.numel() == 0:
            return None

        mem_tokens = hidden_states.transpose(0, 1)
        if mem_tokens.shape[1] < 2:
            return None

        normalized = F.normalize(mem_tokens.float(), dim=-1, p=2.0)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal_mask = ~torch.eye(
            mem_tokens.shape[1],
            device=cosine.device,
            dtype=torch.bool,
        )
        cosine = cosine.masked_select(off_diagonal_mask.unsqueeze(0)).view(mem_tokens.shape[0], -1)

        return (
            cosine.mean(dim=-1).sum(),
            cosine.max(dim=-1).values.sum(),
            cosine.min(dim=-1).values.sum(),
            (cosine > _MEM_TOKEN_COSINE_HIGH_THRESHOLD).float().mean(dim=-1).sum(),
            self._count_tensor(mem_tokens.shape[0], cosine.device),
        )

    def _accumulate_mem_token_cosine_summary(
        self,
        prefix: str,
        summary: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        if summary is None:
            return

        mean_sum, max_sum, min_sum, high_ratio_sum, count_value = summary
        stat_updates = {
            "mem_token_cosine_mean": mean_sum,
            "mem_token_cosine_max_mean": max_sum,
            "mem_token_cosine_min_mean": min_sum,
            "mem_token_cosine_gt_0p8_ratio_mean": high_ratio_sum,
        }
        for stat_name, stat_sum in stat_updates.items():
            self._accumulate_monitoring_stat(f"{prefix}_{stat_name}_sum", stat_sum)
            self._accumulate_monitoring_stat(f"{prefix}_{stat_name}_count", count_value)

    def _current_read_mem_tokens(self) -> int:
        if self.num_mem_tokens == 0 or self._skip_read_memory_for_current_chunk:
            return 0
        return self.num_mem_tokens

    def _current_write_mem_tokens(self) -> int:
        return self.num_mem_tokens

    def _current_total_extra_tokens(self) -> int:
        return self._current_read_mem_tokens() + self._current_write_mem_tokens()

    def _current_context_bounds(self, seq_len: int) -> Tuple[int, int]:
        context_start = self._current_read_mem_tokens()
        return context_start, context_start + seq_len

    def _split_hidden_states_for_current_chunk(
        self, hidden_states: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        read_tokens = self._current_read_mem_tokens()
        write_tokens = self._current_write_mem_tokens()
        context_start, context_end = self._current_context_bounds(seq_len)
        read_part = hidden_states[:read_tokens]
        context_part = hidden_states[context_start:context_end]
        write_part = hidden_states[context_end : context_end + write_tokens]
        return read_part, context_part, write_part

    def _gather_hidden_for_monitoring(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )

        if hidden_states.shape[-1] == self.config.hidden_size:
            return hidden_states

        if parallel_state.model_parallel_is_initialized():
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                return tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)

        return hidden_states

    def _update_monitoring_stats_from_hidden_states(
        self, hidden_states: torch.Tensor, seq_len: int
    ) -> None:
        read_part, context_part, write_part = self._split_hidden_states_for_current_chunk(
            hidden_states, seq_len
        )

        read_summary = self._summarize_token_norms(read_part)
        context_summary = self._summarize_token_norms(context_part)
        write_summary = self._summarize_token_norms(write_part)

        self._accumulate_norm_summary("context_all", context_summary)
        self._accumulate_norm_summary("write_all", write_summary)

        if write_summary is not None and context_summary is not None:
            self._accumulate_norm_summary("write_ctx_write", write_summary)
            self._accumulate_norm_summary("write_ctx_context", context_summary)

        if read_summary is None:
            return

        read_prefix = "read_first" if self._current_chunk_is_first else "read_rest"
        read_ctx_prefix = "read_ctx_first" if self._current_chunk_is_first else "read_ctx_rest"
        self._accumulate_norm_summary(read_prefix, read_summary)

        if context_summary is not None:
            self._accumulate_norm_summary(f"{read_ctx_prefix}_read", read_summary)
            self._accumulate_norm_summary(f"{read_ctx_prefix}_context", context_summary)

        if write_summary is not None:
            self._accumulate_norm_summary("read_write_read", read_summary)
            self._accumulate_norm_summary("read_write_write", write_summary)

    def _update_mem_token_cosine_monitoring_stats_from_hidden_states(
        self, hidden_states: torch.Tensor, seq_len: int
    ) -> None:
        read_part, _, write_part = self._split_hidden_states_for_current_chunk(hidden_states, seq_len)
        read_prefix = "read_first" if self._current_chunk_is_first else "read_rest"
        self._accumulate_mem_token_cosine_summary(
            read_prefix,
            self._summarize_mem_token_cosine(read_part),
        )
        self._accumulate_mem_token_cosine_summary(
            "write_all",
            self._summarize_mem_token_cosine(write_part),
        )

    def _add_mean_metric_if_available(
        self,
        primitives: Dict[str, object],
        stats: Dict[str, torch.Tensor],
        metric_name: str,
        prefix: str,
    ) -> None:
        sum_key = f"{prefix}_sum"
        count_key = f"{prefix}_count"
        if sum_key in stats and count_key in stats:
            primitives[metric_name] = build_mean_metric(stats[sum_key], stats[count_key])

    def _add_ratio_of_means_metric_if_available(
        self,
        primitives: Dict[str, object],
        stats: Dict[str, torch.Tensor],
        metric_name: str,
        numerator_prefix: str,
        denominator_prefix: str,
    ) -> None:
        numerator_sum_key = f"{numerator_prefix}_sum"
        numerator_count_key = f"{numerator_prefix}_count"
        denominator_sum_key = f"{denominator_prefix}_sum"
        denominator_count_key = f"{denominator_prefix}_count"
        if (
            numerator_sum_key in stats
            and numerator_count_key in stats
            and denominator_sum_key in stats
            and denominator_count_key in stats
        ):
            primitives[metric_name] = build_ratio_of_means_metric(
                stats[numerator_sum_key],
                stats[numerator_count_key],
                stats[denominator_sum_key],
                stats[denominator_count_key],
            )

    def consume_all_monitoring_primitives(self):
        stats = self._monitoring_stats
        primitives: Dict[str, object] = {}

        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_norm_mean",
            "read_rest",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_norm_mean_first_chunk",
            "read_first",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_mem_token_norm_mean",
            "write_all",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/context_token_norm_mean",
            "context_all",
        )
        self._add_ratio_of_means_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_ctx_norm_ratio",
            "read_ctx_rest_read",
            "read_ctx_rest_context",
        )
        self._add_ratio_of_means_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_ctx_norm_ratio_first_chunk",
            "read_ctx_first_read",
            "read_ctx_first_context",
        )
        self._add_ratio_of_means_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_ctx_norm_ratio",
            "write_ctx_write",
            "write_ctx_context",
        )
        self._add_ratio_of_means_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_write_norm_ratio",
            "read_write_read",
            "read_write_write",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_mean",
            "read_rest_mem_token_cosine_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_max_mean",
            "read_rest_mem_token_cosine_max_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_min_mean",
            "read_rest_mem_token_cosine_min_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean",
            "read_rest_mem_token_cosine_gt_0p8_ratio_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_mean_first_chunk",
            "read_first_mem_token_cosine_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_max_mean_first_chunk",
            "read_first_mem_token_cosine_max_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_min_mean_first_chunk",
            "read_first_mem_token_cosine_min_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean_first_chunk",
            "read_first_mem_token_cosine_gt_0p8_ratio_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_mem_token_cosine_mean",
            "write_all_mem_token_cosine_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_mem_token_cosine_max_mean",
            "write_all_mem_token_cosine_max_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_mem_token_cosine_min_mean",
            "write_all_mem_token_cosine_min_mean",
        )
        self._add_mean_metric_if_available(
            primitives,
            stats,
            "rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean",
            "write_all_mem_token_cosine_gt_0p8_ratio_mean",
        )

        self._monitoring_stats = {}
        return primitives

    def consume_all_monitoring_metrics(self):
        return finalize_metric_primitives(self.consume_all_monitoring_primitives())

    def _get_memory_state(self, decoder_input: torch.Tensor) -> Optional[torch.Tensor]:
        if self.num_mem_tokens == 0:
            return None

        batch_size = decoder_input.shape[1]
        if self.memory_state is None or self.memory_state.shape[1] != batch_size:
            memory_state = self.memory_embeddings.unsqueeze(1).expand(-1, batch_size, -1)
        else:
            memory_state = self.memory_state

        return memory_state.to(dtype=decoder_input.dtype, device=decoder_input.device)

    def _concat_memory_embeddings(self, decoder_input: torch.Tensor) -> torch.Tensor:
        memory_state = self._get_memory_state(decoder_input)
        if memory_state is None:
            return decoder_input

        parts = []
        if self._current_read_mem_tokens() > 0:
            parts.append(memory_state)
        parts.append(decoder_input)
        if self._current_write_mem_tokens() > 0:
            parts.append(memory_state)
        return torch.cat(parts, dim=0)

    def _concat_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        if self.num_mem_tokens == 0:
            return padding_mask

        batch_size = padding_mask.shape[0]
        mem_mask = torch.zeros(
            batch_size,
            self.num_mem_tokens,
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        parts = []
        if self._current_read_mem_tokens() > 0:
            parts.append(mem_mask)
        parts.append(padding_mask)
        if self._current_write_mem_tokens() > 0:
            parts.append(mem_mask)
        return torch.cat(parts, dim=1)

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if packed_seq_params is not None:
            raise NotImplementedError("RMT v1 does not support packed sequences")

        preproc_output = super()._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preproc_output[:6]
        rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None

        if decoder_input is None:
            return preproc_output

        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered = tensor_parallel.gather_from_sequence_parallel_region(
                decoder_input, group=tp_group
            )
            decoder_input = self._concat_memory_embeddings(gathered)
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input, group=tp_group
            )
            if padding_mask is not None:
                gathered_mask = tensor_parallel.gather_from_sequence_parallel_region(
                    padding_mask.transpose(0, 1), group=tp_group
                ).transpose(0, 1)
                gathered_mask = self._concat_padding_mask(gathered_mask)
                padding_mask = tensor_parallel.scatter_to_sequence_parallel_region(
                    gathered_mask.transpose(0, 1), group=tp_group
                ).transpose(0, 1)
        else:
            decoder_input = self._concat_memory_embeddings(decoder_input)
            if padding_mask is not None:
                padding_mask = self._concat_padding_mask(padding_mask)

        if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
            rotary_seq_len = decoder_input.shape[0]
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
            )
        elif self.position_embedding_type == "yarn":
            rotary_seq_len = decoder_input.shape[0]
            rotary_pos_emb, _ = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
            )

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        )
        if rotary_pos_cos_sin is not None:
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    def _adjust_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        if attention_mask is None:
            return None

        new_seq_len = seq_len + self._current_total_extra_tokens()
        causal_mask = torch.triu(
            torch.ones(
                new_seq_len,
                new_seq_len,
                device=attention_mask.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        new_mask = causal_mask
        while new_mask.dim() < attention_mask.dim():
            new_mask = new_mask.unsqueeze(0)

        new_mask = new_mask.expand(attention_mask.shape[:-2] + (new_seq_len, new_seq_len)).clone()
        start_idx, end_idx = self._current_context_bounds(seq_len)
        new_mask[..., start_idx:end_idx, start_idx:end_idx] = attention_mask
        return new_mask

    def _strip_memory_tokens(self, hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        start_idx, end_idx = self._current_context_bounds(seq_len)
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
            gathered = gathered[start_idx:end_idx]
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                gathered, group=tp_group
            )
        else:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states

    def _update_memory_state_from_hidden_states(self, hidden_states: torch.Tensor):
        if self.num_mem_tokens == 0:
            self.memory_state = None
            return

        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )

        memory_state = hidden_states[-self._current_write_mem_tokens() :]
        if self.tbptt_mode:
            memory_state = memory_state.detach()
        self.memory_state = memory_state

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if packed_seq_params is not None:
            raise NotImplementedError("RMT v1 does not support packed sequences")

        seq_len = input_ids.shape[1] if input_ids is not None else attention_mask.shape[-1]

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = preproc_output[:6]
        rotary_pos_cos_sin = preproc_output[6] if len(preproc_output) == 7 else None

        attention_mask = self._adjust_attention_mask(attention_mask, seq_len)

        block_kwargs = dict(extra_block_kwargs or {})
        if self.num_mem_tokens >= 2:
            def _layer_output_callback(_layer_number: int, layer_hidden_states: torch.Tensor) -> None:
                monitoring_hidden_states = self._gather_hidden_for_monitoring(layer_hidden_states)
                self._update_mem_token_cosine_monitoring_stats_from_hidden_states(
                    monitoring_hidden_states,
                    seq_len,
                )

            block_kwargs["layer_output_callback"] = _layer_output_callback

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            padding_mask=padding_mask,
            **block_kwargs,
        )

        monitoring_hidden_states = self._gather_hidden_for_monitoring(hidden_states)
        self._update_monitoring_stats_from_hidden_states(monitoring_hidden_states, seq_len)
        self._update_memory_state_from_hidden_states(hidden_states)
        hidden_states = self._strip_memory_tokens(hidden_states, seq_len)

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )
