"""ARMT model built on top of GPTModel."""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt.gpt_model import GPTModel

from .armt_layer import ARMTLayer
from .monitoring import finalize_metric_primitives, merge_metric_primitives
from .windowed_attention_utils import should_use_windowed_full_attention


class ARMTModel(GPTModel):
    """GPTModel with associative memory tokens and ARMT layers."""

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        num_mem_tokens: int = 16,
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
        armt_equal_window_full_attn_path: str = "legacy",
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
        self.recurrent_chunk_size = recurrent_chunk_size or max_sequence_length
        self.full_attn_window_size = (
            full_attn_window_size
            if full_attn_window_size is not None
            else self.recurrent_chunk_size
        )
        self.armt_equal_window_full_attn_path = armt_equal_window_full_attn_path
        self._use_windowed_full_attention = should_use_windowed_full_attention(
            recurrent_chunk_size=self.recurrent_chunk_size,
            full_attn_window_size=self.full_attn_window_size,
            equal_window_full_attn_path=self.armt_equal_window_full_attn_path,
        )
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0

        init_std = getattr(config, "init_method_std", 0.02)
        self.memory_embeddings = nn.Parameter(
            torch.randn(num_mem_tokens, config.hidden_size) * init_std
        )

    def _armt_layers(self):
        for module in self.modules():
            if isinstance(module, ARMTLayer):
                yield module

    def get_memory_parameter_breakdown(self) -> list[tuple[str, int]]:
        if self.num_mem_tokens == 0:
            return []

        breakdown = [("memory_embeddings", self.memory_embeddings.numel())]
        for module_name, module in self.named_modules():
            if not isinstance(module, ARMTLayer):
                continue

            for backend_attr in ("associative_layer", "recurrent_memory_layer"):
                backend_module = getattr(module, backend_attr, None)
                if backend_module is None:
                    continue

                breakdown.append(
                    (
                        f"{module_name}.{backend_attr}",
                        sum(param.numel() for param in backend_module.parameters()),
                    )
                )

        return breakdown

    def get_memory_state_breakdown(self, batch_size: int = 1) -> list[tuple[str, int]]:
        if self.num_mem_tokens == 0:
            return []

        state_sizes = OrderedDict()
        for module in self.named_modules():
            _, module_instance = module
            if not isinstance(module_instance, ARMTLayer):
                continue

            for backend_attr in ("associative_layer", "recurrent_memory_layer"):
                backend_module = getattr(module_instance, backend_attr, None)
                if backend_module is None:
                    continue

                state_breakdown_getter = getattr(backend_module, "get_memory_state_breakdown", None)
                if not callable(state_breakdown_getter):
                    continue

                for name, count in state_breakdown_getter(batch_size=batch_size):
                    state_sizes[name] = state_sizes.get(name, 0) + count

        ordered_names = ("initial_slots", "W_mem")
        breakdown = [(name, state_sizes[name]) for name in ordered_names if name in state_sizes]
        breakdown.extend(
            (name, count) for name, count in state_sizes.items() if name not in ordered_names
        )
        return breakdown

    def set_skip_read_memory_for_current_chunk(self, enabled: bool):
        self._skip_read_memory_for_current_chunk = bool(enabled)
        for module in self._armt_layers():
            module.set_skip_read_memory_for_current_chunk(enabled)

    def set_current_chunk_is_first(self, enabled: bool):
        self._current_chunk_is_first = bool(enabled)
        for module in self._armt_layers():
            module.set_current_chunk_is_first(enabled)

    def set_current_chunk_start_position(self, position: int):
        self._current_chunk_start_position = int(position)
        for module in self._armt_layers():
            module.set_current_chunk_start_position(position)

    def reset_all_memory(self):
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        for module in self._armt_layers():
            module.reset_memory()

    def reset_all_monitoring_stats(self):
        for module in self._armt_layers():
            module.reset_monitoring_stats()

    def consume_all_monitoring_primitives(self):
        primitives = {}
        for module in self._armt_layers():
            merge_metric_primitives(primitives, module.consume_monitoring_primitives())
        return primitives

    def consume_all_monitoring_metrics(self):
        return finalize_metric_primitives(self.consume_all_monitoring_primitives())

    def _concat_memory_embeddings(self, decoder_input: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_input.shape[1]
        mem = self.memory_embeddings.unsqueeze(1).expand(-1, batch_size, -1)
        mem = mem.to(dtype=decoder_input.dtype, device=decoder_input.device)
        return torch.cat([decoder_input, mem], dim=0)

    def _concat_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = padding_mask.shape[0]
        mem_mask = torch.zeros(
            batch_size,
            self.num_mem_tokens,
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )
        return torch.cat([padding_mask, mem_mask], dim=1)

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
            raise NotImplementedError("ARMT v1 does not support packed sequences")

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

        # Recompute rotary embeddings for the new sequence length.
        rotary_offset = self._current_chunk_start_position if self._use_windowed_full_attention else 0
        if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
            rotary_seq_len = decoder_input.shape[0]
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                offset=rotary_offset,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
            )
        elif self.position_embedding_type == "yarn":
            rotary_seq_len = decoder_input.shape[0]
            rotary_pos_emb, _ = self.rotary_pos_emb(
                rotary_seq_len,
                offset=rotary_offset,
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

        new_seq_len = seq_len + self.num_mem_tokens
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
        new_mask[..., :seq_len, :seq_len] = attention_mask
        return new_mask

    def _strip_memory_tokens(self, hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
            gathered = gathered[:seq_len]
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                gathered, group=tp_group
            )
        else:
            hidden_states = hidden_states[:seq_len]
        return hidden_states

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
            raise NotImplementedError("ARMT v1 does not support packed sequences")

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
            **(extra_block_kwargs or {}),
        )

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
