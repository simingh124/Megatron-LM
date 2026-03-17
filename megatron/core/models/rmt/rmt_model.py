"""RMT model built on top of GPTModel."""

from typing import Optional

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt.gpt_model import GPTModel


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

        init_std = getattr(config, "init_method_std", 0.02)
        self.memory_embeddings = nn.Parameter(
            torch.randn(num_mem_tokens, config.hidden_size) * init_std
        )

    def reset_all_memory(self):
        self.memory_state = None

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
        return torch.cat([memory_state, decoder_input, memory_state], dim=0)

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
        return torch.cat([mem_mask, padding_mask, mem_mask], dim=1)

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

        new_seq_len = seq_len + 2 * self.num_mem_tokens
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
        start_idx = self.num_mem_tokens
        end_idx = start_idx + seq_len
        new_mask[..., start_idx:end_idx, start_idx:end_idx] = attention_mask
        return new_mask

    def _strip_memory_tokens(self, hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        start_idx = self.num_mem_tokens
        end_idx = start_idx + seq_len
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

        memory_state = hidden_states[-self.num_mem_tokens :]
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
