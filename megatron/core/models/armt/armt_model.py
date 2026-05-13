"""ARMT model built on top of GPTModel."""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams

from .armt_layer import ARMTLayer
from .init_utils import init_parameter
from .monitoring import finalize_metric_primitives, merge_metric_primitives
from .windowed_attention_utils import should_use_windowed_full_attention


class ARMTModel(GPTModel):
    """GPTModel with associative memory tokens and ARMT layers."""

    _LAYER_METRICS_TO_KEEP_AGGREGATED_ONLY = frozenset(
        {
            "armt/read/retrieved_norm_mean",
            "armt/read/retrieved_to_hidden_ratio",
        }
    )

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        num_mem_tokens: int = 16,
        num_read_mem_tokens: int = 0,
        recurrent_chunk_size: Optional[int] = None,
        full_attn_window_size: Optional[int] = None,
        armt_equal_window_full_attn_path: str = "legacy",
        armt_read_memory_mode: str = "none",
        armt_read_memory_residual: bool = False,
        log_layer_metrics_to_tensorboard: bool = False,
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
        self.num_read_mem_tokens = num_read_mem_tokens
        self.armt_read_memory_mode = armt_read_memory_mode
        self.armt_read_memory_residual = bool(armt_read_memory_residual)
        self.recurrent_chunk_size = recurrent_chunk_size or max_sequence_length
        self.full_attn_window_size = (
            full_attn_window_size
            if full_attn_window_size is not None
            else self.recurrent_chunk_size
        )
        self.armt_equal_window_full_attn_path = armt_equal_window_full_attn_path
        self.log_layer_metrics_to_tensorboard = bool(log_layer_metrics_to_tensorboard)
        self._collect_monitoring_for_current_iteration = True
        self._use_windowed_full_attention = should_use_windowed_full_attention(
            recurrent_chunk_size=self.recurrent_chunk_size,
            full_attn_window_size=self.full_attn_window_size,
            equal_window_full_attn_path=self.armt_equal_window_full_attn_path,
        )
        self._skip_read_memory_for_current_chunk = False
        self._current_chunk_is_first = False
        self._current_chunk_start_position = 0
        actual_num_armt_layers = len(self._ordered_armt_layers())
        configured_num_armt_layers = int(getattr(config, "num_layers", actual_num_armt_layers))
        self._num_armt_layers = actual_num_armt_layers or configured_num_armt_layers

        if num_mem_tokens > 0:
            self.memory_embeddings = nn.Parameter(self._new_memory_parameter((num_mem_tokens,)))
            init_parameter(
                self.memory_embeddings,
                config.embedding_init_method,
                perform_initialization=config.perform_initialization,
            )
        else:
            self.register_parameter("memory_embeddings", None)

        if self._uses_read_prefix_mode():
            read_shape = (num_read_mem_tokens,)
            if self.armt_read_memory_mode == "per_layer":
                read_shape = (self._num_armt_layers, num_read_mem_tokens)
            self.read_memory_embeddings = nn.Parameter(self._new_memory_parameter(read_shape))
            init_parameter(
                self.read_memory_embeddings,
                config.embedding_init_method,
                perform_initialization=config.perform_initialization,
            )
        else:
            self.register_parameter("read_memory_embeddings", None)

    def _new_memory_parameter(self, prefix_shape: tuple[int, ...]) -> torch.Tensor:
        memory_device = None
        if not getattr(self.config, "use_cpu_initialization", False) and torch.cuda.is_available():
            memory_device = torch.cuda.current_device()
        return torch.empty(
            *prefix_shape,
            self.config.hidden_size,
            dtype=self.config.params_dtype,
            device=memory_device,
        )

    def _armt_layers(self):
        for module in self.modules():
            if isinstance(module, ARMTLayer):
                yield module

    def _ordered_armt_layers(self) -> list[ARMTLayer]:
        return sorted(
            list(self._armt_layers()),
            key=lambda module: int(getattr(module, "layer_number", 0)),
        )

    @staticmethod
    def _layer_monitoring_metric_name(metric_name: str, layer_number: int) -> str:
        layer_tag = f"layer_{layer_number:02d}"
        if metric_name.startswith("armt/"):
            return f"{metric_name}/{layer_tag}"
        return f"armt/{metric_name}/{layer_tag}"

    @staticmethod
    def _should_expand_layer_metric(metric_name: str) -> bool:
        return (
            "/pos_" not in metric_name
            and metric_name not in ARMTModel._LAYER_METRICS_TO_KEEP_AGGREGATED_ONLY
        )

    def _uses_read_prefix_mode(self) -> bool:
        return self.armt_read_memory_mode != "none" and self.num_read_mem_tokens > 0

    def _current_read_mem_tokens(self) -> int:
        if not self._uses_read_prefix_mode() or self._skip_read_memory_for_current_chunk:
            return 0
        return self.num_read_mem_tokens

    def _current_write_mem_tokens(self) -> int:
        if self.memory_embeddings is None:
            return 0
        return self.num_mem_tokens

    def _current_total_extra_tokens(self) -> int:
        return self._current_read_mem_tokens() + self._current_write_mem_tokens()

    def _current_context_bounds(self, seq_len: int) -> tuple[int, int]:
        context_start = self._current_read_mem_tokens()
        return context_start, context_start + seq_len

    def _get_first_layer_read_embedding_slice(self) -> Optional[torch.Tensor]:
        current_read_tokens = self._current_read_mem_tokens()
        if current_read_tokens == 0 or self.read_memory_embeddings is None:
            return None
        if self.armt_read_memory_mode == "per_layer":
            return self.read_memory_embeddings[0, :current_read_tokens]
        return self.read_memory_embeddings[:current_read_tokens]

    def _expand_token_embeddings(
        self,
        embeddings: Optional[torch.Tensor],
        decoder_input: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if embeddings is None:
            return None
        batch_size = decoder_input.shape[1]
        expanded = embeddings.unsqueeze(1).expand(-1, batch_size, -1)
        return expanded.to(dtype=decoder_input.dtype, device=decoder_input.device)

    def _concat_memory_embeddings(self, decoder_input: torch.Tensor) -> torch.Tensor:
        if self._current_total_extra_tokens() == 0:
            return decoder_input

        read_prefix = self._expand_token_embeddings(
            self._get_first_layer_read_embedding_slice(),
            decoder_input,
        )
        write_suffix = self._expand_token_embeddings(self.memory_embeddings, decoder_input)

        parts = []
        if read_prefix is not None:
            parts.append(read_prefix)
        parts.append(decoder_input)
        if write_suffix is not None:
            parts.append(write_suffix)
        return torch.cat(parts, dim=0)

    def _concat_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        read_tokens = self._current_read_mem_tokens()
        write_tokens = self._current_write_mem_tokens()
        if read_tokens == 0 and write_tokens == 0:
            return padding_mask

        batch_size = padding_mask.shape[0]
        mem_mask = torch.zeros(
            batch_size,
            read_tokens + write_tokens,
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )

        parts = []
        if read_tokens > 0:
            parts.append(mem_mask[:, :read_tokens])
        parts.append(padding_mask)
        if write_tokens > 0:
            parts.append(mem_mask[:, read_tokens:])
        return torch.cat(parts, dim=1)

    def _configure_layers_for_current_forward(self) -> None:
        current_read_tokens = self._current_read_mem_tokens()
        for layer_idx, layer in enumerate(self._ordered_armt_layers()):
            if self.armt_read_memory_mode == "shared" and current_read_tokens > 0:
                layer.set_current_layer_read_memory_embeddings(
                    self.read_memory_embeddings[:current_read_tokens]
                )
            elif self.armt_read_memory_mode == "per_layer" and current_read_tokens > 0:
                layer.set_current_layer_read_memory_embeddings(
                    self.read_memory_embeddings[layer_idx, :current_read_tokens]
                )
            else:
                layer.set_current_layer_read_memory_embeddings(None)

    def get_memory_parameter_breakdown(self) -> list[tuple[str, int]]:
        breakdown = []
        if self.memory_embeddings is not None and self.memory_embeddings.numel() > 0:
            breakdown.append(("memory_embeddings", self.memory_embeddings.numel()))
        if self.read_memory_embeddings is not None and self.read_memory_embeddings.numel() > 0:
            breakdown.append(("read_memory_embeddings", self.read_memory_embeddings.numel()))
        for module_name, module in self.named_modules():
            if not isinstance(module, ARMTLayer):
                continue

            backend_module = getattr(module, "recurrent_memory_layer", None)
            if backend_module is None:
                continue

            breakdown.append(
                (
                    f"{module_name}.recurrent_memory_layer",
                    sum(param.numel() for param in backend_module.parameters()),
                )
            )

        return breakdown

    def get_memory_state_breakdown(self, batch_size: int = 1) -> list[tuple[str, int]]:
        state_sizes = OrderedDict()
        for module in self.named_modules():
            _, module_instance = module
            if not isinstance(module_instance, ARMTLayer):
                continue

            backend_module = getattr(module_instance, "recurrent_memory_layer", None)
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

    def set_collect_monitoring_for_current_iteration(self, enabled: bool):
        self._collect_monitoring_for_current_iteration = bool(enabled)
        for module in self._armt_layers():
            module.set_collect_monitoring_for_current_iteration(enabled)

    def should_collect_monitoring_for_current_iteration(self) -> bool:
        return self._collect_monitoring_for_current_iteration

    def consume_all_monitoring_primitives(self):
        if not self._collect_monitoring_for_current_iteration:
            return {}

        primitives = {}
        for fallback_layer_idx, module in enumerate(self._armt_layers(), start=1):
            layer_number = getattr(module, "layer_number", fallback_layer_idx)
            layer_primitives = module.consume_monitoring_primitives()
            merge_metric_primitives(primitives, layer_primitives)
            if self.log_layer_metrics_to_tensorboard:
                for metric_name, primitive in layer_primitives.items():
                    if not self._should_expand_layer_metric(metric_name):
                        continue
                    primitives[
                        self._layer_monitoring_metric_name(metric_name, int(layer_number))
                    ] = primitive
        return primitives

    def consume_all_monitoring_metrics(self):
        return finalize_metric_primitives(self.consume_all_monitoring_primitives())

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
        if self._current_total_extra_tokens() == 0:
            return attention_mask

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
        context_start, context_end = self._current_context_bounds(seq_len)
        new_mask[..., context_start:context_end, context_start:context_end] = attention_mask
        return new_mask

    def _strip_memory_tokens(self, hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self._current_total_extra_tokens() == 0:
            return hidden_states
        context_start, context_end = self._current_context_bounds(seq_len)
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
            gathered = gathered[context_start:context_end]
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                gathered, group=tp_group
            )
        else:
            hidden_states = hidden_states[context_start:context_end]
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

        self._configure_layers_for_current_forward()
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
