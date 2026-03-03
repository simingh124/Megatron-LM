"""ARMT transformer layer with associative memory."""

from typing import Optional

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayer

from .associative_layer import AssociativeLayer


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

        # Step 4: Update Memory (SP safety mode)
        if getattr(self.config, "sequence_parallel", False):
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=tp_group
            )
            if input_is_sbh:
                mem_part = gathered[-self.num_mem_tokens :, :, :]
            else:
                mem_part = gathered[:, -self.num_mem_tokens :, :]
        else:
            if input_is_sbh:
                mem_part = hidden_states[-self.num_mem_tokens :, :, :]
            else:
                mem_part = hidden_states[:, -self.num_mem_tokens :, :]

        self.associative_layer.update_mem(mem_part, input_is_sbh=input_is_sbh)

        return hidden_states, context

    def reset_memory(self):
        self.associative_layer.reset_memory()
