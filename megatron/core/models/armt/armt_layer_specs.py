"""Layer specs for ARMT layers."""

from dataclasses import replace
from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec

from .armt_layer import ARMTLayer
from .armt_self_attention import ARMTSelfAttention


def get_armt_layer_spec(
    *,
    transformer_impl: str = "local",
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    num_mem_tokens: int = 16,
    d_mem: Optional[int] = None,
    armt_n_heads: int = 1,
    armt_head_dim: Optional[int] = None,
    nu: int = 3,
    use_denom: bool = True,
    gating: bool = False,
    correction: bool = True,
    tbptt_mode: bool = True,
    recurrent_chunk_size: Optional[int] = None,
    full_attn_window_size: Optional[int] = None,
    armt_windowed_full_attn_backend: str = "native",
    armt_equal_window_full_attn_path: str = "legacy",
    recurrent_memory_backend: str = "associative",
    recurrent_gdn_use_fla_kernel: bool = True,
    recurrent_gdn_use_causal_conv1d: bool = True,
    recurrent_gdn_conv_kernel_size: int = 4,
    recurrent_gdn_key_head_dim: Optional[int] = None,
    recurrent_gdn_value_head_dim: Optional[int] = None,
    recurrent_gdn_num_key_heads: Optional[int] = None,
    recurrent_gdn_num_value_heads: Optional[int] = None,
    recurrent_slot_num_slots: Optional[int] = None,
    recurrent_slot_num_heads: Optional[int] = None,
    recurrent_slot_head_dim: Optional[int] = None,
    recurrent_slot_read_attn_backend: str = "flash",
    recurrent_mem_qk_norm: bool = False,
    recurrent_memory_input_pre_norm: bool = False,
) -> ModuleSpec:
    if transformer_impl == "transformer_engine":
        base_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            fp8=fp8,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
        )
    elif transformer_impl == "inference_optimized":
        base_spec = get_gpt_layer_with_inference_spec(
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            qk_l2_norm=qk_l2_norm,
        )
    else:
        base_spec = get_gpt_layer_local_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            fp8=fp8,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=use_kitchen,
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )

    if (
        isinstance(base_spec.submodules.self_attention, ModuleSpec)
        and base_spec.submodules.self_attention.module is SelfAttention
    ):
        armt_self_attention_spec = replace(
            base_spec.submodules.self_attention,
            module=ARMTSelfAttention,
            params={
                **base_spec.submodules.self_attention.params,
                "num_mem_tokens": num_mem_tokens,
                "recurrent_chunk_size": recurrent_chunk_size,
                "full_attn_window_size": full_attn_window_size,
                "armt_windowed_full_attn_backend": armt_windowed_full_attn_backend,
                "armt_equal_window_full_attn_path": armt_equal_window_full_attn_path,
            },
        )
        base_spec = replace(
            base_spec,
            submodules=replace(base_spec.submodules, self_attention=armt_self_attention_spec),
        )

    return ModuleSpec(
        module=ARMTLayer,
        submodules=base_spec.submodules,
        params={
            "num_mem_tokens": num_mem_tokens,
            "d_mem": d_mem,
            "armt_n_heads": armt_n_heads,
            "armt_head_dim": armt_head_dim,
            "nu": nu,
            "use_denom": use_denom,
            "gating": gating,
            "correction": correction,
            "tbptt_mode": tbptt_mode,
            "recurrent_chunk_size": recurrent_chunk_size,
            "full_attn_window_size": full_attn_window_size,
            "armt_windowed_full_attn_backend": armt_windowed_full_attn_backend,
            "armt_equal_window_full_attn_path": armt_equal_window_full_attn_path,
            "recurrent_memory_backend": recurrent_memory_backend,
            "recurrent_gdn_use_fla_kernel": recurrent_gdn_use_fla_kernel,
            "recurrent_gdn_use_causal_conv1d": recurrent_gdn_use_causal_conv1d,
            "recurrent_gdn_conv_kernel_size": recurrent_gdn_conv_kernel_size,
            "recurrent_gdn_key_head_dim": recurrent_gdn_key_head_dim,
            "recurrent_gdn_value_head_dim": recurrent_gdn_value_head_dim,
            "recurrent_gdn_num_key_heads": recurrent_gdn_num_key_heads,
            "recurrent_gdn_num_value_heads": recurrent_gdn_num_value_heads,
            "recurrent_slot_num_slots": recurrent_slot_num_slots,
            "recurrent_slot_num_heads": recurrent_slot_num_heads,
            "recurrent_slot_head_dim": recurrent_slot_head_dim,
            "recurrent_slot_read_attn_backend": recurrent_slot_read_attn_backend,
            "recurrent_mem_qk_norm": recurrent_mem_qk_norm,
            "recurrent_memory_input_pre_norm": recurrent_memory_input_pre_norm,
        },
    )
