"""Layer specs for ARMT layers."""

from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from .armt_layer import ARMTLayer


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
    nu: int = 3,
    use_denom: bool = True,
    gating: bool = False,
    correction: bool = True,
    tbptt_mode: bool = True,
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

    return ModuleSpec(
        module=ARMTLayer,
        submodules=base_spec.submodules,
        params={
            "num_mem_tokens": num_mem_tokens,
            "d_mem": d_mem,
            "armt_n_heads": armt_n_heads,
            "nu": nu,
            "use_denom": use_denom,
            "gating": gating,
            "correction": correction,
            "tbptt_mode": tbptt_mode,
        },
    )
