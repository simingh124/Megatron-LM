"""ARMT training entrypoint (experimental)."""

import os
import sys
from functools import partial

import torch

# Add the parent directory to the path to import from megatron
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.arguments import core_transformer_config_from_args

from pretrain_gpt import loss_func, train_valid_test_datasets_provider
from megatron.core.models.armt.armt_model import ARMTModel
from megatron.core.models.armt.armt_layer_specs import get_armt_layer_spec

from examples.armt.armt_args import add_armt_args, validate_armt_constraints


def _disable_incompatible_fusions_for_no_tbptt(args, config=None):
    """Disable fused kernels that are not safe with a cross-chunk retained graph."""
    if getattr(args, "recurrent_tbptt_mode", True):
        return config

    args.bias_dropout_fusion = False
    args.bias_swiglu_fusion = False
    args.bias_gelu_fusion = False

    if config is not None:
        config.bias_dropout_fusion = False
        config.bias_activation_fusion = False

    return config


def model_provider(
    pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None, **kwargs
):
    args = get_args()
    validate_armt_constraints(args)

    # ARMT introduces new parameters (e.g., memory embeddings and associative layer weights)
    # which do not exist in baseline GPT checkpoints. For `torch_dist` distributed checkpoints,
    # the default strictness ("assume_ok_unexpected") may fail the load because those new keys
    # are "unexpected" w.r.t. the checkpoint metadata. Switch to a logging strictness to
    # automatically drop unexpected keys while still loading the rest of the model weights.
    if (
        getattr(args, "ckpt_format", None) == "torch_dist"
        and getattr(args, "load", None)
        and getattr(args, "dist_ckpt_strictness", None) == "assume_ok_unexpected"
    ):
        args.dist_ckpt_strictness = "log_unexpected"

    _disable_incompatible_fusions_for_no_tbptt(args, config)

    if config is None:
        config = core_transformer_config_from_args(args)
    else:
        _disable_incompatible_fusions_for_no_tbptt(args, config)

    layer_spec = get_armt_layer_spec(
        transformer_impl=args.transformer_impl,
        num_experts=getattr(args, "num_experts", None),
        moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
        qk_layernorm=getattr(args, "qk_layernorm", False),
        multi_latent_attention=getattr(args, "multi_latent_attention", False),
        fp8=getattr(args, "fp8", None),
        moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
        normalization=getattr(args, "normalization", None),
        qk_l2_norm=getattr(args, "qk_l2_norm", False),
        use_kitchen=getattr(args, "use_kitchen", False),
        use_kitchen_attention=getattr(args, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(args, "kitchen_attention_backend", "sdpa"),
        num_mem_tokens=args.num_mem_tokens,
        d_mem=args.armt_d_mem,
        armt_n_heads=args.armt_n_heads,
        armt_head_dim=getattr(args, "armt_head_dim", None),
        nu=args.armt_nu,
        use_denom=args.armt_use_denom,
        gating=args.armt_gating,
        correction=args.armt_correction,
        tbptt_mode=args.recurrent_tbptt_mode,
        recurrent_memory_backend=args.recurrent_memory_backend,
        recurrent_gdn_use_fla_kernel=args.recurrent_gdn_use_fla_kernel,
        recurrent_gdn_use_causal_conv1d=args.recurrent_gdn_use_causal_conv1d,
        recurrent_gdn_conv_kernel_size=args.recurrent_gdn_conv_kernel_size,
        recurrent_gdn_key_head_dim=args.recurrent_gdn_key_head_dim,
        recurrent_gdn_value_head_dim=args.recurrent_gdn_value_head_dim,
        recurrent_gdn_num_key_heads=args.recurrent_gdn_num_key_heads,
        recurrent_gdn_num_value_heads=args.recurrent_gdn_num_value_heads,
        recurrent_gdn_read_mode=getattr(args, "recurrent_gdn_read_mode", "normal"),
        recurrent_slot_num_slots=args.recurrent_slot_num_slots,
        recurrent_slot_num_heads=args.recurrent_slot_num_heads,
        recurrent_slot_head_dim=args.recurrent_slot_head_dim,
        recurrent_slot_read_attn_backend=args.recurrent_slot_read_attn_backend,
        recurrent_mem_qk_norm=args.recurrent_mem_qk_norm,
        recurrent_memory_input_pre_norm=args.recurrent_memory_input_pre_norm,
        armt_memory_write_source=getattr(args, "armt_memory_write_source", "mem_tokens"),
        log_read_position_metrics_to_tensorboard=(
            getattr(args, "armt_log_read_position_metrics_to_tensorboard", False)
        ),
    )

    max_seq_length = getattr(args, "max_position_embeddings", args.seq_length)

    model = ARMTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=max_seq_length,
        num_mem_tokens=args.num_mem_tokens,
        log_layer_metrics_to_tensorboard=getattr(
            args, "armt_log_layer_metrics_to_tensorboard", False
        ),
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(args, "fp16_lm_cross_entropy", False),
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(
            args, "untie_embeddings_and_output_weights", False
        ),
        position_embedding_type=getattr(args, "position_embedding_type", "rope"),
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=getattr(args, "rotary_base", 10000),
        rope_scaling=getattr(args, "use_rope_scaling", False),
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    tokens = batch["tokens"]
    labels = batch.get("labels")
    loss_mask = batch.get("loss_mask")
    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    packed_seq_params = batch.get("packed_seq_params")

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )

    return output_tensor, partial(loss_func, loss_mask, model=model)


if __name__ == "__main__":
    if os.environ.get("ARMT_DETECT_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True, check_nan=False)

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_armt_args,
    )
