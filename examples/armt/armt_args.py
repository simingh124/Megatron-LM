"""ARMT argument definitions and constraint checks."""

import argparse

from examples.recurrent.recurrent_args import add_recurrent_args, validate_recurrent_constraints
from megatron.core.models.armt.seq_mixers import (
    _SUPPORTED_SEQ_MIXER_ATTN_BACKENDS,
    _SUPPORTED_SEQ_MIXER_INITS,
    _SUPPORTED_SEQ_MIXER_TYPES,
)


def add_armt_args(parser):
    parser = add_recurrent_args(parser)
    parser.set_defaults(no_read_memory_from_first_chunk=True)
    group = parser.add_argument_group("ARMT", "ARMT specific arguments")

    group.add_argument(
        "--armt-d-mem",
        dest="armt_d_mem",
        type=int,
        default=None,
        help="Associative backend only: memory key dimension (defaults to hidden_size).",
    )
    group.add_argument(
        "--armt-n-heads",
        type=int,
        default=1,
        help="Associative backend only: number of heads for memory operations.",
    )
    group.add_argument(
        "--armt-head-dim",
        type=int,
        default=None,
        help=(
            "Associative backend only: optional per-head value/read dimension. "
            "Defaults to hidden_size // armt_n_heads when omitted."
        ),
    )
    group.add_argument(
        "--armt-nu",
        type=int,
        default=3,
        help="Associative backend only: DPFP expansion factor (output dim = 2 * nu * d_mem).",
    )
    group.add_argument(
        "--armt-use-denom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Associative backend only: use denominator normalization in retrieval.",
    )
    group.add_argument(
        "--armt-no-denom",
        action="store_false",
        dest="armt_use_denom",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--armt-gating",
        action="store_true",
        default=False,
        help="Associative backend only: use gated memory updates.",
    )
    group.add_argument(
        "--armt-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Associative backend only: apply correction term in delta updates.",
    )
    group.add_argument(
        "--armt-log-layer-metrics-to-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit per-layer ARMT TensorBoard metrics under armt/.../layer_XX in addition to "
            "the existing aggregated armt/* metrics. Requires TensorBoard logging to be enabled."
        ),
    )
    group.add_argument(
        "--armt-log-read-position-metrics-to-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit aggregated per-position ARMT read metrics under "
            "armt/read/*/pos_XXXX. Requires TensorBoard logging to be enabled."
        ),
    )
    group.add_argument(
        "--armt-log-read-chunk-metrics-to-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit chunk-scoped aggregated ARMT read metrics under "
            "armt/read/retrieved_norm_mean/chunk_XX and "
            "armt/read/retrieved_to_hidden_ratio/chunk_XX. "
            "Requires TensorBoard logging to be enabled."
        ),
    )
    group.add_argument(
        "--armt-memory-write-source",
        choices=("mem_tokens", "post_mlp_context", "post_attn_context", "pre_attn_context"),
        default="mem_tokens",
        help=(
            "Select the recurrent-memory write source. Non-mem_tokens modes disable ARMT "
            "memory-token concatenation and write context states instead."
        ),
    )
    group.add_argument(
        "--armt-seq-mixer-type",
        choices=_SUPPORTED_SEQ_MIXER_TYPES,
        default="none",
        help=(
            "Insert a chunk-internal token mixer between _resolve_write_source and "
            "memory_layer.update_mem. 'none' keeps the original behavior. The MLP variants "
            "share weights across hidden dim and require a fixed chunk length; 'attn' uses "
            "a bidirectional self-attention block."
        ),
    )
    group.add_argument(
        "--armt-seq-mixer-init",
        choices=_SUPPORTED_SEQ_MIXER_INITS,
        default="identity",
        help=(
            "Initialization strategy for the seq mixer. 'identity' aims to keep the mixer "
            "a no-op at step 0 (strict for mlp1 and attn+residual; approximate for mlp2 "
            "and attn-no-residual). 'megatron' uses config.init_method / "
            "config.output_layer_init_method like any other Megatron submodule."
        ),
    )
    group.add_argument(
        "--armt-seq-mixer-mlp-expansion",
        type=int,
        default=2,
        help="Hidden-dim expansion factor for --armt-seq-mixer-type=mlp2.",
    )
    group.add_argument(
        "--armt-seq-mixer-attn-num-heads",
        type=int,
        default=1,
        help="Number of attention heads for --armt-seq-mixer-type=attn.",
    )
    group.add_argument(
        "--armt-seq-mixer-attn-head-dim",
        type=int,
        default=None,
        help=(
            "Per-head dimension for --armt-seq-mixer-type=attn. Defaults to "
            "hidden_size // armt_seq_mixer_attn_num_heads when omitted."
        ),
    )
    group.add_argument(
        "--armt-seq-mixer-attn-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add an input residual connection around the attn seq mixer.",
    )
    group.add_argument(
        "--armt-seq-mixer-attn-prenorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a pre-norm before QKV projection inside the attn seq mixer.",
    )
    group.add_argument(
        "--armt-seq-mixer-attn-backend",
        choices=_SUPPORTED_SEQ_MIXER_ATTN_BACKENDS,
        default="flash",
        help="SDPA backend used by the attn seq mixer.",
    )

    return parser


def _validate_associative_args(args):
    armt_n_heads = getattr(args, "armt_n_heads", 1)
    if armt_n_heads <= 0:
        raise ValueError("armt_n_heads must be > 0")

    armt_head_dim = getattr(args, "armt_head_dim", None)
    if armt_head_dim is not None and armt_head_dim <= 0:
        raise ValueError("armt_head_dim must be > 0")

    hidden_size = getattr(args, "hidden_size", None)
    d_mem = getattr(args, "armt_d_mem", None)
    effective_d_mem = hidden_size if d_mem is None else d_mem
    if effective_d_mem is not None and effective_d_mem % armt_n_heads != 0:
        raise ValueError("armt_d_mem (or hidden_size when omitted) must be divisible by armt_n_heads")

    if armt_head_dim is None and hidden_size is not None and hidden_size % armt_n_heads != 0:
        raise ValueError("hidden_size must be divisible by armt_n_heads when armt_head_dim is omitted")


def _validate_seq_mixer_args(args):
    mixer_type = getattr(args, "armt_seq_mixer_type", "none")
    if mixer_type not in _SUPPORTED_SEQ_MIXER_TYPES:
        raise ValueError(
            f"armt_seq_mixer_type must be one of {_SUPPORTED_SEQ_MIXER_TYPES}, "
            f"got {mixer_type!r}"
        )
    if mixer_type == "none":
        return

    init_strategy = getattr(args, "armt_seq_mixer_init", "identity")
    if init_strategy not in _SUPPORTED_SEQ_MIXER_INITS:
        raise ValueError(
            f"armt_seq_mixer_init must be one of {_SUPPORTED_SEQ_MIXER_INITS}, "
            f"got {init_strategy!r}"
        )

    # MLP variants tie their weight shape to S_eff and therefore require
    # every chunk to be exactly recurrent_chunk_size tokens long.
    seq_length = getattr(args, "seq_length", None)
    chunk_size = getattr(args, "recurrent_chunk_size", None)
    if seq_length is not None and chunk_size is not None:
        if seq_length % chunk_size != 0:
            raise ValueError(
                "armt_seq_mixer_type != 'none' requires "
                "seq_length % recurrent_chunk_size == 0 so every chunk has the "
                f"same length; got seq_length={seq_length} chunk_size={chunk_size}"
            )

    write_source = getattr(args, "armt_memory_write_source", "mem_tokens")
    if mixer_type in ("mlp1", "mlp2"):
        if write_source == "mem_tokens":
            num_mem_tokens = getattr(args, "num_mem_tokens", 0)
            if num_mem_tokens <= 0:
                raise ValueError(
                    f"armt_seq_mixer_type={mixer_type!r} with "
                    "armt_memory_write_source='mem_tokens' requires num_mem_tokens > 0"
                )
        if mixer_type == "mlp2":
            expansion = getattr(args, "armt_seq_mixer_mlp_expansion", 2)
            if expansion <= 0:
                raise ValueError("armt_seq_mixer_mlp_expansion must be > 0")

    if mixer_type == "attn":
        num_heads = getattr(args, "armt_seq_mixer_attn_num_heads", 1)
        if num_heads <= 0:
            raise ValueError("armt_seq_mixer_attn_num_heads must be > 0")
        head_dim = getattr(args, "armt_seq_mixer_attn_head_dim", None)
        if head_dim is not None and head_dim <= 0:
            raise ValueError("armt_seq_mixer_attn_head_dim must be > 0")
        if head_dim is None:
            hidden_size = getattr(args, "hidden_size", None)
            if hidden_size is not None and hidden_size % num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by armt_seq_mixer_attn_num_heads "
                    "when armt_seq_mixer_attn_head_dim is omitted"
                )
        backend = getattr(args, "armt_seq_mixer_attn_backend", "flash")
        if backend not in _SUPPORTED_SEQ_MIXER_ATTN_BACKENDS:
            raise ValueError(
                f"armt_seq_mixer_attn_backend must be one of "
                f"{_SUPPORTED_SEQ_MIXER_ATTN_BACKENDS}, got {backend!r}"
            )


def validate_armt_constraints(args):
    if not hasattr(args, "no_read_memory_from_first_chunk"):
        args.no_read_memory_from_first_chunk = True

    recurrent_memory_backend = getattr(args, "recurrent_memory_backend", "associative")
    if recurrent_memory_backend == "associative":
        _validate_associative_args(args)

    if getattr(args, "armt_memory_write_source", "mem_tokens") != "mem_tokens":
        args.num_mem_tokens = 0

    _validate_seq_mixer_args(args)

    return validate_recurrent_constraints(
        args,
        model_name="ARMT",
        sequence_parallel_extra_tokens=args.num_mem_tokens,
        divisibility_expr="(recurrent_chunk_size + num_mem_tokens) % TP == 0",
    )
