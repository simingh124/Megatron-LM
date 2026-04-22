"""ARMT argument definitions and constraint checks."""

import argparse

from examples.recurrent.recurrent_args import add_recurrent_args, validate_recurrent_constraints


def add_armt_args(parser):
    parser = add_recurrent_args(parser)
    parser.set_defaults(no_read_memory_from_first_chunk=True)
    group = parser.add_argument_group("ARMT", "ARMT specific arguments")

    group.add_argument(
        "--armt-d-mem",
        dest="armt_d_mem",
        type=int,
        default=None,
        help="Memory key dimension (defaults to hidden_size)",
    )
    group.add_argument(
        "--armt-n-heads",
        type=int,
        default=1,
        help="Number of heads for memory operations",
    )
    group.add_argument(
        "--armt-head-dim",
        type=int,
        default=None,
        help=(
            "Optional per-head value/read dimension for associative memory. "
            "Defaults to hidden_size // armt_n_heads when omitted."
        ),
    )
    group.add_argument(
        "--armt-nu",
        type=int,
        default=3,
        help="DPFP expansion factor (output dim = 2 * nu * d_mem)",
    )
    group.add_argument(
        "--armt-use-denom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use denominator normalization in retrieval",
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
        help="Use gated memory updates",
    )
    group.add_argument(
        "--armt-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply correction term in delta updates",
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

    return parser


def validate_armt_constraints(args):
    if not hasattr(args, "no_read_memory_from_first_chunk"):
        args.no_read_memory_from_first_chunk = True
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

    if getattr(args, "armt_memory_write_source", "mem_tokens") != "mem_tokens":
        args.num_mem_tokens = 0

    return validate_recurrent_constraints(
        args,
        model_name="ARMT",
        sequence_parallel_extra_tokens=args.num_mem_tokens,
        divisibility_expr="(recurrent_chunk_size + num_mem_tokens) % TP == 0",
    )
