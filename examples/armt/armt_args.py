"""ARMT argument definitions and constraint checks."""

from examples.recurrent.recurrent_args import add_recurrent_args, validate_recurrent_constraints


def add_armt_args(parser):
    parser = add_recurrent_args(parser)
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
        "--armt-nu",
        type=int,
        default=3,
        help="DPFP expansion factor (output dim = 2 * nu * d_mem)",
    )
    group.add_argument(
        "--armt-use-denom",
        action="store_true",
        default=True,
        help="Use denominator normalization in retrieval",
    )
    group.add_argument(
        "--armt-no-denom",
        action="store_false",
        dest="armt_use_denom",
        help="Disable denominator normalization",
    )
    group.add_argument(
        "--armt-gating",
        action="store_true",
        default=False,
        help="Use gated memory updates",
    )
    group.add_argument(
        "--armt-correction",
        action="store_true",
        default=True,
        help="Apply correction term in delta updates",
    )

    return parser


def validate_armt_constraints(args):
    return validate_recurrent_constraints(
        args,
        model_name="ARMT",
        sequence_parallel_extra_tokens=args.num_mem_tokens,
        divisibility_expr="(recurrent_chunk_size + num_mem_tokens) % TP == 0",
    )
