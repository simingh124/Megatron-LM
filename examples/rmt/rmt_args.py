"""RMT argument definitions and constraint checks."""

from examples.recurrent.recurrent_args import add_recurrent_args, validate_recurrent_constraints


def add_rmt_args(parser):
    parser = add_recurrent_args(parser)
    parser.set_defaults(no_read_memory_from_first_chunk=False)
    return parser


def validate_rmt_constraints(args):
    return validate_recurrent_constraints(
        args,
        model_name="RMT",
        sequence_parallel_extra_tokens=2 * args.num_mem_tokens,
        divisibility_expr="(recurrent_chunk_size + 2 * num_mem_tokens) % TP == 0",
    )
