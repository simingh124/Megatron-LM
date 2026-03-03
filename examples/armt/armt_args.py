"""ARMT argument definitions and constraint checks."""


def add_armt_args(parser):
    group = parser.add_argument_group("ARMT", "ARMT specific arguments")

    group.add_argument(
        "--use-armt-tbptt",
        action="store_true",
        default=False,
        help="Enable ARMT TBPTT schedule",
    )
    group.add_argument(
        "--num-mem-tokens",
        type=int,
        default=16,
        help="Number of memory tokens per layer",
    )
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

    group.add_argument(
        "--armt-chunk-size",
        type=int,
        default=512,
        help="Number of tokens per TBPTT chunk",
    )
    group.add_argument(
        "--armt-tbptt-mode",
        action="store_true",
        default=True,
        help="Enable TBPTT (detach gradients across chunks)",
    )

    group.add_argument(
        "--no-loss-from-first-chunk",
        action="store_true",
        default=False,
        help=(
            "Do not compute LM loss (loss_mask=0) on the first TBPTT chunk. "
            "Forward still runs so memory can be written."
        ),
    )
    return parser


def validate_armt_constraints(args):
    if args.pipeline_model_parallel_size != 1:
        raise ValueError(
            f"ARMT only supports PP=1. Got PP={args.pipeline_model_parallel_size}"
        )
    if args.context_parallel_size != 1:
        raise ValueError(
            f"ARMT only supports CP=1. Got CP={args.context_parallel_size}"
        )

    if getattr(args, "sequence_parallel", False):
        tp = args.tensor_model_parallel_size
        if tp <= 1:
            raise ValueError("sequence_parallel=True requires TP>1")
        if args.seq_length % args.armt_chunk_size != 0:
            raise ValueError(
                "V1 SP safety-mode requires seq_length % armt_chunk_size == 0 "
                "(unless padding is implemented)"
            )
        if (args.armt_chunk_size + args.num_mem_tokens) % tp != 0:
            raise ValueError(
                "V1 SP safety-mode requires (armt_chunk_size + num_mem_tokens) % TP == 0"
            )

    if getattr(args, "position_embedding_type", None) not in ("rope", "yarn"):
        raise ValueError(
            f"ARMT only supports RoPE/Yarn position embedding. "
            f"Got {args.position_embedding_type}"
        )

    # FP8 is not supported for ARMT v1. Note: Megatron's global default for
    # `--fp8-recipe` is often non-None (e.g., "delayed") even when FP8 is disabled,
    # so we must gate this constraint on FP8 actually being enabled.
    if getattr(args, "fp8", None) or getattr(args, "fp8_format", None):
        raise ValueError("ARMT v1 supports TE modules in bf16/fp16 only; FP8 is not supported.")
