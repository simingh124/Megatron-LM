"""Shared recurrent training arguments and validation helpers."""

from argparse import Namespace
from typing import Any


RECURRENT_LEGACY_ALIASES = {
    "use_recurrent_tbptt": "use_armt_tbptt",
    "recurrent_chunk_size": "armt_chunk_size",
    "recurrent_tbptt_mode": "armt_tbptt_mode",
}

RECURRENT_DEFAULTS = {
    "use_recurrent_tbptt": False,
    "recurrent_chunk_size": 512,
    "recurrent_tbptt_mode": True,
    "num_mem_tokens": 16,
    "no_read_memory_from_first_chunk": False,
    "no_loss_from_first_chunk": False,
}


def add_recurrent_args(parser):
    group = parser.add_argument_group("recurrent", "Shared recurrent training arguments")

    group.add_argument(
        "--use-recurrent-tbptt",
        "--use-armt-tbptt",
        dest="use_recurrent_tbptt",
        action="store_true",
        default=RECURRENT_DEFAULTS["use_recurrent_tbptt"],
        help="Enable recurrent TBPTT schedule.",
    )
    group.add_argument(
        "--num-mem-tokens",
        type=int,
        default=RECURRENT_DEFAULTS["num_mem_tokens"],
        help="Number of recurrent memory tokens.",
    )
    group.add_argument(
        "--recurrent-chunk-size",
        "--armt-chunk-size",
        dest="recurrent_chunk_size",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_chunk_size"],
        help="Number of tokens per recurrent TBPTT chunk.",
    )
    group.add_argument(
        "--recurrent-tbptt-mode",
        "--armt-tbptt-mode",
        dest="recurrent_tbptt_mode",
        action="store_true",
        default=RECURRENT_DEFAULTS["recurrent_tbptt_mode"],
        help=(
            "Use truncated recurrent training: detach recurrent memory state between chunks "
            "and run backward independently for each chunk."
        ),
    )
    group.add_argument(
        "--no-recurrent-tbptt-mode",
        dest="recurrent_tbptt_mode",
        action="store_false",
        help=(
            "Disable truncated recurrent training: keep recurrent memory state connected across "
            "chunks and run a single backward after all chunks in the microbatch finish forward."
        ),
    )
    group.add_argument(
        "--no-read-memory-from-first-chunk",
        dest="no_read_memory_from_first_chunk",
        action="store_true",
        default=RECURRENT_DEFAULTS["no_read_memory_from_first_chunk"],
        help=(
            "Do not read recurrent memory on the first chunk. "
            "RMT skips prepending read-memory tokens; ARMT skips calling associate()."
        ),
    )
    group.add_argument(
        "--read-memory-from-first-chunk",
        dest="no_read_memory_from_first_chunk",
        action="store_false",
        help="Explicitly read recurrent memory on the first chunk.",
    )
    group.add_argument(
        "--no-loss-from-first-chunk",
        action="store_true",
        default=RECURRENT_DEFAULTS["no_loss_from_first_chunk"],
        help=(
            "Do not compute LM loss (loss_mask=0) on the first TBPTT chunk. "
            "Forward still runs so memory can be written."
        ),
    )
    return parser


def _get_attr(args: Namespace, name: str) -> Any:
    return getattr(args, name) if hasattr(args, name) else None


def get_recurrent_arg(args: Namespace, canonical_name: str, default: Any = None) -> Any:
    """Read a recurrent argument, falling back to legacy ARMT field names."""
    if hasattr(args, canonical_name):
        value = getattr(args, canonical_name)
        if value is not None:
            return value

    legacy_name = RECURRENT_LEGACY_ALIASES.get(canonical_name)
    if legacy_name and hasattr(args, legacy_name):
        value = getattr(args, legacy_name)
        if value is not None:
            return value

    if default is not None:
        return default
    return RECURRENT_DEFAULTS.get(canonical_name)


def normalize_recurrent_args(args: Namespace) -> Namespace:
    """Normalize canonical and legacy recurrent argument names onto the same namespace."""
    for canonical_name, default_value in RECURRENT_DEFAULTS.items():
        value = get_recurrent_arg(args, canonical_name, default_value)
        setattr(args, canonical_name, value)

    for canonical_name, legacy_name in RECURRENT_LEGACY_ALIASES.items():
        setattr(args, legacy_name, getattr(args, canonical_name))

    return args


def validate_recurrent_constraints(
    args: Namespace,
    *,
    model_name: str,
    sequence_parallel_extra_tokens: int,
    divisibility_expr: str,
) -> Namespace:
    args = normalize_recurrent_args(args)

    if args.pipeline_model_parallel_size != 1:
        raise ValueError(
            f"{model_name} only supports PP=1. Got PP={args.pipeline_model_parallel_size}"
        )
    if args.context_parallel_size != 1:
        raise ValueError(f"{model_name} only supports CP=1. Got CP={args.context_parallel_size}")

    if getattr(args, "sequence_parallel", False):
        tp = args.tensor_model_parallel_size
        if tp <= 1:
            raise ValueError("sequence_parallel=True requires TP>1")
        if args.seq_length % args.recurrent_chunk_size != 0:
            raise ValueError(
                "V1 SP safety-mode requires seq_length % recurrent_chunk_size == 0 "
                "(unless padding is implemented)"
            )
        if (args.recurrent_chunk_size + sequence_parallel_extra_tokens) % tp != 0:
            raise ValueError(f"V1 SP safety-mode requires {divisibility_expr}")

    if getattr(args, "position_embedding_type", None) not in ("rope", "yarn"):
        raise ValueError(
            f"{model_name} only supports RoPE/Yarn position embedding. "
            f"Got {args.position_embedding_type}"
        )

    # Megatron often keeps fp8 recipe defaults even when fp8 itself is disabled.
    if getattr(args, "fp8", None) or getattr(args, "fp8_format", None):
        raise ValueError(
            f"{model_name} supports TE modules in bf16/fp16 only; FP8 is not supported."
        )

    return args
