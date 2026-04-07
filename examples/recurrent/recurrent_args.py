"""Shared recurrent training arguments and validation helpers."""

import argparse
from argparse import Namespace
from importlib.util import find_spec
from typing import Any


RECURRENT_LEGACY_ALIASES = {
    "recurrent_chunk_size": "armt_chunk_size",
    "recurrent_tbptt_mode": "armt_tbptt_mode",
}

RECURRENT_DEFAULTS = {
    "use_recurrent_model_schedule": False,
    "recurrent_chunk_size": 512,
    "full_attn_window_size": None,
    "armt_windowed_full_attn_backend": "native",
    "armt_equal_window_full_attn_path": "legacy",
    "recurrent_tbptt_mode": True,
    "num_mem_tokens": 16,
    "no_read_memory_from_first_chunk": False,
    "no_loss_from_first_chunk": False,
    "recurrent_memory_backend": "associative",
    "recurrent_gdn_use_fla_kernel": True,
    "recurrent_gdn_use_causal_conv1d": True,
    "recurrent_gdn_conv_kernel_size": 4,
    "recurrent_gdn_key_head_dim": None,
    "recurrent_gdn_value_head_dim": None,
    "recurrent_gdn_num_key_heads": None,
    "recurrent_gdn_num_value_heads": None,
    "recurrent_slot_num_slots": None,
    "recurrent_slot_num_heads": None,
    "recurrent_slot_head_dim": None,
    "recurrent_slot_read_attn_backend": "flash",
    "recurrent_mem_qk_norm": False,
    "recurrent_memory_input_pre_norm": False,
}


def add_recurrent_args(parser):
    group = parser.add_argument_group("recurrent", "Shared recurrent training arguments")

    group.add_argument(
        "--use-recurrent-model-schedule",
        dest="use_recurrent_model_schedule",
        action="store_true",
        default=RECURRENT_DEFAULTS["use_recurrent_model_schedule"],
        help="Enable the recurrent-model forward/backward schedule.",
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
        "--full-attn-window-size",
        type=int,
        default=RECURRENT_DEFAULTS["full_attn_window_size"],
        help=(
            "Number of real tokens visible to full attention within a recurrent chunk. "
            "Defaults to recurrent_chunk_size. Values larger than recurrent_chunk_size "
            "enable overlapping full-attention windows across chunks."
        ),
    )
    group.add_argument(
        "--armt-windowed-full-attn-backend",
        type=str,
        default=RECURRENT_DEFAULTS["armt_windowed_full_attn_backend"],
        choices=["flash_attn", "native"],
        help=(
            "Backend for ARMT decoupled windowed full attention when "
            "full_attn_window_size > recurrent_chunk_size."
        ),
    )
    group.add_argument(
        "--armt-equal-window-full-attn-path",
        type=str,
        default=RECURRENT_DEFAULTS["armt_equal_window_full_attn_path"],
        choices=["legacy", "window"],
        help=(
            "When full_attn_window_size == recurrent_chunk_size, keep the legacy TE attention "
            "path or explicitly force the ARMT windowed-attention path."
        ),
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
    group.add_argument(
        "--recurrent-memory-backend",
        choices=["associative", "gated_deltanet", "cross_attn_slots"],
        default=RECURRENT_DEFAULTS["recurrent_memory_backend"],
        help="Recurrent memory backend used inside ARMT layers.",
    )
    group.add_argument(
        "--recurrent-gdn-use-fla-kernel",
        dest="recurrent_gdn_use_fla_kernel",
        action="store_true",
        default=RECURRENT_DEFAULTS["recurrent_gdn_use_fla_kernel"],
        help="Use FLA gated delta kernel for recurrent GDN memory backend.",
    )
    group.add_argument(
        "--no-recurrent-gdn-use-fla-kernel",
        dest="recurrent_gdn_use_fla_kernel",
        action="store_false",
        help="Disable FLA gated delta kernel and use torch fallback.",
    )
    group.add_argument(
        "--recurrent-gdn-use-causal-conv1d",
        dest="recurrent_gdn_use_causal_conv1d",
        action="store_true",
        default=RECURRENT_DEFAULTS["recurrent_gdn_use_causal_conv1d"],
        help="Use causal_conv1d kernel for recurrent GDN backend.",
    )
    group.add_argument(
        "--no-recurrent-gdn-use-causal-conv1d",
        dest="recurrent_gdn_use_causal_conv1d",
        action="store_false",
        help="Disable causal_conv1d kernel and use nn.Conv1d fallback.",
    )
    group.add_argument(
        "--recurrent-gdn-conv-kernel-size",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_gdn_conv_kernel_size"],
        help="Depth-wise convolution kernel size for recurrent GDN backend.",
    )
    group.add_argument(
        "--recurrent-gdn-key-head-dim",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_gdn_key_head_dim"],
        help="Key head dimension for recurrent GDN backend.",
    )
    group.add_argument(
        "--recurrent-gdn-value-head-dim",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_gdn_value_head_dim"],
        help="Value head dimension for recurrent GDN backend.",
    )
    group.add_argument(
        "--recurrent-gdn-num-key-heads",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_gdn_num_key_heads"],
        help="Number of key heads for recurrent GDN backend.",
    )
    group.add_argument(
        "--recurrent-gdn-num-value-heads",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_gdn_num_value_heads"],
        help="Number of value heads for recurrent GDN backend.",
    )
    group.add_argument(
        "--recurrent-slot-num-slots",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_slot_num_slots"],
        help="Number of persistent memory slots for the cross_attn_slots backend.",
    )
    group.add_argument(
        "--recurrent-slot-num-heads",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_slot_num_heads"],
        help="Number of attention heads for the cross_attn_slots backend.",
    )
    group.add_argument(
        "--recurrent-slot-head-dim",
        type=int,
        default=RECURRENT_DEFAULTS["recurrent_slot_head_dim"],
        help="Optional per-head dimension for the cross_attn_slots backend.",
    )
    group.add_argument(
        "--recurrent-slot-read-attn-backend",
        choices=["sdpa", "flash"],
        default=RECURRENT_DEFAULTS["recurrent_slot_read_attn_backend"],
        help="Read-path attention backend for the cross_attn_slots backend.",
    )
    group.add_argument(
        "--recurrent-mem-qk-norm",
        action=argparse.BooleanOptionalAction,
        default=RECURRENT_DEFAULTS["recurrent_mem_qk_norm"],
        help="Apply backend-specific QK normalization inside the active recurrent memory backend.",
    )
    group.add_argument(
        "--recurrent-memory-input-pre-norm",
        action=argparse.BooleanOptionalAction,
        default=RECURRENT_DEFAULTS["recurrent_memory_input_pre_norm"],
        help="Apply input pre-norm before recurrent memory associate()/update_mem() projections.",
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

    if args.full_attn_window_size is None:
        args.full_attn_window_size = args.recurrent_chunk_size

    if args.full_attn_window_size < args.recurrent_chunk_size:
        raise ValueError(
            "full_attn_window_size must be greater than or equal to recurrent_chunk_size."
        )

    using_windowed_full_attention = (
        args.full_attn_window_size > args.recurrent_chunk_size
        or (
            args.full_attn_window_size == args.recurrent_chunk_size
            and args.armt_equal_window_full_attn_path == "window"
        )
    )
    if args.armt_windowed_full_attn_backend not in ("flash_attn", "native"):
        raise ValueError(
            "armt_windowed_full_attn_backend must be either 'flash_attn' or 'native'."
        )
    if args.armt_equal_window_full_attn_path not in ("legacy", "window"):
        raise ValueError(
            "armt_equal_window_full_attn_path must be either 'legacy' or 'window'."
        )
    if using_windowed_full_attention and args.recurrent_tbptt_mode:
        raise ValueError(
            "Decoupled ARMT full-attention windows currently require No-TBPTT "
            "(set --no-recurrent-tbptt-mode)."
        )
    if using_windowed_full_attention and getattr(args, "sequence_parallel", False):
        raise ValueError(
            "Decoupled ARMT full-attention windows do not support sequence_parallel yet."
        )
    if (
        using_windowed_full_attention
        and args.armt_windowed_full_attn_backend == "flash_attn"
        and find_spec("flash_attn") is None
    ):
        raise ImportError(
            "armt_windowed_full_attn_backend='flash_attn' requires flash-attn to be installed."
        )

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

    if args.recurrent_memory_backend == "gated_deltanet":
        required_fields = (
            "recurrent_gdn_key_head_dim",
            "recurrent_gdn_value_head_dim",
            "recurrent_gdn_num_key_heads",
            "recurrent_gdn_num_value_heads",
        )
        missing = [field for field in required_fields if getattr(args, field, None) is None]
        if missing:
            raise ValueError(
                "gated_deltanet backend requires explicit recurrent GDN hyperparameters: "
                + ", ".join(missing)
            )

        if args.recurrent_gdn_conv_kernel_size <= 0:
            raise ValueError("recurrent_gdn_conv_kernel_size must be > 0")
        if args.recurrent_gdn_key_head_dim <= 0:
            raise ValueError("recurrent_gdn_key_head_dim must be > 0")
        if args.recurrent_gdn_value_head_dim <= 0:
            raise ValueError("recurrent_gdn_value_head_dim must be > 0")
        if args.recurrent_gdn_num_key_heads <= 0:
            raise ValueError("recurrent_gdn_num_key_heads must be > 0")
        if args.recurrent_gdn_num_value_heads <= 0:
            raise ValueError("recurrent_gdn_num_value_heads must be > 0")
        if args.recurrent_gdn_num_value_heads % args.recurrent_gdn_num_key_heads != 0:
            raise ValueError(
                "recurrent_gdn_num_value_heads must be a multiple of recurrent_gdn_num_key_heads"
            )

        tp = args.tensor_model_parallel_size
        if args.recurrent_gdn_num_key_heads % tp != 0:
            raise ValueError("recurrent_gdn_num_key_heads must be a multiple of TP")
        if args.recurrent_gdn_num_value_heads % tp != 0:
            raise ValueError("recurrent_gdn_num_value_heads must be a multiple of TP")

        if args.recurrent_gdn_use_fla_kernel and find_spec("fla") is None:
            raise ImportError(
                "recurrent_gdn_use_fla_kernel=True requires flash-linear-attention to be "
                "installed."
            )
        if args.recurrent_gdn_use_causal_conv1d and find_spec("causal_conv1d") is None:
            raise ImportError(
                "recurrent_gdn_use_causal_conv1d=True requires causal_conv1d to be installed."
            )

    if args.recurrent_memory_backend == "cross_attn_slots":
        required_fields = (
            "recurrent_slot_num_slots",
            "recurrent_slot_num_heads",
        )
        missing = [field for field in required_fields if getattr(args, field, None) is None]
        if missing:
            raise ValueError(
                "cross_attn_slots backend requires explicit recurrent slot hyperparameters: "
                + ", ".join(missing)
            )

        if args.recurrent_slot_num_slots <= 0:
            raise ValueError("recurrent_slot_num_slots must be > 0")
        if args.recurrent_slot_num_heads <= 0:
            raise ValueError("recurrent_slot_num_heads must be > 0")

        hidden_size = getattr(args, "hidden_size", None)
        if args.recurrent_slot_head_dim is not None:
            if args.recurrent_slot_head_dim <= 0:
                raise ValueError("recurrent_slot_head_dim must be > 0")
        elif hidden_size is not None and hidden_size % args.recurrent_slot_num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by recurrent_slot_num_heads when head_dim is omitted"
            )

        if args.recurrent_slot_read_attn_backend not in ("sdpa", "flash"):
            raise ValueError(
                "recurrent_slot_read_attn_backend must be one of ('sdpa', 'flash')"
            )

    return args
