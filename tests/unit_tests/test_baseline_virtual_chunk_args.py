import pytest

from megatron.training.arguments import parse_args, validate_args


def _build_base_args(**overrides):
    args = parse_args(ignore_unknown_args=True)
    args.num_layers = 4
    args.num_attention_heads = 4
    args.hidden_size = 16
    args.max_position_embeddings = 16
    args.seq_length = 8
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    return args


def test_validate_args_accepts_valid_baseline_virtual_chunk_size():
    args = validate_args(_build_base_args(baseline_virtual_chunk_size=4))

    assert args.baseline_virtual_chunk_size == 4


def test_validate_args_rejects_non_divisible_baseline_virtual_chunk_size():
    with pytest.raises(
        AssertionError,
        match="seq-length % baseline-virtual-chunk-size == 0",
    ):
        validate_args(_build_base_args(baseline_virtual_chunk_size=3))


def test_validate_args_rejects_too_large_baseline_virtual_chunk_size():
    with pytest.raises(
        AssertionError,
        match="less than or equal to seq-length",
    ):
        validate_args(_build_base_args(baseline_virtual_chunk_size=16))


def test_validate_args_rejects_context_parallel_baseline_virtual_chunk_size():
    with pytest.raises(
        AssertionError,
        match="context-parallel-size == 1",
    ):
        validate_args(
            _build_base_args(
                baseline_virtual_chunk_size=4,
                context_parallel_size=2,
                world_size=2,
            )
        )
