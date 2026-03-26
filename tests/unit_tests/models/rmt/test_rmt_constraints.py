import argparse
from argparse import Namespace

import pytest

from examples.rmt.rmt_args import add_rmt_args, validate_rmt_constraints


def _make_args(**overrides):
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        fp8=None,
        fp8_format=None,
        use_recurrent_model_schedule=True,
        position_embedding_type="rope",
        seq_length=2048,
        recurrent_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_constraint_pp_must_be_1():
    with pytest.raises(ValueError, match="PP=1"):
        validate_rmt_constraints(_make_args(pipeline_model_parallel_size=2))


def test_constraint_cp_must_be_1():
    with pytest.raises(ValueError, match="CP=1"):
        validate_rmt_constraints(_make_args(context_parallel_size=2))


def test_constraint_fp8_forbidden():
    with pytest.raises(ValueError, match="FP8"):
        validate_rmt_constraints(_make_args(fp8="e4m3"))


def test_constraint_chunk_size_divisibility():
    with pytest.raises(ValueError, match="2 \\* num_mem_tokens"):
        validate_rmt_constraints(
            _make_args(
                seq_length=2044,
                tensor_model_parallel_size=2,
                recurrent_chunk_size=511,
                sequence_parallel=True,
            )
        )


def test_valid_config_passes():
    validate_rmt_constraints(
        _make_args(
            tensor_model_parallel_size=2,
            recurrent_chunk_size=512,
            sequence_parallel=True,
        )
    )


def test_rmt_specific_flag_is_registered_on_rmt_args():
    parser = argparse.ArgumentParser()
    add_rmt_args(parser)
    args = parser.parse_args([])

    assert args.no_read_memory_from_first_chunk is False


def test_rmt_recurrent_schedule_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_rmt_args(parser)
    args = parser.parse_args(["--use-recurrent-model-schedule"])

    assert args.use_recurrent_model_schedule is True


def test_rmt_legacy_schedule_flag_is_rejected():
    parser = argparse.ArgumentParser()
    add_rmt_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--use-recurrent-tbptt"])


def test_rmt_shared_read_flag_can_be_explicitly_enabled():
    parser = argparse.ArgumentParser()
    add_rmt_args(parser)
    args = parser.parse_args(["--read-memory-from-first-chunk"])

    assert args.no_read_memory_from_first_chunk is False
