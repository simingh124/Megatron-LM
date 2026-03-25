import argparse
from argparse import Namespace

import pytest

from examples.armt.armt_args import add_armt_args, validate_armt_constraints
from examples.recurrent.recurrent_args import normalize_recurrent_args


def _make_args(**overrides):
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        fp8=None,
        fp8_format=None,
        use_recurrent_tbptt=True,
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
    args = _make_args(pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="PP=1"):
        validate_armt_constraints(args)


def test_constraint_cp_must_be_1():
    args = _make_args(context_parallel_size=2)
    with pytest.raises(ValueError, match="CP=1"):
        validate_armt_constraints(args)


def test_constraint_fp8_forbidden():
    args = _make_args(fp8="e4m3")
    with pytest.raises(ValueError, match="FP8"):
        validate_armt_constraints(args)


def test_constraint_chunk_size_divisibility():
    args = _make_args(
        tensor_model_parallel_size=2,
        recurrent_chunk_size=511,
        sequence_parallel=True,
    )
    with pytest.raises(ValueError, match="recurrent_chunk_size"):
        validate_armt_constraints(args)


def test_valid_config_passes():
    args = _make_args(
        tensor_model_parallel_size=2,
        recurrent_chunk_size=512,
        sequence_parallel=True,
    )
    validate_armt_constraints(args)


def test_legacy_recurrent_names_are_normalized():
    args = Namespace(
        use_armt_tbptt=True,
        armt_chunk_size=256,
        armt_tbptt_mode=False,
        num_mem_tokens=8,
    )
    normalize_recurrent_args(args)

    assert args.use_recurrent_tbptt is True
    assert args.recurrent_chunk_size == 256
    assert args.recurrent_tbptt_mode is False
    assert args.armt_chunk_size == 256
    assert args.armt_tbptt_mode is False


def test_legacy_yaml_like_namespace_passes_validation():
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        fp8=None,
        fp8_format=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=True,
    )
    validate_armt_constraints(args)
    assert args.recurrent_chunk_size == 512


def test_armt_defaults_skip_read_memory_on_first_chunk():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.no_read_memory_from_first_chunk is True


def test_armt_shared_read_flag_can_enable_first_chunk_read():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--read-memory-from-first-chunk"])

    assert args.no_read_memory_from_first_chunk is False


def test_armt_use_denom_defaults_to_true():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.armt_use_denom is True


def test_armt_use_denom_can_be_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--no-armt-use-denom"])

    assert args.armt_use_denom is False


def test_armt_no_denom_legacy_alias_still_works():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-no-denom"])

    assert args.armt_use_denom is False


def test_armt_correction_defaults_to_true():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.armt_correction is True


def test_armt_correction_can_be_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--no-armt-correction"])

    assert args.armt_correction is False


def test_armt_validation_sets_skip_read_default_for_legacy_namespaces():
    args = _make_args()
    validate_armt_constraints(args)

    assert args.no_read_memory_from_first_chunk is True
