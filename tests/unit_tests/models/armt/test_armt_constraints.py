import argparse
from argparse import Namespace
from unittest.mock import patch

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
        use_recurrent_model_schedule=True,
        position_embedding_type="rope",
        seq_length=2048,
        recurrent_chunk_size=512,
        full_attn_window_size=None,
        recurrent_tbptt_mode=True,
        num_mem_tokens=16,
        sequence_parallel=False,
        armt_windowed_full_attn_backend="native",
        armt_equal_window_full_attn_path="legacy",
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


def test_full_attn_window_defaults_to_chunk_size():
    args = _make_args(full_attn_window_size=None)

    validate_armt_constraints(args)

    assert args.full_attn_window_size == args.recurrent_chunk_size


def test_constraint_full_attn_window_must_not_be_smaller_than_chunk_size():
    args = _make_args(
        recurrent_chunk_size=512,
        full_attn_window_size=256,
    )

    with pytest.raises(ValueError, match="full_attn_window_size"):
        validate_armt_constraints(args)


def test_windowed_mode_rejects_tbptt():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=512,
        recurrent_tbptt_mode=True,
    )

    with pytest.raises(ValueError, match="No-TBPTT"):
        validate_armt_constraints(args)


def test_windowed_mode_rejects_sequence_parallel():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=512,
        recurrent_tbptt_mode=False,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )

    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_armt_constraints(args)


def test_recurrent_schedule_flag_is_preserved_during_normalization():
    args = Namespace(
        use_recurrent_model_schedule=True,
        armt_chunk_size=256,
        armt_tbptt_mode=False,
        num_mem_tokens=8,
    )
    normalize_recurrent_args(args)

    assert args.use_recurrent_model_schedule is True
    assert args.recurrent_chunk_size == 256
    assert args.recurrent_tbptt_mode is False
    assert args.armt_chunk_size == 256
    assert args.armt_tbptt_mode is False


def test_current_yaml_like_namespace_passes_validation():
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        fp8=None,
        fp8_format=None,
        use_recurrent_model_schedule=True,
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


def test_armt_recurrent_schedule_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--use-recurrent-model-schedule"])

    assert args.use_recurrent_model_schedule is True


def test_armt_full_attn_window_size_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--full-attn-window-size", "1024"])

    assert args.full_attn_window_size == 1024


def test_armt_windowed_full_attn_backend_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-windowed-full-attn-backend", "native"])

    assert args.armt_windowed_full_attn_backend == "native"


def test_armt_equal_window_full_attn_path_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-equal-window-full-attn-path", "window"])

    assert args.armt_equal_window_full_attn_path == "window"


def test_armt_legacy_schedule_flags_are_rejected():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--use-recurrent-tbptt"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--use-armt-tbptt"])


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


def test_gdn_requires_explicit_hyperparameters():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
    )
    with pytest.raises(ValueError, match="explicit recurrent GDN hyperparameters"):
        validate_armt_constraints(args)


def test_gdn_head_counts_must_match_tp_and_ratio():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
        recurrent_gdn_key_head_dim=64,
        recurrent_gdn_value_head_dim=64,
        recurrent_gdn_num_key_heads=3,
        recurrent_gdn_num_value_heads=5,
        tensor_model_parallel_size=2,
    )
    with pytest.raises(ValueError, match="multiple"):
        validate_armt_constraints(args)


def test_gdn_use_fla_requires_installed_package():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
        recurrent_gdn_key_head_dim=64,
        recurrent_gdn_value_head_dim=64,
        recurrent_gdn_num_key_heads=2,
        recurrent_gdn_num_value_heads=2,
        recurrent_gdn_use_fla_kernel=True,
        recurrent_gdn_use_causal_conv1d=False,
    )
    with patch("examples.recurrent.recurrent_args.find_spec", side_effect=lambda name: None):
        with pytest.raises(ImportError, match="flash-linear-attention"):
            validate_armt_constraints(args)


def test_gdn_use_causal_conv1d_requires_installed_package():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
        recurrent_gdn_key_head_dim=64,
        recurrent_gdn_value_head_dim=64,
        recurrent_gdn_num_key_heads=2,
        recurrent_gdn_num_value_heads=2,
        recurrent_gdn_use_fla_kernel=False,
        recurrent_gdn_use_causal_conv1d=True,
    )
    with patch(
        "examples.recurrent.recurrent_args.find_spec",
        side_effect=lambda name: None if name == "causal_conv1d" else object(),
    ):
        with pytest.raises(ImportError, match="causal_conv1d"):
            validate_armt_constraints(args)


def test_windowed_full_attn_backend_requires_valid_choice():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=512,
        recurrent_tbptt_mode=False,
        armt_windowed_full_attn_backend="invalid",
    )

    with pytest.raises(ValueError, match="armt_windowed_full_attn_backend"):
        validate_armt_constraints(args)


def test_windowed_flash_attn_backend_requires_installed_package():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=512,
        recurrent_tbptt_mode=False,
        armt_windowed_full_attn_backend="flash_attn",
    )

    with patch(
        "examples.recurrent.recurrent_args.find_spec",
        side_effect=lambda name: None if name == "flash_attn" else object(),
    ):
        with pytest.raises(ImportError, match="flash-attn"):
            validate_armt_constraints(args)


def test_windowed_native_backend_does_not_require_flash_attn_package():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=512,
        recurrent_tbptt_mode=False,
        armt_windowed_full_attn_backend="native",
    )

    with patch(
        "examples.recurrent.recurrent_args.find_spec",
        side_effect=lambda name: None if name == "flash_attn" else object(),
    ):
        validate_armt_constraints(args)


def test_equal_window_does_not_require_flash_attn_package():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=256,
        recurrent_tbptt_mode=True,
        armt_windowed_full_attn_backend="flash_attn",
    )

    with patch(
        "examples.recurrent.recurrent_args.find_spec",
        side_effect=lambda name: None if name == "flash_attn" else object(),
    ):
        validate_armt_constraints(args)


def test_equal_window_window_path_rejects_tbptt():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=256,
        recurrent_tbptt_mode=True,
        armt_equal_window_full_attn_path="window",
    )

    with pytest.raises(ValueError, match="No-TBPTT"):
        validate_armt_constraints(args)


def test_equal_window_window_path_requires_flash_attn_when_requested():
    args = _make_args(
        recurrent_chunk_size=256,
        full_attn_window_size=256,
        recurrent_tbptt_mode=False,
        armt_windowed_full_attn_backend="flash_attn",
        armt_equal_window_full_attn_path="window",
    )

    with patch(
        "examples.recurrent.recurrent_args.find_spec",
        side_effect=lambda name: None if name == "flash_attn" else object(),
    ):
        with pytest.raises(ImportError, match="flash-attn"):
            validate_armt_constraints(args)
