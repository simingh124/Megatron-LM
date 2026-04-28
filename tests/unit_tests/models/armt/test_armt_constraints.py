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
        hidden_size=256,
        armt_d_mem=None,
        armt_n_heads=1,
        armt_head_dim=None,
        fp8=None,
        fp8_format=None,
        use_recurrent_model_schedule=True,
        position_embedding_type="rope",
        seq_length=2048,
        recurrent_chunk_size=512,
        recurrent_tbptt_mode=True,
        num_mem_tokens=16,
        sequence_parallel=False,
        recurrent_memory_lr=None,
        min_lr=None,
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


def test_armt_legacy_schedule_flags_are_rejected():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--use-recurrent-tbptt"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--use-armt-tbptt"])


def test_armt_windowed_full_attention_flags_are_rejected():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)

    removed_flags = (
        ["--full-attn-window-size", "1024"],
        ["--armt-windowed-full-attn-backend", "native"],
        ["--armt-equal-window-full-attn-path", "window"],
    )
    for args in removed_flags:
        with pytest.raises(SystemExit):
            parser.parse_args(args)


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


def test_armt_layer_metrics_tensorboard_switch_defaults_to_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.armt_log_layer_metrics_to_tensorboard is False


def test_armt_layer_metrics_tensorboard_switch_can_be_enabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-log-layer-metrics-to-tensorboard"])

    assert args.armt_log_layer_metrics_to_tensorboard is True


def test_armt_read_position_metrics_tensorboard_switch_defaults_to_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.armt_log_read_position_metrics_to_tensorboard is False


def test_armt_read_position_metrics_tensorboard_switch_can_be_enabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-log-read-position-metrics-to-tensorboard"])

    assert args.armt_log_read_position_metrics_to_tensorboard is True


def test_armt_read_chunk_metrics_tensorboard_switch_defaults_to_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.armt_log_read_chunk_metrics_to_tensorboard is False


def test_armt_read_chunk_metrics_tensorboard_switch_can_be_enabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-log-read-chunk-metrics-to-tensorboard"])

    assert args.armt_log_read_chunk_metrics_to_tensorboard is True


def test_armt_head_dim_arg_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-head-dim", "12"])

    assert args.armt_head_dim == 12


def test_armt_memory_write_source_arg_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--armt-memory-write-source", "post_attn_context"])

    assert args.armt_memory_write_source == "post_attn_context"


def test_armt_validation_sets_skip_read_default_for_legacy_namespaces():
    args = _make_args()
    validate_armt_constraints(args)

    assert args.no_read_memory_from_first_chunk is True


def test_armt_non_mem_write_source_forces_zero_mem_tokens():
    args = _make_args(
        armt_memory_write_source="post_mlp_context",
        num_mem_tokens=16,
    )

    validate_armt_constraints(args)

    assert args.num_mem_tokens == 0


def test_gdn_requires_explicit_hyperparameters():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
    )
    with pytest.raises(ValueError, match="explicit recurrent GDN hyperparameters"):
        validate_armt_constraints(args)


def test_gdn_ignores_associative_head_constraints():
    args = _make_args(
        hidden_size=70,
        armt_n_heads=4,
        recurrent_memory_backend="gated_deltanet",
        recurrent_gdn_key_head_dim=64,
        recurrent_gdn_value_head_dim=64,
        recurrent_gdn_num_key_heads=2,
        recurrent_gdn_num_value_heads=2,
        recurrent_gdn_use_fla_kernel=False,
        recurrent_gdn_use_causal_conv1d=False,
    )

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


def test_cross_attn_slots_requires_explicit_hyperparameters():
    args = _make_args(
        recurrent_memory_backend="cross_attn_slots",
    )
    with pytest.raises(ValueError, match="explicit recurrent slot hyperparameters"):
        validate_armt_constraints(args)


def test_cross_attn_slots_ignores_associative_head_constraints():
    args = _make_args(
        hidden_size=70,
        armt_n_heads=4,
        recurrent_memory_backend="cross_attn_slots",
        recurrent_slot_num_slots=8,
        recurrent_slot_num_heads=4,
        recurrent_slot_head_dim=32,
    )

    validate_armt_constraints(args)


def test_cross_attn_slots_rejects_non_positive_hyperparameters():
    args = _make_args(
        recurrent_memory_backend="cross_attn_slots",
        recurrent_slot_num_slots=0,
        recurrent_slot_num_heads=0,
    )
    with pytest.raises(ValueError, match="recurrent_slot_num_slots must be > 0"):
        validate_armt_constraints(args)


def test_cross_attn_slots_allows_explicit_head_dim_without_hidden_size_match():
    args = _make_args(
        recurrent_memory_backend="cross_attn_slots",
        recurrent_slot_num_slots=8,
        recurrent_slot_num_heads=4,
        recurrent_slot_head_dim=32,
        hidden_size=96,
    )
    validate_armt_constraints(args)


def test_cross_attn_slots_can_infer_head_dim_from_hidden_size():
    args = _make_args(
        recurrent_memory_backend="cross_attn_slots",
        recurrent_slot_num_slots=8,
        recurrent_slot_num_heads=4,
    )
    validate_armt_constraints(args)


def test_cross_attn_slots_read_backend_defaults_to_flash():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.recurrent_slot_read_attn_backend == "flash"


def test_recurrent_memory_lr_defaults_to_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.recurrent_memory_lr is None


def test_recurrent_memory_lr_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--recurrent-memory-lr", "0.0025"])

    assert args.recurrent_memory_lr == pytest.approx(0.0025)


def test_recurrent_memory_lr_must_be_positive():
    args = _make_args(recurrent_memory_lr=0.0)

    with pytest.raises(ValueError, match="recurrent_memory_lr"):
        validate_armt_constraints(args)


def test_recurrent_memory_lr_must_not_be_below_min_lr():
    args = _make_args(recurrent_memory_lr=1.0e-4, min_lr=2.0e-4)

    with pytest.raises(ValueError, match="min_lr"):
        validate_armt_constraints(args)


def test_recurrent_memory_norm_switches_default_to_disabled():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args([])

    assert args.recurrent_mem_qk_norm is False
    assert args.recurrent_memory_input_pre_norm is False
    assert args.recurrent_gdn_read_mode == "normal"


def test_recurrent_gdn_read_mode_flag_is_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(["--recurrent-gdn-read-mode", "buggy"])

    assert args.recurrent_gdn_read_mode == "buggy"


def test_recurrent_memory_norm_switches_are_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(
        [
            "--recurrent-mem-qk-norm",
            "--recurrent-memory-input-pre-norm",
        ]
    )

    assert args.recurrent_mem_qk_norm is True
    assert args.recurrent_memory_input_pre_norm is True


def test_recurrent_memory_norm_switches_can_be_disabled_explicitly():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(
        [
            "--recurrent-mem-qk-norm",
            "--no-recurrent-mem-qk-norm",
            "--recurrent-memory-input-pre-norm",
            "--no-recurrent-memory-input-pre-norm",
        ]
    )

    assert args.recurrent_mem_qk_norm is False
    assert args.recurrent_memory_input_pre_norm is False


def test_gated_deltanet_read_mode_must_be_known():
    args = _make_args(
        recurrent_memory_backend="gated_deltanet",
        recurrent_gdn_key_head_dim=64,
        recurrent_gdn_value_head_dim=64,
        recurrent_gdn_num_key_heads=2,
        recurrent_gdn_num_value_heads=2,
        recurrent_gdn_read_mode="unknown",
    )

    with pytest.raises(ValueError, match="recurrent_gdn_read_mode"):
        validate_armt_constraints(args)


def test_recurrent_slot_qk_norm_legacy_flag_is_rejected():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--recurrent-slot-qk-norm"])


def test_cross_attn_slots_args_are_registered():
    parser = argparse.ArgumentParser()
    add_armt_args(parser)
    args = parser.parse_args(
        [
            "--recurrent-memory-backend",
            "cross_attn_slots",
            "--recurrent-slot-num-slots",
            "8",
            "--recurrent-slot-num-heads",
            "4",
            "--recurrent-slot-head-dim",
            "64",
            "--recurrent-slot-read-attn-backend",
            "sdpa",
        ]
    )

    assert args.recurrent_memory_backend == "cross_attn_slots"
    assert args.recurrent_slot_num_slots == 8
    assert args.recurrent_slot_num_heads == 4
    assert args.recurrent_slot_head_dim == 64
    assert args.recurrent_slot_read_attn_backend == "sdpa"


def test_associative_allows_explicit_head_dim_without_hidden_size_match():
    args = _make_args(
        hidden_size=70,
        armt_d_mem=64,
        armt_n_heads=4,
        armt_head_dim=6,
    )

    validate_armt_constraints(args)


def test_associative_still_requires_head_divisibility():
    args = _make_args(
        hidden_size=70,
        armt_n_heads=4,
        recurrent_memory_backend="associative",
    )

    with pytest.raises(ValueError, match="divisible by armt_n_heads"):
        validate_armt_constraints(args)
