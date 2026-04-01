# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from megatron.core.enums import ModelType
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training import training as training_module
from megatron.training.training import build_train_valid_test_data_iterators, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def test_format_parameter_count_display():
    assert training_module._format_parameter_count_display(999) == "999"
    assert training_module._format_parameter_count_display(1_500) == "1,500 (1.500K)"
    assert (
        training_module._format_parameter_count_display(12_582_912)
        == "12,582,912 (12.583M)"
    )
    assert (
        training_module._format_parameter_count_display(1_536_000_000)
        == "1,536,000,000 (1.536B)"
    )


def test_render_ascii_table():
    table = training_module._render_ascii_table(
        headers=["a", "bb"],
        rows=[["ccc", "d"]],
        indent=" ",
    )

    assert table == "\n".join(
        [
            " +-----+----+",
            " | a   | bb |",
            " +-----+----+",
            " | ccc | d  |",
            " +-----+----+",
        ]
    )


def test_format_memory_parameter_report_includes_summary_and_table():
    report = training_module._format_memory_parameter_report(
        total_model_parameters=10_000,
        breakdown=[
            ("memory_embeddings", 1_000),
            ("decoder.layers.0.recurrent_memory_layer", 500),
        ],
        memory_state_breakdown=[
            ("initial_slots", 120),
            ("W_mem", 340),
        ],
        tp_rank=0,
        pp_rank=0,
    )

    assert " > memory parameter summary on (tensor, pipeline) model parallel rank (0, 0)" in report
    assert "total memory params: 1,500 (1.500K)" in report
    assert "memory / model params: 15.00%" in report
    assert "modules:" in report
    assert "percentage" in report
    assert "memory_embeddings" in report
    assert "initial_slots state size (batch=1): 120" in report
    assert "W_mem state size (batch=1): 340" in report
    assert report.index("memory / model params: 15.00%") < report.index("initial_slots state size (batch=1): 120")
    assert report.index("W_mem state size (batch=1): 340") < report.index("modules:")
    assert "10.00%" in report
    assert "5.00%" in report


def test_format_memory_parameter_report_omits_module_table_when_breakdown_empty():
    report = training_module._format_memory_parameter_report(
        total_model_parameters=10_000,
        breakdown=[],
        memory_state_breakdown=[],
        tp_rank=0,
        pp_rank=0,
    )

    assert "total memory params: 0" in report
    assert "memory / model params: 0.00%" in report
    assert "modules:" not in report


def test_print_memory_parameter_breakdown_uses_structured_report(monkeypatch, capsys):
    monkeypatch.setattr(
        training_module,
        "_collect_memory_parameter_breakdown",
        lambda model: (True, [("memory_embeddings", 1_000)]),
    )
    monkeypatch.setattr(training_module, "get_pg_rank", lambda group: 0)

    pg_collection = SimpleNamespace(dp="dp", cp="cp", tp="tp", pp="pp")
    training_module._print_memory_parameter_breakdown(
        [torch.nn.Linear(2, 3)],
        10_000,
        pg_collection,
    )

    captured = capsys.readouterr()
    assert "total memory params: 1,000 (1.000K)" in captured.out
    assert "memory / model params: 10.00%" in captured.out
    assert "percentage" in captured.out


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == _vocab_size_with_padding(
                    vocab, args, False
                ), (vocab, mult)

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == _vocab_size_with_padding(
                    vocab, args, False
                ), (vocab, mult)

    def test_collect_memory_parameter_breakdown_ignores_models_without_interface(self):
        has_memory_breakdown, breakdown = training_module._collect_memory_parameter_breakdown(
            [torch.nn.Linear(2, 3)]
        )

        assert has_memory_breakdown is False
        assert breakdown == []

    def test_collect_memory_parameter_breakdown_collects_chunk_prefixed_entries(self):
        model = [
            SimpleNamespace(
                get_memory_parameter_breakdown=lambda: [("memory_embeddings", 100)],
            ),
            SimpleNamespace(
                get_memory_parameter_breakdown=lambda: [
                    ("decoder.layers.0.recurrent_memory_layer", 200)
                ],
            ),
        ]

        has_memory_breakdown, breakdown = training_module._collect_memory_parameter_breakdown(model)

        assert has_memory_breakdown is True
        assert breakdown == [
            ("model_chunk0.memory_embeddings", 100),
            ("model_chunk1.decoder.layers.0.recurrent_memory_layer", 200),
        ]

    def test_collect_memory_state_breakdown_aggregates_across_model_chunks(self):
        model = [
            SimpleNamespace(
                get_memory_state_breakdown=lambda batch_size=1: [
                    ("initial_slots", 96 * batch_size),
                    ("W_mem", 384 * batch_size),
                ],
            ),
            SimpleNamespace(
                get_memory_state_breakdown=lambda batch_size=1: [
                    ("W_mem", 128 * batch_size),
                ],
            ),
        ]

        has_memory_state_breakdown, breakdown = training_module._collect_memory_state_breakdown(
            model,
            batch_size=1,
        )

        assert has_memory_state_breakdown is True
        assert breakdown == [
            ("initial_slots", 96),
            ("W_mem", 512),
        ]

    def test_maybe_exit_after_parameter_statistics(self, monkeypatch):
        barrier_mock = MagicMock()
        print_mock = MagicMock()

        monkeypatch.setattr(
            training_module.torch.distributed,
            "is_initialized",
            lambda: True,
        )
        monkeypatch.setattr(training_module.torch.distributed, "barrier", barrier_mock)
        monkeypatch.setattr(training_module, "print_rank_0", print_mock)

        args = SimpleNamespace(param_stats_only=True)

        assert training_module._maybe_exit_after_parameter_statistics(args) is True
        barrier_mock.assert_called_once()
        print_mock.assert_called_once()

    def test_setup_model_and_optimizer_skips_optimizer_and_checkpoint_load_for_param_stats_only(
        self, monkeypatch
    ):
        args = SimpleNamespace(
            skip_train=False,
            param_stats_only=True,
            moe_use_upcycling=False,
            load="/tmp/checkpoint",
            pretrained_checkpoint=None,
        )
        get_model_calls = {}
        fake_model = [torch.nn.Linear(1, 1)]

        monkeypatch.setattr(training_module, "get_args", lambda: args)
        monkeypatch.setattr(training_module, "get_timers", lambda: MagicMock())
        monkeypatch.setattr(training_module, "get_one_logger", lambda: None)
        monkeypatch.setattr(training_module, "unwrap_model", lambda model: model)

        def fake_get_model(model_provider_func, model_type, wrap_with_ddp=True, config=None, pg_collection=None):
            del model_provider_func, model_type, config, pg_collection
            get_model_calls["wrap_with_ddp"] = wrap_with_ddp
            return fake_model

        monkeypatch.setattr(training_module, "get_model", fake_get_model)
        monkeypatch.setattr(
            training_module,
            "get_megatron_optimizer",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer should not be built")),
        )
        monkeypatch.setattr(
            training_module,
            "get_megatron_muon_optimizer",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("muon optimizer should not be built")),
        )
        monkeypatch.setattr(
            training_module,
            "load_checkpoint",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("checkpoint should not be loaded")),
        )

        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            lambda: None,
            ModelType.encoder_or_decoder,
        )

        assert model == fake_model
        assert optimizer is None
        assert opt_param_scheduler is None
        assert get_model_calls["wrap_with_ddp"] is False

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
