from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.armt.monitoring import (
    build_mean_metric,
    build_ratio_metric,
    build_ratio_of_means_metric,
    build_rms_metric,
    consume_armt_tensorboard_metrics,
    publish_armt_tensorboard_metrics,
)
from megatron.training.training import (
    _should_collect_armt_monitoring_for_iteration,
    _write_scalar_metrics_to_tensorboard,
    training_log,
)


def test_armt_monitoring_collection_follows_tensorboard_interval():
    args = SimpleNamespace(
        tensorboard_dir="/tmp/tensorboard",
        disable_tensorboard_writer=False,
        tensorboard_log_interval=10,
    )

    assert _should_collect_armt_monitoring_for_iteration(args, 8) is False
    assert _should_collect_armt_monitoring_for_iteration(args, 9) is True

    args.disable_tensorboard_writer = True
    assert _should_collect_armt_monitoring_for_iteration(args, 9) is False


def test_training_log_writes_armt_metrics_only_to_tensorboard():
    """验证 training_log 只把 tracker 指标写入 TensorBoard，不写入 WandB。"""
    publish_armt_tensorboard_metrics(
        {
            "armt/read/context_retrieved_norm_mean": build_mean_metric(
                torch.tensor(4.0),
                torch.tensor(2.0),
            ),
            "armt/read/context_retrieved_norm_mean/layer_01": build_mean_metric(
                torch.tensor(9.0),
                torch.tensor(3.0),
            ),
            "armt/read/retrieved_norm_mean": build_mean_metric(
                torch.tensor(15.0),
                torch.tensor(5.0),
            ),
            "armt/read/retrieved_to_hidden_ratio": build_ratio_metric(
                torch.tensor(15.0),
                torch.tensor(30.0),
            ),
            "armt/read/retrieved_norm_mean/chunk_00": build_mean_metric(
                torch.tensor(0.0),
                torch.tensor(0.0),
            ),
            "armt/read/retrieved_to_hidden_ratio/chunk_00": build_ratio_metric(
                torch.tensor(0.0),
                torch.tensor(0.0),
            ),
            "armt/read/retrieved_to_memory_hidden_ratio": build_ratio_metric(
                torch.tensor(6.0),
                torch.tensor(12.0),
            ),
            "armt/read/retrieved_norm_mean/pos_0000": build_mean_metric(
                torch.tensor(12.0),
                torch.tensor(3.0),
            ),
            "armt/read/retrieved_to_hidden_ratio/pos_0000": build_ratio_metric(
                torch.tensor(12.0),
                torch.tensor(24.0),
            ),
            "train/chunk_00_loss": build_ratio_metric(
                torch.tensor(9.0),
                torch.tensor(3.0),
            ),
            "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                torch.tensor(4.0),
                torch.tensor(2.0),
                torch.tensor(30.0),
                torch.tensor(3.0),
            ),
        }
    )
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        tensorboard_dir="/tmp/tensorboard",
        tensorboard_log_interval=1,
        consumed_train_samples=0,
        skipped_train_samples=0,
        log_loss_scale_to_tensorboard=False,
        log_world_size_to_tensorboard=False,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=False,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=0,
        log_interval=10,
        log_timers_to_tensorboard=False,
        log_memory_interval=None,
        record_memory_history=False,
        armt_log_read_chunk_metrics_to_tensorboard=True,
    )
    writer = MagicMock()
    wandb_writer = MagicMock()

    with (
        patch("megatron.training.training.get_args", return_value=args),
        patch("megatron.training.training.get_timers", return_value=MagicMock()),
        patch("megatron.training.training.get_tensorboard_writer", return_value=writer),
        patch("megatron.training.training.get_wandb_writer", return_value=wandb_writer),
        patch("megatron.training.training.get_one_logger", return_value=None),
        patch("megatron.training.training.get_energy_monitor", return_value=MagicMock()),
        patch("megatron.training.training.get_num_microbatches", return_value=1),
        patch(
            "megatron.training.training.reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value: value,
        ),
        patch("megatron.training.training.one_logger_utils.track_app_tag"),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=torch.tensor(1.0),
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=0.0,
        )

    tb_metric_names = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "armt/read/context_retrieved_norm_mean" in tb_metric_names
    assert "armt/read/context_retrieved_norm_mean/layer_01" in tb_metric_names
    assert "armt/read/retrieved_norm_mean" in tb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio" in tb_metric_names
    assert "armt/read/retrieved_norm_mean/chunk_00" in tb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio/chunk_00" in tb_metric_names
    assert "armt/read/retrieved_to_memory_hidden_ratio" in tb_metric_names
    assert "armt/read/retrieved_norm_mean/pos_0000" in tb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio/pos_0000" in tb_metric_names
    assert "armt/token/mem_ctx_norm_ratio" in tb_metric_names
    assert "train/chunk_00_loss" in tb_metric_names
    assert "batch-size-tokens" in tb_metric_names
    assert "batch-size-tokens vs samples" in tb_metric_names

    tb_metrics = {call.args[0]: call.args[1:] for call in writer.add_scalar.call_args_list}
    assert tb_metrics["armt/read/retrieved_norm_mean"] == (3.0, 1)
    assert tb_metrics["armt/read/retrieved_to_hidden_ratio"] == (0.5, 1)
    assert tb_metrics["armt/read/retrieved_norm_mean/chunk_00"] == (0.0, 1)
    assert tb_metrics["armt/read/retrieved_to_hidden_ratio/chunk_00"] == (0.0, 1)
    assert tb_metrics["armt/read/retrieved_to_memory_hidden_ratio"] == (0.5, 1)
    assert tb_metrics["armt/read/retrieved_norm_mean/pos_0000"] == (4.0, 1)
    assert tb_metrics["armt/read/retrieved_to_hidden_ratio/pos_0000"] == (0.5, 1)
    assert tb_metrics["armt/token/mem_ctx_norm_ratio"] == (0.2, 1)
    assert tb_metrics["batch-size-tokens"] == (8, 1)
    assert tb_metrics["batch-size-tokens vs samples"] == (8, 0)

    wandb_metric_names = []
    for call in wandb_writer.log.call_args_list:
        wandb_metric_names.extend(call.args[0].keys())
    assert "armt/read/context_retrieved_norm_mean" not in wandb_metric_names
    assert "armt/read/context_retrieved_norm_mean/layer_01" not in wandb_metric_names
    assert "armt/read/retrieved_norm_mean" not in wandb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio" not in wandb_metric_names
    assert "armt/read/retrieved_norm_mean/chunk_00" not in wandb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio/chunk_00" not in wandb_metric_names
    assert "armt/read/retrieved_to_memory_hidden_ratio" not in wandb_metric_names
    assert "armt/read/retrieved_norm_mean/pos_0000" not in wandb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio/pos_0000" not in wandb_metric_names
    assert "armt/token/mem_ctx_norm_ratio" not in wandb_metric_names
    assert "train/chunk_00_loss" not in wandb_metric_names


def test_training_log_skips_sample_axis_for_full_sequence_avg_loss():
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        tensorboard_dir="/tmp/tensorboard",
        tensorboard_log_interval=1,
        consumed_train_samples=16,
        skipped_train_samples=0,
        log_loss_scale_to_tensorboard=False,
        log_world_size_to_tensorboard=False,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=False,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=0,
        log_interval=10,
        log_timers_to_tensorboard=False,
        log_memory_interval=None,
        record_memory_history=False,
        armt_log_read_chunk_metrics_to_tensorboard=True,
    )
    writer = MagicMock()

    with (
        patch("megatron.training.training.get_args", return_value=args),
        patch("megatron.training.training.get_timers", return_value=MagicMock()),
        patch("megatron.training.training.get_tensorboard_writer", return_value=writer),
        patch("megatron.training.training.get_wandb_writer", return_value=None),
        patch("megatron.training.training.get_one_logger", return_value=None),
        patch("megatron.training.training.get_energy_monitor", return_value=MagicMock()),
        patch("megatron.training.training.get_num_microbatches", return_value=1),
        patch(
            "megatron.training.training.reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value: value,
        ),
        patch("megatron.training.training.one_logger_utils.track_app_tag"),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        training_log(
            loss_dict={"lm loss": 1.0, "train/avg_loss": 2.0},
            total_loss_dict={},
            learning_rate=torch.tensor(1.0),
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=0.0,
        )

    tb_metric_names = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "lm loss" in tb_metric_names
    assert "lm loss vs samples" in tb_metric_names
    assert "train/avg_loss" in tb_metric_names
    assert "train/avg_loss vs samples" not in tb_metric_names


def test_training_log_filters_armt_chunk_metrics_when_disabled():
    publish_armt_tensorboard_metrics(
        {
            "armt/read/retrieved_norm_mean": build_mean_metric(
                torch.tensor(15.0),
                torch.tensor(5.0),
            ),
            "armt/read/retrieved_to_hidden_ratio": build_ratio_metric(
                torch.tensor(15.0),
                torch.tensor(30.0),
            ),
            "armt/read/retrieved_norm_mean/chunk_00": build_mean_metric(
                torch.tensor(0.0),
                torch.tensor(0.0),
            ),
            "armt/read/retrieved_to_hidden_ratio/chunk_00": build_ratio_metric(
                torch.tensor(0.0),
                torch.tensor(0.0),
            ),
        }
    )
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        tensorboard_dir="/tmp/tensorboard",
        tensorboard_log_interval=1,
        consumed_train_samples=0,
        skipped_train_samples=0,
        log_loss_scale_to_tensorboard=False,
        log_world_size_to_tensorboard=False,
        log_memory_to_tensorboard=False,
        log_max_attention_logit=False,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=0,
        log_interval=10,
        log_timers_to_tensorboard=False,
        log_memory_interval=None,
        record_memory_history=False,
        armt_log_read_chunk_metrics_to_tensorboard=False,
    )
    writer = MagicMock()

    with (
        patch("megatron.training.training.get_args", return_value=args),
        patch("megatron.training.training.get_timers", return_value=MagicMock()),
        patch("megatron.training.training.get_tensorboard_writer", return_value=writer),
        patch("megatron.training.training.get_wandb_writer", return_value=None),
        patch("megatron.training.training.get_one_logger", return_value=None),
        patch("megatron.training.training.get_energy_monitor", return_value=MagicMock()),
        patch("megatron.training.training.get_num_microbatches", return_value=1),
        patch(
            "megatron.training.training.reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value: value,
        ),
        patch("megatron.training.training.one_logger_utils.track_app_tag"),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=torch.tensor(1.0),
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=0.0,
        )

    tb_metric_names = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "armt/read/retrieved_norm_mean" in tb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio" in tb_metric_names
    assert "armt/read/retrieved_norm_mean/chunk_00" not in tb_metric_names
    assert "armt/read/retrieved_to_hidden_ratio/chunk_00" not in tb_metric_names


def test_consume_armt_metrics_batches_all_reduce_calls():
    publish_armt_tensorboard_metrics(
        {
            "armt/read/retrieved_norm_mean": build_mean_metric(
                torch.tensor(12.0),
                torch.tensor(3.0),
            ),
            "armt/read/retrieved_to_hidden_ratio": build_ratio_metric(
                torch.tensor(12.0),
                torch.tensor(24.0),
            ),
            "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                torch.tensor(4.0),
                torch.tensor(2.0),
                torch.tensor(30.0),
                torch.tensor(3.0),
            ),
            "armt/write/delta_mem_norm": build_rms_metric(
                torch.tensor(16.0),
                torch.tensor(4.0),
            ),
        }
    )

    fake_reduce_group = object()

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce") as all_reduce_mock,
    ):
        metrics = consume_armt_tensorboard_metrics(reduce_group=fake_reduce_group)

    assert all_reduce_mock.call_count == 1
    reduced_tensor = all_reduce_mock.call_args.kwargs.get("tensor")
    if reduced_tensor is None:
        reduced_tensor = all_reduce_mock.call_args.args[0]
    assert reduced_tensor.numel() == 10
    assert all_reduce_mock.call_args.kwargs["group"] is fake_reduce_group
    assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(4.0)
    assert float(metrics["armt/read/retrieved_to_hidden_ratio"]) == pytest.approx(0.5)
    assert float(metrics["armt/token/mem_ctx_norm_ratio"]) == pytest.approx(0.2)
    assert float(metrics["armt/write/delta_mem_norm"]) == pytest.approx(2.0)


def test_consume_armt_metrics_can_skip_finalize_on_non_writer_rank():
    publish_armt_tensorboard_metrics(
        {
            "armt/read/retrieved_norm_mean": build_mean_metric(
                torch.tensor(12.0),
                torch.tensor(3.0),
            ),
            "armt/read/retrieved_to_hidden_ratio": build_ratio_metric(
                torch.tensor(12.0),
                torch.tensor(24.0),
            ),
        }
    )

    fake_reduce_group = object()

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce") as all_reduce_mock,
    ):
        metrics = consume_armt_tensorboard_metrics(
            reduce_group=fake_reduce_group,
            finalize_on_this_rank=False,
        )

    assert metrics == {}
    assert all_reduce_mock.call_count == 1
    reduced_tensor = all_reduce_mock.call_args.kwargs.get("tensor")
    if reduced_tensor is None:
        reduced_tensor = all_reduce_mock.call_args.args[0]
    assert reduced_tensor.numel() == 4
    assert all_reduce_mock.call_args.kwargs["group"] is fake_reduce_group


def test_write_scalar_metrics_to_tensorboard_batches_summary_writer_calls():
    recorded = {}

    class FakeFileWriter:
        def add_summary(self, summary, global_step=None, walltime=None):
            recorded["summary"] = summary
            recorded["global_step"] = global_step
            recorded["walltime"] = walltime

    class FakeSummaryWriter:
        def __init__(self):
            self._file_writer = FakeFileWriter()

        def _get_file_writer(self):
            return self._file_writer

        def add_scalar(self, *_args, **_kwargs):
            raise AssertionError("batched tensorboard helper should not fall back to add_scalar")

    writer = FakeSummaryWriter()
    metrics = {
        "armt/read/retrieved_norm_mean": torch.tensor(3.0),
        "armt/read/retrieved_to_hidden_ratio": 0.5,
    }

    with patch("megatron.training.training.TorchSummaryWriter", FakeSummaryWriter):
        _write_scalar_metrics_to_tensorboard(writer, metrics, iteration=7)

    assert recorded["global_step"] == 7
    summary_values = {value.tag: value.simple_value for value in recorded["summary"].value}
    assert summary_values["armt/read/retrieved_norm_mean"] == pytest.approx(3.0)
    assert summary_values["armt/read/retrieved_to_hidden_ratio"] == pytest.approx(0.5)
