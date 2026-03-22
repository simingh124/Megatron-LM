from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from megatron.core.models.armt.monitoring import (
    build_mean_metric,
    build_ratio_of_means_metric,
    publish_armt_tensorboard_metrics,
)
from megatron.training.training import training_log


def test_training_log_writes_rmt_metrics_only_to_tensorboard():
    publish_armt_tensorboard_metrics(
        {
            "rmt/token/read_mem_token_norm_mean_first_chunk": build_mean_metric(
                torch.tensor(6.0),
                torch.tensor(2.0),
            ),
            "rmt/token/read_mem_token_cosine_mean_first_chunk": build_mean_metric(
                torch.tensor(1.0),
                torch.tensor(2.0),
            ),
            "rmt/token/read_ctx_norm_ratio_first_chunk": build_ratio_of_means_metric(
                torch.tensor(6.0),
                torch.tensor(2.0),
                torch.tensor(18.0),
                torch.tensor(3.0),
            ),
            "rmt/token/write_ctx_norm_ratio": build_ratio_of_means_metric(
                torch.tensor(20.0),
                torch.tensor(4.0),
                torch.tensor(10.0),
                torch.tensor(2.0),
            ),
            "rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean": build_mean_metric(
                torch.tensor(3.0),
                torch.tensor(4.0),
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
    assert "rmt/token/read_mem_token_norm_mean_first_chunk" in tb_metric_names
    assert "rmt/token/read_mem_token_cosine_mean_first_chunk" in tb_metric_names
    assert "rmt/token/read_ctx_norm_ratio_first_chunk" in tb_metric_names
    assert "rmt/token/write_ctx_norm_ratio" in tb_metric_names
    assert "rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean" in tb_metric_names

    tb_metrics = {call.args[0]: call.args[1:] for call in writer.add_scalar.call_args_list}
    assert tb_metrics["rmt/token/read_mem_token_norm_mean_first_chunk"] == (3.0, 1)
    assert tb_metrics["rmt/token/read_mem_token_cosine_mean_first_chunk"] == (0.5, 1)
    assert tb_metrics["rmt/token/read_ctx_norm_ratio_first_chunk"] == (0.5, 1)
    assert tb_metrics["rmt/token/write_ctx_norm_ratio"] == (1.0, 1)
    assert tb_metrics["rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean"] == (0.75, 1)

    wandb_metric_names = []
    for call in wandb_writer.log.call_args_list:
        wandb_metric_names.extend(call.args[0].keys())
    assert "rmt/token/read_mem_token_norm_mean_first_chunk" not in wandb_metric_names
    assert "rmt/token/read_mem_token_cosine_mean_first_chunk" not in wandb_metric_names
    assert "rmt/token/read_ctx_norm_ratio_first_chunk" not in wandb_metric_names
    assert "rmt/token/write_ctx_norm_ratio" not in wandb_metric_names
    assert "rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean" not in wandb_metric_names
