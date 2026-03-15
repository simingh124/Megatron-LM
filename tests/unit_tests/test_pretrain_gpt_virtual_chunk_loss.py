from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.armt.monitoring import (
    clear_armt_tensorboard_metrics,
    consume_armt_tensorboard_metrics,
)
import pretrain_gpt


def test_build_virtual_chunk_monitoring_primitives_uses_token_weighted_sums():
    losses = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )

    primitives = pretrain_gpt._build_virtual_chunk_monitoring_primitives(
        losses,
        loss_mask,
        virtual_chunk_size=2,
    )

    clear_armt_tensorboard_metrics()
    with patch("torch.distributed.is_initialized", return_value=False):
        pretrain_gpt.accumulate_armt_tensorboard_metrics(primitives)
        metrics = consume_armt_tensorboard_metrics()

    assert torch.allclose(metrics["train/chunk_00_loss"], torch.tensor(8.0 / 3.0))
    assert torch.allclose(metrics["train/chunk_01_loss"], torch.tensor(15.0 / 2.0))


def test_loss_func_publishes_virtual_chunk_metrics_without_log_report():
    output_tensor = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )
    args = SimpleNamespace(
        baseline_virtual_chunk_size=2,
        check_for_nan_in_loss_and_grad=False,
        check_for_spiky_loss=False,
        modelopt_enabled=False,
    )

    clear_armt_tensorboard_metrics()
    with (
        patch.object(pretrain_gpt, "get_args", return_value=args),
        patch.object(pretrain_gpt, "get_rerun_state_machine", return_value=MagicMock()),
        patch.object(pretrain_gpt, "has_nvidia_modelopt", False),
    ):
        loss, num_tokens, report = pretrain_gpt.loss_func(loss_mask, output_tensor)
    with patch("torch.distributed.is_initialized", return_value=False):
        metrics = consume_armt_tensorboard_metrics()

    assert float(loss) == pytest.approx(23.0)
    assert int(num_tokens) == 5
    assert torch.allclose(report["lm loss"], torch.tensor([23.0, 5.0]))
    assert set(report.keys()) == {"lm loss"}
    assert torch.allclose(metrics["train/chunk_00_loss"], torch.tensor(8.0 / 3.0))
    assert torch.allclose(metrics["train/chunk_01_loss"], torch.tensor(15.0 / 2.0))


def test_loss_func_keeps_original_report_when_virtual_chunk_disabled():
    output_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    args = SimpleNamespace(
        baseline_virtual_chunk_size=None,
        check_for_nan_in_loss_and_grad=False,
        check_for_spiky_loss=False,
        modelopt_enabled=False,
    )

    with (
        patch.object(pretrain_gpt, "get_args", return_value=args),
        patch.object(pretrain_gpt, "get_rerun_state_machine", return_value=MagicMock()),
        patch.object(pretrain_gpt, "has_nvidia_modelopt", False),
    ):
        _, _, report = pretrain_gpt.loss_func(loss_mask, output_tensor)

    assert set(report.keys()) == {"lm loss"}
