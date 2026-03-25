from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.armt.monitoring import (
    build_mean_metric,
    clear_armt_tensorboard_metrics,
    consume_armt_tensorboard_metrics,
)
from megatron.core.pipeline_parallel.recurrent_schedules import (
    recurrent_forward_backward_no_pipelining,
)
from megatron.training import global_vars as training_global_vars

SCHEDULE_MODULE = "megatron.core.pipeline_parallel.recurrent_schedules"


@contextmanager
def _patched_schedule(raw_batch, config, unwrapped_model):
    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.get_model_config", return_value=config)
        )
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.get_model_type", return_value=MagicMock())
        )
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.unwrap_model", return_value=unwrapped_model)
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_context_parallel_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_embedding_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_pipeline_model_parallel_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_position_embedding_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                f"{SCHEDULE_MODULE}.parallel_state.get_data_parallel_group",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "megatron.training.utils.get_batch_on_this_tp_rank",
                side_effect=lambda _it: raw_batch,
            )
        )
        yield


def _build_loss_func(batch):
    def _loss_func(output_tensor):
        num_tokens = batch["loss_mask"].float().sum().to(torch.int)
        loss_reduced = {
            "lm loss": torch.cat([num_tokens.float().view(1), num_tokens.view(1)]),
        }
        return output_tensor * 0.0, num_tokens, loss_reduced

    return _loss_func


def test_scheduler_sets_skip_read_memory_only_on_first_chunk():
    raw_batch = {
        "tokens": torch.randint(0, 100, (1, 8)),
        "labels": torch.randint(0, 100, (1, 8)),
        "loss_mask": torch.ones(1, 8),
    }

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        return torch.zeros([], requires_grad=True), _build_loss_func(batch)

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    unwrapped_model = MagicMock()

    with _patched_schedule(raw_batch, config, unwrapped_model):
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                recurrent_chunk_size=4,
                no_read_memory_from_first_chunk=True,
                no_loss_from_first_chunk=False,
            )
            recurrent_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=MagicMock(),
                num_microbatches=1,
                seq_length=8,
                micro_batch_size=1,
                forward_only=True,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

    skip_calls = [
        call.args[0] for call in unwrapped_model.set_skip_read_memory_for_current_chunk.call_args_list
    ]
    first_chunk_calls = [
        call.args[0] for call in unwrapped_model.set_current_chunk_is_first.call_args_list
    ]

    assert skip_calls[0] is True
    assert all(call is False for call in skip_calls[1:])
    assert first_chunk_calls[0] is True
    assert all(call is False for call in first_chunk_calls[1:])


def test_scheduler_skip_read_memory_does_not_conflict_with_first_chunk_loss_masking():
    raw_batch = {
        "tokens": torch.randint(0, 100, (1, 8)),
        "labels": torch.randint(0, 100, (1, 8)),
        "loss_mask": torch.ones(1, 8),
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        forward_calls["n"] += 1
        if forward_calls["n"] == 1:
            assert batch["loss_mask"].sum().item() == 0.0
        else:
            assert batch["loss_mask"].sum().item() > 0.0
        return torch.zeros([], requires_grad=True), _build_loss_func(batch)

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    unwrapped_model = MagicMock()

    with _patched_schedule(raw_batch, config, unwrapped_model), patch(
        f"{SCHEDULE_MODULE}.backward_step"
    ) as backward_step_mock:
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                recurrent_chunk_size=4,
                no_read_memory_from_first_chunk=True,
                no_loss_from_first_chunk=True,
            )
            recurrent_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=MagicMock(),
                num_microbatches=1,
                seq_length=8,
                micro_batch_size=1,
                forward_only=False,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

    skip_calls = [
        call.args[0] for call in unwrapped_model.set_skip_read_memory_for_current_chunk.call_args_list
    ]

    assert forward_calls["n"] == 2
    assert backward_step_mock.call_count == 1
    assert skip_calls[0] is True
    assert all(call is False for call in skip_calls[1:])


def test_scheduler_publishes_rmt_mem_token_cosine_metrics():
    clear_armt_tensorboard_metrics()
    raw_batch = {
        "tokens": torch.randint(0, 100, (1, 8)),
        "labels": torch.randint(0, 100, (1, 8)),
        "loss_mask": torch.ones(1, 8),
    }

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        return torch.zeros([], requires_grad=True), _build_loss_func(batch)

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    unwrapped_model = MagicMock()
    unwrapped_model.consume_all_monitoring_primitives.return_value = {
        "rmt/token/write_mem_token_cosine_mean": build_mean_metric(
            torch.tensor(6.0),
            torch.tensor(3.0),
        ),
    }

    with _patched_schedule(raw_batch, config, unwrapped_model), patch(
        f"{SCHEDULE_MODULE}.backward_step"
    ):
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                recurrent_chunk_size=4,
                no_read_memory_from_first_chunk=False,
                no_loss_from_first_chunk=False,
            )
            recurrent_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=MagicMock(),
                num_microbatches=1,
                seq_length=8,
                micro_batch_size=1,
                forward_only=False,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

    metrics = consume_armt_tensorboard_metrics()

    assert float(metrics["rmt/token/write_mem_token_cosine_mean"]) == pytest.approx(2.0)
