from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.training import global_vars as training_global_vars
from megatron.core.pipeline_parallel.recurrent_schedules import (
    chunk_data,
    recurrent_forward_backward_no_pipelining,
)

SCHEDULE_MODULE = "megatron.core.pipeline_parallel.recurrent_schedules"


@contextmanager
def _patched_schedule(raw_batch, config):
    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.get_model_config", return_value=config)
        )
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.get_model_type", return_value=MagicMock())
        )
        stack.enter_context(
            patch(f"{SCHEDULE_MODULE}.unwrap_model", return_value=MagicMock())
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


def test_chunk_data_shapes():
    batch_size, seq_length = 2, 2048
    chunk_size = 512
    tokens = torch.randint(0, 32000, (batch_size, seq_length))
    labels = torch.randint(0, 32000, (batch_size, seq_length))
    loss_mask = torch.ones(batch_size, seq_length)
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1).bool()

    data = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

    chunks = chunk_data(data, chunk_size=chunk_size, seq_length=seq_length)
    assert len(chunks) == seq_length // chunk_size
    for chunk in chunks:
        assert chunk["tokens"].shape == (batch_size, chunk_size)
        assert chunk["labels"].shape == (batch_size, chunk_size)
        assert chunk["loss_mask"].shape == (batch_size, chunk_size)
        assert chunk["position_ids"].shape == (batch_size, chunk_size)
        assert chunk["attention_mask"].shape[-1] == chunk_size
        assert chunk["attention_mask"].shape[-2] == chunk_size


def test_chunk_data_content_continuity():
    batch_size, seq_length = 2, 1024
    chunk_size = 256
    tokens = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    chunks = chunk_data({"tokens": tokens}, chunk_size=chunk_size, seq_length=seq_length)
    reconstructed = torch.cat([chunk["tokens"] for chunk in chunks], dim=1)
    assert torch.equal(reconstructed, tokens)


def test_no_loss_from_first_chunk_masks_and_skips_backward():
    batch_size, seq_length = 2, 1024
    chunk_size = 512
    raw_batch = {
        "tokens": torch.randint(0, 100, (batch_size, seq_length)),
        "labels": torch.randint(0, 100, (batch_size, seq_length)),
        "loss_mask": torch.ones(batch_size, seq_length),
        "attention_mask": torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1).bool(),
        "position_ids": torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1),
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

        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            loss_reduced = {
                "lm loss": torch.cat([_output_tensor.detach().view(1), num_tokens.view(1)])
            }
            return _output_tensor * 1.0, num_tokens, loss_reduced

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    with _patched_schedule(raw_batch, config), patch(
        f"{SCHEDULE_MODULE}.backward_step"
    ) as backward_step_mock:
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                recurrent_chunk_size=chunk_size,
                no_loss_from_first_chunk=True,
            )
            losses = recurrent_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=MagicMock(),
                num_microbatches=1,
                seq_length=seq_length,
                micro_batch_size=batch_size,
                forward_only=False,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

    assert forward_calls["n"] == seq_length // chunk_size
    assert backward_step_mock.call_count == 1
    assert len(losses) == 1


def test_loss_reduced_is_summed_across_chunks_new_style():
    batch_size, seq_length = 1, 1024
    chunk_size = 512
    raw_batch = {
        "tokens": torch.randint(0, 100, (batch_size, seq_length)),
        "labels": torch.randint(0, 100, (batch_size, seq_length)),
        "loss_mask": torch.ones(batch_size, seq_length),
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        forward_calls["n"] += 1
        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            loss_sum = num_tokens.float() if forward_calls["n"] == 1 else num_tokens.float() * 2.0
            loss_reduced = {"lm loss": torch.cat([loss_sum.view(1), num_tokens.view(1)])}
            return _output_tensor * 0.0 + loss_sum, num_tokens, loss_reduced

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    with _patched_schedule(raw_batch, config):
        losses = recurrent_forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=iter([raw_batch]),
            model=MagicMock(),
            num_microbatches=1,
            chunk_size=chunk_size,
            seq_length=seq_length,
            micro_batch_size=batch_size,
            forward_only=True,
        )

    assert forward_calls["n"] == seq_length // chunk_size
    assert len(losses) == 1
    assert torch.allclose(losses[0]["lm loss"], torch.tensor([1536.0, 1024.0]))


def test_scalar_metrics_are_token_weighted_across_chunks():
    batch_size, seq_length = 1, 8
    chunk_size = 4
    loss_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.float)
    raw_batch = {
        "tokens": torch.randint(0, 100, (batch_size, seq_length)),
        "labels": torch.randint(0, 100, (batch_size, seq_length)),
        "loss_mask": loss_mask,
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        forward_calls["n"] += 1
        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            scalar = torch.tensor(1.0) if forward_calls["n"] == 1 else torch.tensor(3.0)
            return _output_tensor * 0.0, num_tokens, {"scalar_metric": scalar}

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    with _patched_schedule(raw_batch, config):
        losses = recurrent_forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=iter([raw_batch]),
            model=MagicMock(),
            num_microbatches=1,
            chunk_size=chunk_size,
            seq_length=seq_length,
            micro_batch_size=batch_size,
            forward_only=True,
        )

    assert pytest.approx(losses[0]["scalar_metric"].item()) == 7.0 / 3.0


def test_get_forward_backward_func_selects_recurrent_tbptt_schedule():
    from megatron.core.pipeline_parallel.schedules import (
        forward_backward_no_pipelining,
        get_forward_backward_func,
    )

    old_global_args = training_global_vars._GLOBAL_ARGS
    try:
        training_global_vars._GLOBAL_ARGS = SimpleNamespace(use_recurrent_tbptt=True)
        func = get_forward_backward_func(pp_size=1, vp_size=None)
        assert func is recurrent_forward_backward_no_pipelining

        training_global_vars._GLOBAL_ARGS = SimpleNamespace(use_armt_tbptt=True)
        func = get_forward_backward_func(pp_size=1, vp_size=None)
        assert func is recurrent_forward_backward_no_pipelining

        training_global_vars._GLOBAL_ARGS = SimpleNamespace()
        func = get_forward_backward_func(pp_size=1, vp_size=None)
        assert func is forward_backward_no_pipelining
    finally:
        training_global_vars._GLOBAL_ARGS = old_global_args


def test_armt_schedule_compatibility_alias_points_to_recurrent_schedule():
    from megatron.core.pipeline_parallel.armt_schedules import (
        armt_forward_backward_no_pipelining,
    )

    assert armt_forward_backward_no_pipelining is recurrent_forward_backward_no_pipelining


def test_no_loss_from_first_chunk_requires_loss_mask():
    raw_batch = {
        "tokens": torch.randint(0, 100, (1, 8)),
        "labels": torch.randint(0, 100, (1, 8)),
    }

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    with _patched_schedule(raw_batch, config):
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                recurrent_chunk_size=4,
                no_loss_from_first_chunk=True,
            )
            with pytest.raises(ValueError, match="loss_mask"):
                recurrent_forward_backward_no_pipelining(
                    forward_step_func=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError(
                            "forward_step_func should not be called when loss_mask is missing"
                        )
                    ),
                    data_iterator=iter([raw_batch]),
                    model=MagicMock(),
                    num_microbatches=1,
                    seq_length=8,
                    micro_batch_size=1,
                    forward_only=True,
                )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args


def test_recurrent_schedule_accepts_legacy_chunk_size_attr():
    batch_size, seq_length = 1, 8
    chunk_size = 4
    raw_batch = {
        "tokens": torch.randint(0, 100, (batch_size, seq_length)),
        "labels": torch.randint(0, 100, (batch_size, seq_length)),
        "loss_mask": torch.ones(batch_size, seq_length),
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        del model
        batch = next(data_it)
        forward_calls["n"] += 1
        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            return _output_tensor * 0.0, num_tokens, {"loss": _output_tensor.detach().view(1)}

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    with _patched_schedule(raw_batch, config):
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                armt_chunk_size=chunk_size,
                no_loss_from_first_chunk=False,
            )
            recurrent_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=MagicMock(),
                num_microbatches=1,
                seq_length=seq_length,
                micro_batch_size=batch_size,
                forward_only=True,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

    assert forward_calls["n"] == seq_length // chunk_size
