import torch

from megatron.core.pipeline_parallel.armt_schedules import chunk_data


def test_chunk_data_shapes():
    """验证 TBPTT chunk_data 会按 chunk_size 切分并同步切分 tokens/labels/loss_mask/position_ids/mask。"""
    B, S = 2, 2048
    chunk_size = 512
    tokens = torch.randint(0, 32000, (B, S))
    labels = torch.randint(0, 32000, (B, S))
    loss_mask = torch.ones(B, S)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    attention_mask = torch.triu(torch.ones(1, 1, S, S), diagonal=1).bool()

    data = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

    chunks = chunk_data(data, chunk_size=chunk_size, seq_length=S)
    assert len(chunks) == S // chunk_size
    for chunk in chunks:
        assert chunk["tokens"].shape == (B, chunk_size)
        assert chunk["labels"].shape == (B, chunk_size)
        assert chunk["loss_mask"].shape == (B, chunk_size)
        assert chunk["position_ids"].shape == (B, chunk_size)
        assert chunk["attention_mask"].shape[-1] == chunk_size
        assert chunk["attention_mask"].shape[-2] == chunk_size


def test_chunk_data_content_continuity():
    """验证 chunk_data 切分后的 token 序列可按顺序拼回原序列（内容连续性）。"""
    B, S = 2, 1024
    chunk_size = 256
    tokens = torch.arange(S).unsqueeze(0).expand(B, -1)
    data = {"tokens": tokens}

    chunks = chunk_data(data, chunk_size=chunk_size, seq_length=S)
    reconstructed = torch.cat([c["tokens"] for c in chunks], dim=1)
    assert torch.equal(reconstructed, tokens)


def test_no_loss_from_first_chunk_masks_and_skips_backward():
    """验证 --no-loss-from-first-chunk 会将首个 chunk 的 loss_mask 置 0 并跳过 backward_step。

    该测试聚焦 schedule 逻辑：
    - 首 chunk 仍会 forward（以保证 memory 写入语义不被破坏）
    - 但由于 loss_mask=0 导致 chunk_tokens=0，从而 skip backward
    """
    from unittest.mock import MagicMock, patch

    from megatron.core.pipeline_parallel.armt_schedules import armt_forward_backward_no_pipelining

    B, S = 2, 1024
    chunk_size = 512
    num_microbatches = 1

    # Build a fake raw batch matching pretrain forward_step expectations.
    raw_batch = {
        "tokens": torch.randint(0, 100, (B, S)),
        "labels": torch.randint(0, 100, (B, S)),
        "loss_mask": torch.ones(B, S),
        "attention_mask": torch.triu(torch.ones(1, 1, S, S), diagonal=1).bool(),
        "position_ids": torch.arange(S).unsqueeze(0).expand(B, -1),
    }

    # Forward step: return a constant scalar loss tensor and a loss_func that returns (loss, num_tokens, reduced)
    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        batch = next(data_it)
        forward_calls["n"] += 1
        # The schedule should have masked the first chunk's loss_mask to zeros.
        if forward_calls["n"] == 1:
            assert batch["loss_mask"].sum().item() == 0.0
        else:
            assert batch["loss_mask"].sum().item() > 0.0

        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            loss_reduced = {"lm loss": torch.cat([_output_tensor.clone().detach().view(1), num_tokens.view(1)])}
            # Make loss a non-leaf tensor (schedule may rescale it in-place).
            return _output_tensor * 1.0, num_tokens, loss_reduced

        return output_tensor, _loss_func

    # Minimal config and model metadata
    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    model = MagicMock()

    with (
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_config", return_value=config),
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_type", return_value=MagicMock()),
        patch("megatron.core.pipeline_parallel.armt_schedules.unwrap_model", return_value=MagicMock()),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_tensor_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_context_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_pipeline_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_position_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_data_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.training.utils.get_batch_on_this_tp_rank",
            side_effect=lambda it: raw_batch,
        ),
        patch("megatron.core.pipeline_parallel.armt_schedules.backward_step") as backward_step_mock,
    ):
        # Provide args via training_global_vars used inside the schedule.
        from types import SimpleNamespace
        from megatron.training import global_vars as training_global_vars

        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                armt_chunk_size=chunk_size,
                no_loss_from_first_chunk=True,
            )

            losses = armt_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter([raw_batch]),
                model=model,
                num_microbatches=num_microbatches,
                seq_length=S,
                micro_batch_size=B,
                forward_only=False,
            )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args

        # Two chunks -> two forward calls.
        assert forward_calls["n"] == S // chunk_size
        # Only the second chunk should backward (first is masked => chunk_tokens==0 => skip).
        assert backward_step_mock.call_count == 1
        assert len(losses) == 1


def test_loss_reduced_is_summed_across_chunks_new_style():
    """验证 TBPTT schedule 会在 chunk 维度累加 new-style 的 [loss_sum, num_tokens] report。"""
    from unittest.mock import MagicMock, patch

    from megatron.core.pipeline_parallel.armt_schedules import armt_forward_backward_no_pipelining

    B, S = 1, 1024
    chunk_size = 512

    raw_batch = {
        "tokens": torch.randint(0, 100, (B, S)),
        "labels": torch.randint(0, 100, (B, S)),
        "loss_mask": torch.ones(B, S),
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        batch = next(data_it)
        forward_calls["n"] += 1

        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            if forward_calls["n"] == 1:
                loss_sum = num_tokens.float()
            else:
                loss_sum = num_tokens.float() * 2.0
            loss_reduced = {"lm loss": torch.cat([loss_sum.clone().detach().view(1), num_tokens.view(1)])}
            return _output_tensor * 0.0 + loss_sum, num_tokens, loss_reduced

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    model = MagicMock()

    with (
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_config", return_value=config),
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_type", return_value=MagicMock()),
        patch("megatron.core.pipeline_parallel.armt_schedules.unwrap_model", return_value=MagicMock()),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_tensor_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_context_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_pipeline_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_position_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_data_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.training.utils.get_batch_on_this_tp_rank",
            side_effect=lambda it: raw_batch,
        ),
    ):
        losses = armt_forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=iter([raw_batch]),
            model=model,
            num_microbatches=1,
            chunk_size=chunk_size,
            seq_length=S,
            micro_batch_size=B,
            forward_only=True,
        )

    assert forward_calls["n"] == S // chunk_size
    assert len(losses) == 1
    report = losses[0]
    assert isinstance(report, dict)
    assert "lm loss" in report
    assert torch.allclose(report["lm loss"], torch.tensor([1536.0, 1024.0]))


def test_scalar_metrics_are_token_weighted_across_chunks():
    """验证 legacy scalar 指标会按 token 数做加权平均（避免 chunk 数量导致偏置）。"""
    from unittest.mock import MagicMock, patch

    from megatron.core.pipeline_parallel.armt_schedules import armt_forward_backward_no_pipelining

    B, S = 1, 8
    chunk_size = 4
    loss_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.float)

    raw_batch = {
        "tokens": torch.randint(0, 100, (B, S)),
        "labels": torch.randint(0, 100, (B, S)),
        "loss_mask": loss_mask,
    }

    forward_calls = {"n": 0}

    def forward_step_func(data_it, model):
        batch = next(data_it)
        forward_calls["n"] += 1
        output_tensor = torch.zeros([], requires_grad=True)

        def _loss_func(_output_tensor):
            num_tokens = batch["loss_mask"].float().sum().to(torch.int)
            scalar = torch.tensor(1.0) if forward_calls["n"] == 1 else torch.tensor(3.0)
            loss_reduced = {"scalar_metric": scalar}
            return _output_tensor * 0.0, num_tokens, loss_reduced

        return output_tensor, _loss_func

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    model = MagicMock()

    with (
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_config", return_value=config),
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_type", return_value=MagicMock()),
        patch("megatron.core.pipeline_parallel.armt_schedules.unwrap_model", return_value=MagicMock()),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_tensor_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_context_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_pipeline_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_position_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_data_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.training.utils.get_batch_on_this_tp_rank",
            side_effect=lambda it: raw_batch,
        ),
    ):
        losses = armt_forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=iter([raw_batch]),
            model=model,
            num_microbatches=1,
            chunk_size=chunk_size,
            seq_length=S,
            micro_batch_size=B,
            forward_only=True,
        )

    assert forward_calls["n"] == S // chunk_size
    report = losses[0]
    expected = (1.0 * 2.0 + 3.0 * 4.0) / 6.0
    assert abs(float(report["scalar_metric"].item()) - expected) < 1e-6


def test_no_loss_from_first_chunk_requires_loss_mask():
    """验证启用 --no-loss-from-first-chunk 时缺少 loss_mask 会显式报错。"""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace

    from megatron.core.pipeline_parallel.armt_schedules import armt_forward_backward_no_pipelining
    from megatron.training import global_vars as training_global_vars

    B, S = 2, 1024
    chunk_size = 512

    # Deliberately omit loss_mask.
    raw_batch = {
        "tokens": torch.randint(0, 100, (B, S)),
        "labels": torch.randint(0, 100, (B, S)),
        "attention_mask": torch.triu(torch.ones(1, 1, S, S), diagonal=1).bool(),
        "position_ids": torch.arange(S).unsqueeze(0).expand(B, -1),
    }

    config = MagicMock()
    config.no_sync_func = None
    config.calculate_per_token_loss = False
    config.finalize_model_grads_func = None

    model = MagicMock()

    def forward_step_func(_data_it, _model):
        # Should not be reached because schedule should fail before first forward.
        raise AssertionError("forward_step_func should not be called when loss_mask is missing")

    with (
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_config", return_value=config),
        patch("megatron.core.pipeline_parallel.armt_schedules.get_model_type", return_value=MagicMock()),
        patch("megatron.core.pipeline_parallel.armt_schedules.unwrap_model", return_value=MagicMock()),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_tensor_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_context_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_pipeline_model_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_position_embedding_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.core.pipeline_parallel.armt_schedules.parallel_state.get_data_parallel_group",
            return_value=MagicMock(),
        ),
        patch(
            "megatron.training.utils.get_batch_on_this_tp_rank",
            side_effect=lambda it: raw_batch,
        ),
    ):
        old_global_args = training_global_vars._GLOBAL_ARGS
        try:
            training_global_vars._GLOBAL_ARGS = SimpleNamespace(
                armt_chunk_size=chunk_size,
                no_loss_from_first_chunk=True,
            )
            import pytest

            with pytest.raises(ValueError, match="no-loss-from-first-chunk requires batch\\['loss_mask'\\]"):
                armt_forward_backward_no_pipelining(
                    forward_step_func=forward_step_func,
                    data_iterator=iter([raw_batch]),
                    model=model,
                    num_microbatches=1,
                    seq_length=S,
                    micro_batch_size=B,
                    forward_only=False,
                )
        finally:
            training_global_vars._GLOBAL_ARGS = old_global_args
