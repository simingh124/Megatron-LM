from unittest.mock import patch

import pytest
import torch

from megatron.training import global_vars as training_global_vars
from megatron.core.models.armt.gated_deltanet_memory import GatedDeltaNetMemory
from megatron.core.models.armt.monitoring import finalize_metric_primitives

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _build_layer(**overrides):
    kwargs = dict(
        d_model=32,
        num_mem_tokens=4,
        conv_kernel_size=2,
        key_head_dim=8,
        value_head_dim=8,
        num_key_heads=2,
        num_value_heads=2,
        use_fla_kernel=False,
        use_causal_conv1d=False,
        dtype=torch.float32,
        tbptt_mode=True,
    )
    kwargs.update(overrides)
    return GatedDeltaNetMemory(**kwargs)


def test_gdn_memory_reset_and_first_chunk_returns_zero():
    layer = _build_layer()
    batch_size = 2
    hidden_states = torch.randn(batch_size, 6, layer.d_model)

    layer.reset_memory(batch_size=batch_size)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)

    assert layer.recurrent_state.shape == (
        batch_size,
        layer.num_value_heads,
        layer.key_head_dim,
        layer.value_head_dim,
    )
    assert torch.allclose(retrieved, torch.zeros_like(retrieved), atol=1e-6)


def test_gdn_memory_tbptt_detaches_inputs_but_trains_backend_params():
    layer = _build_layer()
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    mem_tokens = torch.randn(batch_size, layer.num_mem_tokens, layer.d_model, requires_grad=True)
    layer.update_mem(mem_tokens, input_is_sbh=False)

    hidden_states = torch.randn(batch_size, 6, layer.d_model, requires_grad=True)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)
    retrieved.float().sum().backward()

    assert mem_tokens.grad is None
    assert layer.in_proj.weight.grad is not None
    assert layer.in_proj.weight.grad.abs().sum().item() > 0.0
    assert layer.conv1d.weight.grad is not None
    assert layer.conv1d.weight.grad.abs().sum().item() > 0.0


def test_gdn_memory_allows_multiple_chunk_backwards_without_retain_graph():
    layer = _build_layer()
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    first_mem = torch.randn(batch_size, layer.num_mem_tokens, layer.d_model, requires_grad=True)
    layer.update_mem(first_mem, input_is_sbh=False)
    layer.associate(
        torch.randn(batch_size, 6, layer.d_model, requires_grad=True),
        input_is_sbh=False,
    ).float().sum().backward()

    second_mem = torch.randn(batch_size, layer.num_mem_tokens, layer.d_model, requires_grad=True)
    layer.update_mem(second_mem, input_is_sbh=False)
    layer.associate(
        torch.randn(batch_size, 6, layer.d_model, requires_grad=True),
        input_is_sbh=False,
    ).float().sum().backward()

    assert layer.out_proj.weight.grad is not None
    assert layer.out_proj.weight.grad.abs().sum().item() > 0.0


def test_gdn_reset_memory_detaches_recurrent_state_between_microbatches():
    layer = _build_layer(tbptt_mode=False)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    mem_tokens = torch.randn(batch_size, layer.num_mem_tokens, layer.d_model, requires_grad=True)
    layer.update_mem(mem_tokens, input_is_sbh=False)

    assert layer.recurrent_state.grad_fn is not None

    layer.reset_memory()

    assert layer.recurrent_state.grad_fn is None


def test_gdn_memory_monitoring_metrics_present():
    layer = _build_layer()
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    second_hidden = torch.randn(batch_size, 6, layer.d_model)
    retrieved = layer.associate(second_hidden, input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

    assert "armt/read/retrieved_norm_mean" in metrics
    assert "armt/read/retrieved_to_hidden_ratio" in metrics
    assert "armt/write/delta_mem_norm" in metrics
    assert "armt/write/write_gate_mean" in metrics
    assert "armt/state/W_mem_norm" in metrics
    assert "armt/state/log_decay_norm" in metrics
    assert "armt/state/z_norm" not in metrics
    expected_retrieved_norm_mean = torch.linalg.vector_norm(retrieved.float(), dim=-1).mean()
    assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(
        float(expected_retrieved_norm_mean), rel=1e-5
    )


@requires_cuda
def test_gdn_memory_bf16_read_preserves_state_and_emits_nonzero_read_metrics():
    layer = _build_layer(dtype=torch.bfloat16, tbptt_mode=False).cuda()
    batch_size = 2
    device = torch.device("cuda")
    layer.reset_memory(batch_size=batch_size, device=device)

    layer.update_mem(
        torch.randn(
            batch_size,
            layer.num_mem_tokens,
            layer.d_model,
            device=device,
            dtype=torch.bfloat16,
        ),
        input_is_sbh=False,
    )
    state_norm_before_read = layer.recurrent_state.float().norm()
    assert float(state_norm_before_read) > 0.0

    retrieved = layer.associate(
        torch.randn(batch_size, 6, layer.d_model, device=device, dtype=torch.bfloat16),
        input_is_sbh=False,
    )
    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

    assert float(layer.recurrent_state.float().norm()) > 0.0
    assert float(torch.linalg.vector_norm(retrieved.float(), dim=-1).mean()) > 0.0
    assert float(metrics["armt/read/retrieved_norm_mean"]) > 0.0
    assert float(metrics["armt/read/retrieved_to_hidden_ratio"]) > 0.0


def test_gdn_memory_requires_fla_when_requested():
    with patch("megatron.core.models.armt.gated_deltanet_memory.chunk_gated_delta_rule", None):
        with pytest.raises(ImportError, match="FLA"):
            _build_layer(use_fla_kernel=True)


def test_gdn_memory_requires_causal_conv1d_when_requested():
    with patch("megatron.core.models.armt.gated_deltanet_memory.causal_conv1d_fn", None):
        with pytest.raises(ImportError, match="causal_conv1d"):
            _build_layer(use_causal_conv1d=True)


def test_gdn_memory_ignores_legacy_schedule_attr_when_using_causal_conv1d():
    def _fake_causal_conv1d_fn(x, weight, bias, activation):
        del weight, bias, activation
        return x + 7.0

    old_global_args = training_global_vars._GLOBAL_ARGS
    try:
        training_global_vars._GLOBAL_ARGS = type(
            "LegacyArgs", (), {"use_recurrent_tbptt": True}
        )()
        with patch(
            "megatron.core.models.armt.gated_deltanet_memory.causal_conv1d_fn",
            side_effect=_fake_causal_conv1d_fn,
        ) as causal_conv_mock:
            layer = _build_layer(use_causal_conv1d=True)
            with torch.no_grad():
                layer.conv1d.weight.zero_()

            qkv = torch.randn(2, 5, layer.conv_dim)
            output = layer._apply_conv(qkv, allow_causal_kernel=True)

            causal_conv_mock.assert_called_once()
            assert torch.allclose(output, qkv + 7.0)
    finally:
        training_global_vars._GLOBAL_ARGS = old_global_args
