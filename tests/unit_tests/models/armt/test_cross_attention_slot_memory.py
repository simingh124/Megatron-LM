import pytest
import torch

from megatron.core.models.armt.cross_attention_slot_memory import CrossAttentionSlotMemory
from megatron.core.models.armt.monitoring import finalize_metric_primitives


def _build_layer(**overrides):
    kwargs = dict(
        d_model=32,
        num_mem_tokens=4,
        num_slots=6,
        num_heads=4,
        head_dim=8,
        dtype=torch.float32,
        tbptt_mode=True,
        read_attn_backend="sdpa",
    )
    kwargs.update(overrides)
    return CrossAttentionSlotMemory(**kwargs)


def test_cross_attention_slot_memory_defaults_read_backend_to_flash():
    layer = CrossAttentionSlotMemory(
        d_model=32,
        num_mem_tokens=4,
        num_slots=6,
        num_heads=4,
        head_dim=8,
        dtype=torch.float32,
    )

    assert layer.read_attn_backend == "flash"


def test_cross_attention_slot_memory_reset_and_first_chunk_returns_zero():
    layer = _build_layer()
    batch_size = 2
    hidden_states = torch.randn(batch_size, 6, layer.d_model)

    layer.reset_memory(batch_size=batch_size)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)

    assert layer.mem_slots.shape == (batch_size, layer.num_slots, layer.d_model)
    assert torch.allclose(retrieved, torch.zeros_like(retrieved), atol=1e-6)


def test_cross_attention_slot_memory_accepts_explicit_head_dim_without_hidden_size_match():
    layer = _build_layer(
        d_model=32,
        num_heads=3,
        head_dim=5,
    )
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)

    retrieved = layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    assert retrieved.shape == (batch_size, 6, layer.d_model)
    assert layer.mem_slots.shape == (batch_size, layer.num_slots, layer.d_model)


def test_cross_attention_slot_memory_tbptt_detaches_inputs_but_trains_backend_params():
    layer = _build_layer()
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    mem_tokens = torch.randn(batch_size, layer.num_mem_tokens, layer.d_model, requires_grad=True)
    layer.update_mem(mem_tokens, input_is_sbh=False)

    hidden_states = torch.randn(batch_size, 6, layer.d_model, requires_grad=True)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)
    retrieved.float().sum().backward()

    assert mem_tokens.grad is None
    assert layer.W_write_q.weight.grad is not None
    assert layer.W_write_q.weight.grad.abs().sum().item() > 0.0
    assert layer.W_write_v.weight.grad is not None
    assert layer.W_write_v.weight.grad.abs().sum().item() > 0.0
    assert layer.W_gate.weight.grad is not None
    assert layer.W_gate.weight.grad.abs().sum().item() > 0.0


def test_cross_attention_slot_memory_allows_multiple_chunk_backwards_without_retain_graph():
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

    assert layer.W_read_q.weight.grad is not None
    assert layer.W_read_q.weight.grad.abs().sum().item() > 0.0


def test_cross_attention_slot_memory_flash_requires_cuda_device():
    layer = _build_layer(dtype=torch.float16, read_attn_backend="flash")
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer._first_chunk = False

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer.associate(torch.randn(batch_size, 6, layer.d_model, dtype=torch.float16), input_is_sbh=False)


def test_cross_attention_slot_memory_monitoring_metrics_present():
    layer = _build_layer()
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    retrieved = layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

    assert "armt/read/retrieved_norm_mean" in metrics
    assert "armt/read/retrieved_to_hidden_ratio" in metrics
    assert "armt/write/delta_mem_norm" in metrics
    assert "armt/write/write_gate_mean" in metrics
    assert "armt/state/slot_norm" in metrics
    assert "armt/state/slot_usage_entropy" in metrics
    assert "armt/state/max_slot_mass_ratio" in metrics
    expected_retrieved_norm_mean = torch.linalg.vector_norm(retrieved.float(), dim=-1).mean()
    assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(
        float(expected_retrieved_norm_mean), rel=1e-5
    )


def test_cross_attention_slot_memory_memory_state_breakdown_reports_initial_slots():
    layer = _build_layer(d_model=16, num_slots=6, num_heads=4, head_dim=4)

    initial_slots_params = layer.initial_slots.numel()

    assert layer.get_memory_state_breakdown(batch_size=1) == [("initial_slots", initial_slots_params)]
