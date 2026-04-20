import pytest
import torch

from megatron.core.models.armt.cross_attention_slot_memory import CrossAttentionSlotMemory
from megatron.core.models.armt.monitoring import finalize_metric_primitives
from megatron.core.transformer.transformer_config import TransformerConfig


class _ScaleNorm(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


def _build_config(hidden_size: int, *, dtype: torch.dtype = torch.float32, num_layers: int = 2):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4 if hidden_size % 4 == 0 else 1,
        ffn_hidden_size=hidden_size * 4,
        params_dtype=dtype,
    )


def _expected_partitioned_read_metrics(
    hidden_states: torch.Tensor,
    retrieved_states: torch.Tensor,
    num_mem_tokens: int,
):
    memory_tokens = min(num_mem_tokens, hidden_states.shape[1])
    context_tokens = hidden_states.shape[1] - memory_tokens
    expected = {}

    if context_tokens > 0:
        context_hidden = hidden_states[:, :context_tokens, :]
        context_retrieved = retrieved_states[:, :context_tokens, :]
        expected["armt/read/context_retrieved_norm_mean"] = torch.linalg.vector_norm(
            context_retrieved.float(), dim=-1
        ).mean()
        expected["armt/read/retrieved_to_context_hidden_ratio"] = (
            torch.linalg.vector_norm(context_retrieved.float(), dim=-1).sum()
            / torch.linalg.vector_norm(context_hidden.float(), dim=-1).sum()
        )

    if memory_tokens > 0:
        memory_hidden = hidden_states[:, context_tokens:, :]
        memory_retrieved = retrieved_states[:, context_tokens:, :]
        expected["armt/read/memory_retrieved_norm_mean"] = torch.linalg.vector_norm(
            memory_retrieved.float(), dim=-1
        ).mean()
        expected["armt/read/retrieved_to_memory_hidden_ratio"] = (
            torch.linalg.vector_norm(memory_retrieved.float(), dim=-1).sum()
            / torch.linalg.vector_norm(memory_hidden.float(), dim=-1).sum()
        )

    return expected


def _expected_position_read_metrics(
    hidden_states: torch.Tensor,
    retrieved_states: torch.Tensor,
):
    hidden_norms = torch.linalg.vector_norm(hidden_states.float(), dim=-1)
    retrieved_norms = torch.linalg.vector_norm(retrieved_states.float(), dim=-1)
    expected = {}

    for position in range(hidden_states.shape[1]):
        position_tag = f"pos_{position:04d}"
        expected[f"armt/read/retrieved_norm_mean/{position_tag}"] = retrieved_norms[
            :, position
        ].mean()
        expected[f"armt/read/retrieved_to_hidden_ratio/{position_tag}"] = (
            retrieved_norms[:, position].sum() / hidden_norms[:, position].sum()
        )

    return expected


def _build_layer(**overrides):
    d_model = overrides.get("d_model", overrides.get("hidden_size", 32))
    params_dtype = overrides.pop("dtype", torch.float32)
    num_layers = overrides.pop("num_layers", 2)
    kwargs = dict(
        config=overrides.pop("config", _build_config(d_model, dtype=params_dtype, num_layers=num_layers)),
        d_model=d_model,
        num_mem_tokens=4,
        num_slots=6,
        num_heads=4,
        head_dim=8,
        dtype=params_dtype,
        tbptt_mode=True,
        read_attn_backend="sdpa",
        use_qk_norm=False,
        use_input_pre_norm=False,
        normalization="LayerNorm",
    )
    kwargs.update(overrides)
    return CrossAttentionSlotMemory(**kwargs)


def test_cross_attention_slot_memory_defaults_read_backend_to_flash():
    layer = CrossAttentionSlotMemory(
        config=_build_config(32, dtype=torch.float32),
        d_model=32,
        num_mem_tokens=4,
        num_slots=6,
        num_heads=4,
        head_dim=8,
        dtype=torch.float32,
    )

    assert layer.read_attn_backend == "flash"


def test_cross_attention_slot_memory_norm_switches_default_to_disabled():
    layer = _build_layer()

    assert layer.use_qk_norm is False
    assert layer.use_input_pre_norm is False
    assert layer.input_pre_norm is None
    assert layer.read_q_norm is None
    assert layer.read_k_norm is None
    assert layer.write_q_norm is None
    assert layer.write_k_norm is None


def test_cross_attention_slot_memory_native_init_matches_megatron_defaults():
    torch.manual_seed(1234)
    layer = _build_layer(d_model=128, num_layers=8)

    read_q_std = float(layer.W_read_q.weight.float().std())
    read_o_std = float(layer.W_read_o.weight.float().std())
    slot_std = float(layer.initial_slots.float().std())

    assert read_q_std == pytest.approx(0.02, rel=0.2)
    assert read_o_std == pytest.approx(0.005, rel=0.3)
    assert slot_std == pytest.approx(0.02, rel=0.2)
    assert torch.allclose(layer.W_gate.bias, torch.zeros_like(layer.W_gate.bias))


def test_cross_attention_slot_memory_can_enable_norm_switches():
    layer = _build_layer(
        use_qk_norm=True,
        use_input_pre_norm=True,
        normalization="RMSNorm",
    )

    assert layer.use_qk_norm is True
    assert layer.use_input_pre_norm is True
    assert isinstance(layer.input_pre_norm, torch.nn.RMSNorm)
    assert isinstance(layer.read_q_norm, torch.nn.RMSNorm)
    assert isinstance(layer.read_k_norm, torch.nn.RMSNorm)
    assert isinstance(layer.write_q_norm, torch.nn.RMSNorm)
    assert isinstance(layer.write_k_norm, torch.nn.RMSNorm)


def test_cross_attention_slot_memory_norm_switches_preserve_shape_and_finiteness():
    layer = _build_layer(
        use_qk_norm=True,
        use_input_pre_norm=True,
        normalization="RMSNorm",
    )
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    retrieved = layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    assert retrieved.shape == (batch_size, 6, layer.d_model)
    assert torch.isfinite(retrieved).all()


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
    hidden_states = torch.randn(batch_size, 6, layer.d_model)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())
    expected_metrics = _expected_partitioned_read_metrics(
        hidden_states,
        retrieved,
        layer.num_mem_tokens,
    )

    assert "armt/read/context_retrieved_norm_mean" in metrics
    assert "armt/read/memory_retrieved_norm_mean" in metrics
    assert "armt/read/retrieved_to_context_hidden_ratio" in metrics
    assert "armt/read/retrieved_to_memory_hidden_ratio" in metrics
    assert "armt/write/delta_mem_norm" in metrics
    assert "armt/write/write_gate_mean" in metrics
    assert "armt/state/slot_norm" in metrics
    assert "armt/state/slot_usage_entropy" in metrics
    assert "armt/state/max_slot_mass_ratio" in metrics
    for metric_name, expected_value in expected_metrics.items():
        rel = 1e-4 if "ratio" in metric_name else 1e-5
        assert float(metrics[metric_name]) == pytest.approx(float(expected_value), rel=rel)
    assert "armt/read/retrieved_norm_mean" not in metrics
    assert "armt/read/retrieved_to_hidden_ratio" not in metrics
    assert "armt/read/retrieved_norm_mean/pos_0000" not in metrics
    assert "armt/read/retrieved_to_hidden_ratio/pos_0000" not in metrics


def test_cross_attention_slot_memory_position_monitoring_metrics_present():
    layer = _build_layer(log_read_position_metrics_to_tensorboard=True)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    hidden_states = torch.randn(batch_size, 6, layer.d_model)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())
    expected_metrics = _expected_position_read_metrics(hidden_states, retrieved)

    for metric_name, expected_value in expected_metrics.items():
        rel = 1e-4 if "ratio" in metric_name else 1e-5
        assert float(metrics[metric_name]) == pytest.approx(float(expected_value), rel=rel)


def test_cross_attention_slot_memory_monitoring_can_be_disabled_for_current_iteration():
    layer = _build_layer(log_read_position_metrics_to_tensorboard=True)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer.set_collect_monitoring_for_current_iteration(False)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    assert finalize_metric_primitives(layer.consume_monitoring_primitives()) == {}


def test_cross_attention_slot_memory_memory_state_breakdown_reports_initial_slots():
    layer = _build_layer(d_model=16, num_slots=6, num_heads=4, head_dim=4)

    initial_slots_params = layer.initial_slots.numel()

    assert layer.get_memory_state_breakdown(batch_size=1) == [("initial_slots", initial_slots_params)]


def test_cross_attention_slot_memory_uses_hidden_ratio_even_with_input_pre_norm():
    layer = _build_layer(use_input_pre_norm=True)
    layer.input_pre_norm = _ScaleNorm(2.0)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    hidden_states = torch.randn(batch_size, 6, layer.d_model)
    retrieved = layer.associate(hidden_states, input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())
    expected_metrics = _expected_partitioned_read_metrics(
        hidden_states,
        retrieved,
        layer.num_mem_tokens,
    )

    assert float(metrics["armt/read/retrieved_to_context_hidden_ratio"]) == pytest.approx(
        float(expected_metrics["armt/read/retrieved_to_context_hidden_ratio"]),
        rel=1e-4,
    )
    assert "armt/read/retrieved_to_context_memory_input_ratio" not in metrics
