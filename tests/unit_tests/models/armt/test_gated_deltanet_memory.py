from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars as training_global_vars
from megatron.core.models.armt.gated_deltanet_memory import GatedDeltaNetMemory
from megatron.core.models.armt.monitoring import finalize_metric_primitives

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class _ScaleNorm(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


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
    num_layers = overrides.pop("num_layers", 2)
    params_dtype = overrides.pop("dtype", torch.float32)
    perform_initialization = overrides.pop("perform_initialization", True)
    config = overrides.pop(
        "config",
        TransformerConfig(
            num_layers=num_layers,
            hidden_size=d_model,
            num_attention_heads=4 if d_model % 4 == 0 else 1,
            ffn_hidden_size=d_model * 4,
            params_dtype=params_dtype,
            perform_initialization=perform_initialization,
        ),
    )
    kwargs = dict(
        config=config,
        d_model=d_model,
        num_mem_tokens=4,
        conv_kernel_size=2,
        key_head_dim=8,
        value_head_dim=8,
        num_key_heads=2,
        num_value_heads=2,
        use_fla_kernel=False,
        use_causal_conv1d=False,
        tbptt_mode=True,
        use_input_pre_norm=False,
        normalization="LayerNorm",
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


def test_gdn_memory_input_pre_norm_defaults_to_enabled():
    layer = GatedDeltaNetMemory(
        config=TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=128,
            params_dtype=torch.float32,
        ),
        d_model=32,
        num_mem_tokens=4,
        conv_kernel_size=2,
        key_head_dim=8,
        value_head_dim=8,
        num_key_heads=2,
        num_value_heads=2,
        use_fla_kernel=False,
        use_causal_conv1d=False,
    )

    assert layer.use_input_pre_norm is True
    assert isinstance(layer.input_pre_norm, torch.nn.LayerNorm)
    assert layer.read_mode == "normal"


def test_gdn_memory_can_enable_input_pre_norm():
    layer = _build_layer(use_input_pre_norm=True, normalization="RMSNorm")
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    retrieved = layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    assert isinstance(layer.input_pre_norm, torch.nn.RMSNorm)
    assert retrieved.shape == (batch_size, 6, layer.d_model)
    assert torch.isfinite(retrieved).all()


def test_gdn_memory_can_skip_redundant_input_pre_norm_on_update():
    layer = _build_layer(use_input_pre_norm=True, normalization="RMSNorm")
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    with patch.object(layer.input_pre_norm, "forward", side_effect=lambda x: x) as forward_mock:
        layer.update_mem(
            torch.randn(batch_size, layer.num_mem_tokens, layer.d_model),
            input_is_sbh=False,
            input_already_pre_normed=True,
        )

    forward_mock.assert_not_called()


def test_gdn_memory_native_init_scales_projection_weights_and_a_log():
    torch.manual_seed(1234)
    layer = _build_layer(
        d_model=256,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=8,
        num_value_heads=8,
        num_layers=8,
    )

    with torch.no_grad():
        layer.in_proj.weight.zero_()
        layer.out_proj.weight.zero_()
        layer.A_log.zero_()
        original_conv = layer.conv1d.weight.clone()
        layer.conv1d.weight.zero_()

    torch.manual_seed(1234)
    layer.reset_parameters()

    in_proj_std = float(layer.in_proj.weight.float().std())
    out_proj_std = float(layer.out_proj.weight.float().std())
    A = layer.A_log.exp()

    assert in_proj_std == pytest.approx(0.02, rel=0.2)
    assert out_proj_std == pytest.approx(0.005, rel=0.3)
    assert out_proj_std < in_proj_std * 0.5
    assert float(A.min()) >= 1.0
    assert float(A.max()) <= 16.0
    assert torch.allclose(layer.conv1d.weight, torch.zeros_like(layer.conv1d.weight))
    assert not torch.allclose(original_conv, torch.zeros_like(original_conv))


def test_gdn_memory_conv_init_override_matches_native_gdn():
    layer = _build_layer(conv_init=0.03)

    with torch.no_grad():
        layer.conv1d.weight.zero_()

    torch.manual_seed(1234)
    layer.reset_parameters()

    assert float(layer.conv1d.weight.abs().max()) <= 0.03 + 1e-6
    assert not torch.allclose(layer.conv1d.weight, torch.zeros_like(layer.conv1d.weight))


def test_gdn_memory_reset_parameters_respects_perform_initialization_flag():
    layer = _build_layer(perform_initialization=False, dtype=torch.float32)

    with torch.no_grad():
        layer.in_proj.weight.fill_(0.25)
        layer.out_proj.weight.fill_(0.5)
        layer.dt_bias.fill_(0.75)
        layer.A_log.fill_(1.25)

    layer.reset_parameters()

    assert layer.config.perform_initialization is False
    assert layer.config.params_dtype == torch.float32
    assert torch.allclose(layer.in_proj.weight, torch.full_like(layer.in_proj.weight, 0.25))
    assert torch.allclose(layer.out_proj.weight, torch.full_like(layer.out_proj.weight, 0.5))
    assert torch.allclose(layer.dt_bias, torch.full_like(layer.dt_bias, 0.75))
    assert torch.allclose(layer.A_log, torch.full_like(layer.A_log, 1.25))


def test_gdn_memory_uses_fla_l2norm_when_fla_kernel_enabled():
    def _fake_l2norm(x):
        return x + 1.0

    with patch(
        "megatron.core.models.armt.gated_deltanet_memory.chunk_gated_delta_rule",
        object(),
    ), patch(
        "megatron.core.models.armt.gated_deltanet_memory.fla_l2norm",
        side_effect=_fake_l2norm,
    ) as l2norm_mock:
        layer = _build_layer(use_fla_kernel=True)
        hidden_states = torch.randn(2, 5, layer.d_model)

        query, key, *_ = layer._project_hidden_states(hidden_states, allow_causal_kernel=False)

    assert query.shape == (2, 5, layer.num_value_heads, layer.key_head_dim)
    assert key.shape == (2, 5, layer.num_value_heads, layer.key_head_dim)
    assert l2norm_mock.call_count == 2


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


def test_gdn_memory_normal_read_mode_uses_direct_state_readout_without_decay():
    layer = _build_layer(read_mode="normal")
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)

    hidden_states = torch.randn(batch_size, 6, layer.d_model)
    memory_input_states = layer._prepare_memory_input(hidden_states)
    query, _, _, gate, _, _ = layer._project_hidden_states(
        hidden_states,
        allow_causal_kernel=True,
        memory_input_states=memory_input_states,
    )
    expected_core_attn_out = torch.einsum(
        "bshk,bhkv->bshv",
        query.to(dtype=layer.recurrent_state.dtype),
        layer.recurrent_state,
    )
    expected_core_attn_out = expected_core_attn_out * (layer.key_head_dim ** -0.5)
    expected = layer._apply_output_gate(expected_core_attn_out.to(dtype=query.dtype), gate)
    state_before_read = layer.recurrent_state.clone()

    with patch.object(
        layer,
        "_run_gated_delta_rule",
        side_effect=AssertionError("normal read mode must not use gated delta rule"),
    ), patch.object(
        layer,
        "_compute_decay",
        side_effect=AssertionError("normal read mode must not use read-side decay"),
    ):
        retrieved = layer.associate(hidden_states, input_is_sbh=False)

    assert torch.equal(layer.recurrent_state, state_before_read)
    assert torch.allclose(retrieved, expected, atol=1e-5, rtol=1e-5)


def test_gdn_memory_buggy_read_mode_still_uses_legacy_delta_rule_path():
    layer = _build_layer(read_mode="buggy")
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)

    with patch.object(layer, "_run_gated_delta_rule", wraps=layer._run_gated_delta_rule) as run_mock:
        retrieved = layer.associate(
            torch.randn(batch_size, 6, layer.d_model),
            input_is_sbh=False,
        )

    assert run_mock.call_count == 1
    assert torch.isfinite(retrieved).all()


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
    expected_metrics = _expected_partitioned_read_metrics(
        second_hidden,
        retrieved,
        layer.num_mem_tokens,
    )

    assert "armt/read/context_retrieved_norm_mean" in metrics
    assert "armt/read/memory_retrieved_norm_mean" in metrics
    assert "armt/read/retrieved_to_context_hidden_ratio" in metrics
    assert "armt/read/retrieved_to_memory_hidden_ratio" in metrics
    assert "armt/write/delta_mem_norm" in metrics
    assert "armt/write/write_gate_mean" in metrics
    assert "armt/state/W_mem_norm" in metrics
    assert "armt/state/log_decay_norm" in metrics
    assert "armt/state/z_norm" not in metrics
    for metric_name, expected_value in expected_metrics.items():
        rel = 1e-4 if "ratio" in metric_name else 1e-5
        assert float(metrics[metric_name]) == pytest.approx(float(expected_value), rel=rel)
    assert "armt/read/retrieved_norm_mean" not in metrics
    assert "armt/read/retrieved_to_hidden_ratio" not in metrics
    assert "armt/read/retrieved_norm_mean/pos_0000" not in metrics
    assert "armt/read/retrieved_to_hidden_ratio/pos_0000" not in metrics


def test_gdn_memory_position_monitoring_metrics_present():
    layer = _build_layer(log_read_position_metrics_to_tensorboard=True)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    second_hidden = torch.randn(batch_size, 6, layer.d_model)
    retrieved = layer.associate(second_hidden, input_is_sbh=False)

    metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())
    expected_metrics = _expected_position_read_metrics(second_hidden, retrieved)

    for metric_name, expected_value in expected_metrics.items():
        rel = 1e-4 if "ratio" in metric_name else 1e-5
        assert float(metrics[metric_name]) == pytest.approx(float(expected_value), rel=rel)


def test_gdn_memory_monitoring_can_be_disabled_for_current_iteration():
    layer = _build_layer(log_read_position_metrics_to_tensorboard=True)
    batch_size = 2
    layer.reset_memory(batch_size=batch_size)
    layer.set_collect_monitoring_for_current_iteration(False)

    layer.update_mem(torch.randn(batch_size, layer.num_mem_tokens, layer.d_model), input_is_sbh=False)
    layer.associate(torch.randn(batch_size, 6, layer.d_model), input_is_sbh=False)

    assert finalize_metric_primitives(layer.consume_monitoring_primitives()) == {}


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
    assert float(metrics["armt/read/context_retrieved_norm_mean"]) > 0.0
    assert float(metrics["armt/read/memory_retrieved_norm_mean"]) > 0.0
    assert float(metrics["armt/read/retrieved_to_context_hidden_ratio"]) > 0.0
    assert float(metrics["armt/read/retrieved_to_memory_hidden_ratio"]) > 0.0


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


def test_gdn_memory_uses_hidden_ratio_even_with_input_pre_norm():
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
