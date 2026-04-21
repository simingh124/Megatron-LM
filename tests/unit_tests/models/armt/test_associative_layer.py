import pytest
import torch
from unittest.mock import patch

from megatron.core.models.armt.associative_layer import DPFP, AssociativeLayer
from megatron.core.models.armt.monitoring import finalize_metric_primitives


class TestDPFP:
    def test_dpfp_output_shape(self):
        """验证 DPFP 的输出维度扩展是否符合 2 * nu * D 规则。"""
        B, H, S, D = 2, 8, 128, 64
        nu = 4
        dpfp = DPFP(nu=nu)
        x = torch.randn(B, H, S, D)

        out = dpfp(x)

        expected_shape = (B, H, S, 2 * nu * D)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    def test_dpfp_deterministic(self):
        """验证 DPFP 为纯确定性映射：同输入必得同输出（无随机性）。"""
        dpfp = DPFP(nu=4)
        x = torch.randn(2, 8, 128, 64)
        out1 = dpfp(x)
        out2 = dpfp(x)
        assert torch.allclose(out1, out2), "DPFP should be deterministic"


class TestAssociativeLayer:
    @pytest.fixture
    def layer(self):
        return AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
        )

    def test_associative_layer_accepts_explicit_head_dim_without_hidden_size_match(self):
        layer = AssociativeLayer(
            d_model=70,
            d_mem=64,
            n_heads=4,
            head_dim=6,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
        )
        batch_size = 2
        layer.reset_memory(batch_size)
        layer.update_mem(
            torch.randn(batch_size, layer.num_mem_tokens, layer.d_model),
            input_is_sbh=False,
        )

        retrieved = layer.associate(torch.randn(batch_size, 8, layer.d_model), input_is_sbh=False)

        assert retrieved.shape == (batch_size, 8, layer.d_model)
        assert layer.W_mem.shape == (
            batch_size,
            layer.n_heads,
            layer.d_key // layer.n_heads,
            layer.head_dim,
        )

    def test_associative_layer_input_pre_norm_defaults_to_disabled(self):
        layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
        )

        assert layer.use_input_pre_norm is False
        assert layer.input_pre_norm is None

    def test_associative_layer_qk_norm_defaults_to_disabled(self):
        layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
        )

        assert layer.use_qk_norm is False

    def test_associative_layer_qk_norm_controls_normalize_calls(self):
        batch_size = 2

        disabled_layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
            use_qk_norm=False,
        )
        disabled_layer.reset_memory(batch_size)
        disabled_layer._first_chunk = False

        with patch(
            "megatron.core.models.armt.associative_layer.F.normalize",
            side_effect=lambda x, *args, **kwargs: x,
        ) as normalize_mock:
            disabled_layer.update_mem(
                torch.randn(batch_size, disabled_layer.num_mem_tokens, disabled_layer.d_model),
                input_is_sbh=False,
            )
            disabled_layer.associate(
                torch.randn(batch_size, 8, disabled_layer.d_model),
                input_is_sbh=False,
            )

        assert normalize_mock.call_count == 0

        enabled_layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
            use_qk_norm=True,
        )
        enabled_layer.reset_memory(batch_size)
        enabled_layer._first_chunk = False

        with patch(
            "megatron.core.models.armt.associative_layer.F.normalize",
            side_effect=lambda x, *args, **kwargs: x,
        ) as normalize_mock:
            enabled_layer.update_mem(
                torch.randn(batch_size, enabled_layer.num_mem_tokens, enabled_layer.d_model),
                input_is_sbh=False,
            )
            enabled_layer.associate(
                torch.randn(batch_size, 8, enabled_layer.d_model),
                input_is_sbh=False,
            )

        assert normalize_mock.call_count == 2

    def test_associative_layer_can_enable_input_pre_norm(self):
        layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
            use_input_pre_norm=True,
            normalization="RMSNorm",
        )
        batch_size = 2
        layer.reset_memory(batch_size)
        layer.update_mem(
            torch.randn(batch_size, layer.num_mem_tokens, layer.d_model),
            input_is_sbh=False,
        )
        retrieved = layer.associate(torch.randn(batch_size, 8, layer.d_model), input_is_sbh=False)

        assert isinstance(layer.input_pre_norm, torch.nn.RMSNorm)
        assert retrieved.shape == (batch_size, 8, layer.d_model)
        assert torch.isfinite(retrieved).all()

    def test_associative_layer_can_skip_redundant_input_pre_norm_on_update(self):
        layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
            use_input_pre_norm=True,
            normalization="RMSNorm",
        )
        batch_size = 2
        layer.reset_memory(batch_size)

        with patch.object(layer.input_pre_norm, "forward", side_effect=lambda x: x) as forward_mock:
            layer.update_mem(
                torch.randn(batch_size, layer.num_mem_tokens, layer.d_model),
                input_is_sbh=False,
                input_already_pre_normed=True,
            )

        forward_mock.assert_not_called()

    def test_associative_layer_reset(self, layer):
        """验证 reset_memory 会按 batch 维度初始化/清零 W_mem 与 z。"""
        batch_size = 2
        layer.reset_memory(batch_size)

        assert layer.W_mem is not None, "W_mem should be initialized"
        assert layer.z is not None, "z should be initialized"
        assert torch.allclose(layer.W_mem, torch.zeros_like(layer.W_mem))
        assert torch.allclose(layer.z, torch.zeros_like(layer.z))
        assert layer.W_mem.shape[0] == batch_size

    def test_memory_state_breakdown_uses_pre_dpfp_memory_width(self):
        layer = AssociativeLayer(
            d_model=256,
            d_mem=64,
            n_heads=4,
            head_dim=6,
            nu=4,
            tbptt_mode=True,
            dtype=torch.float32,
        )

        assert layer.get_memory_state_breakdown(batch_size=1) == [
            ("W_mem", layer.n_heads * (layer.d_mem // layer.n_heads) * layer.head_dim)
        ]

    def test_associative_layer_associate_first_chunk(self, layer):
        """验证首个 chunk（memory 为空）时 associate 返回全零检索结果。"""
        B, S, H = 2, 128, 256
        layer.reset_memory(B)
        hidden_states = torch.randn(B, S, H)

        out = layer.associate(hidden_states, input_is_sbh=False)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_associative_layer_first_chunk_does_not_emit_read_metrics(self, layer):
        """验证首个 chunk 的 read 监控会被跳过，不污染 armt/read/* 指标。"""
        batch_size, hidden_size = 2, 256
        layer.reset_memory(batch_size)

        layer.associate(torch.randn(batch_size, 32, hidden_size), input_is_sbh=False)

        metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

        assert "armt/read/retrieved_norm_mean" not in metrics
        assert "armt/read/retrieved_to_hidden_ratio" not in metrics

    def test_associative_layer_tbptt_detaches_inputs_but_trains_write_weights(self, layer):
        """验证 TBPTT 模式：

        - 允许后续 chunk 的 loss 回传到 memory-write 参数（W_mk/W_mv/W_mb）
        - 但不会回传到上一 chunk 的 mem_tokens 激活（避免跨 chunk 反传 & 无需 retain_graph）
        """
        B, H = 2, 256
        layer.reset_memory(B)

        # Ensure memory updates are non-trivial (avoid near-zero retrieval).
        layer.W_mv.weight.data.normal_()

        mem_tokens = torch.randn(B, layer.num_mem_tokens, H, requires_grad=True)
        layer.update_mem(mem_tokens, input_is_sbh=False)

        hidden_states = torch.randn(B, 32, H, requires_grad=True)
        retrieved = layer.associate(hidden_states, input_is_sbh=False)
        loss = retrieved.float().sum()
        loss.backward()

        assert mem_tokens.grad is None
        assert layer.W_mk.weight.grad is not None
        assert layer.W_mk.weight.grad.abs().sum().item() > 0.0
        assert layer.W_mv.weight.grad is not None
        assert layer.W_mv.weight.grad.abs().sum().item() > 0.0
        assert layer.W_mb.weight.grad is not None
        assert layer.W_mb.weight.grad.abs().sum().item() > 0.0

    def test_associative_layer_tbptt_allows_multiple_chunk_backwards_without_retain_graph(
        self, layer
    ):
        """验证 TBPTT 下可以连续进行多次 chunk-wise backward，不需要 retain_graph。"""
        B, H = 2, 256
        layer.reset_memory(B)

        # Ensure memory updates are non-trivial (avoid near-zero retrieval).
        layer.W_mv.weight.data.normal_()

        # Chunk 1: simulate a token loss that does not depend on update_mem().
        hidden_states = torch.randn(B, 32, H, requires_grad=True)
        (hidden_states.float() ** 2).sum().backward()

        # Chunk 1 update -> Chunk 2 loss backprops into memory-write weights.
        mem_tokens = torch.randn(B, layer.num_mem_tokens, H, requires_grad=True)
        layer.update_mem(mem_tokens, input_is_sbh=False)
        hidden_states = torch.randn(B, 32, H, requires_grad=True)
        layer.associate(hidden_states, input_is_sbh=False).float().sum().backward()

        # Chunk 2 update -> Chunk 3 loss should be able to backward again without errors.
        mem_tokens = torch.randn(B, layer.num_mem_tokens, H, requires_grad=True)
        layer.update_mem(mem_tokens, input_is_sbh=False)
        hidden_states = torch.randn(B, 32, H, requires_grad=True)
        layer.associate(hidden_states, input_is_sbh=False).float().sum().backward()

        assert layer.W_mk.weight.grad is not None
        assert layer.W_mk.weight.grad.abs().sum().item() > 0.0

    def test_associative_layer_memory_flow(self, layer):
        """验证多次 update_mem 后 W_mem 会累积更新（不再保持初始全零）。"""
        B, S, H = 2, 128, 256
        layer.reset_memory(B)

        # Use non-trivial weights to make memory updates observable.
        layer.W_mv.weight.data.normal_()
        initial_W_mem = layer.W_mem.clone()

        for _ in range(3):
            hidden_states = torch.randn(B, S, H)
            layer.update_mem(hidden_states, input_is_sbh=False)

        assert not torch.allclose(layer.W_mem, initial_W_mem)

    def test_reset_memory_detaches_recurrent_state_between_microbatches(self):
        """验证 reset_memory 会切断上一微批挂在 memory state 上的计算图。"""
        layer = AssociativeLayer(
            d_model=128,
            d_mem=64,
            n_heads=4,
            nu=4,
            tbptt_mode=False,
        )
        batch_size = 2
        layer.reset_memory(batch_size)

        mem_tokens = torch.randn(
            batch_size,
            layer.num_mem_tokens,
            layer.d_model,
            requires_grad=True,
        )
        layer.update_mem(mem_tokens, input_is_sbh=False)

        assert layer.W_mem.grad_fn is not None
        if layer.use_denom:
            assert layer.z.grad_fn is not None

        layer.reset_memory()

        assert layer.W_mem.grad_fn is None
        if layer.use_denom:
            assert layer.z.grad_fn is None

    def test_associative_layer_monitoring_metrics(self, layer):
        """验证 AssociativeLayer 会累计并导出 read/write/state 监控指标。"""
        batch_size, hidden_size = 2, 256
        layer.reset_memory(batch_size)
        layer.W_mv.weight.data.normal_()

        layer.associate(torch.randn(batch_size, 32, hidden_size), input_is_sbh=False)
        mem_tokens = torch.randn(batch_size, layer.num_mem_tokens, hidden_size)
        layer.update_mem(mem_tokens, input_is_sbh=False)
        second_hidden_states = torch.randn(batch_size, 32, hidden_size)
        second_retrieved = layer.associate(second_hidden_states, input_is_sbh=False)

        metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

        assert "armt/read/retrieved_norm_mean" in metrics
        assert "armt/read/retrieved_to_hidden_ratio" in metrics
        assert "armt/write/delta_mem_norm" in metrics
        assert "armt/write/write_gate_mean" in metrics
        assert "armt/state/W_mem_norm" in metrics
        assert "armt/state/z_norm" in metrics
        expected_retrieved_norm_mean = torch.linalg.vector_norm(
            second_retrieved.float(), dim=-1
        ).mean()
        expected_retrieved_to_hidden_ratio = (
            torch.linalg.vector_norm(second_retrieved.float(), dim=-1).sum()
            / torch.linalg.vector_norm(second_hidden_states.float(), dim=-1).sum()
        )
        assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(
            float(expected_retrieved_norm_mean), rel=1e-5
        )
        assert float(metrics["armt/read/retrieved_to_hidden_ratio"]) == pytest.approx(
            float(expected_retrieved_to_hidden_ratio), rel=1e-4
        )
        assert float(metrics["armt/write/delta_mem_norm"]) >= 0.0
        assert float(metrics["armt/state/W_mem_norm"]) >= 0.0
        assert layer.consume_monitoring_primitives() == {}

    def test_associative_layer_monitoring_skips_z_norm_without_denom(self):
        """验证 use_denom=False 时不会导出 z_norm。"""
        layer = AssociativeLayer(
            d_model=128,
            d_mem=64,
            n_heads=4,
            head_dim=16,
            nu=4,
            use_denom=False,
            tbptt_mode=True,
        )
        batch_size = 2
        layer.reset_memory(batch_size)
        layer.update_mem(
            torch.randn(batch_size, layer.num_mem_tokens, layer.d_model),
            input_is_sbh=False,
        )
        metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

        assert "armt/state/z_norm" not in metrics
