import torch
import pytest

from megatron.core.models.armt.associative_layer import DPFP, AssociativeLayer


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

    def test_associative_layer_reset(self, layer):
        """验证 reset_memory 会按 batch 维度初始化/清零 W_mem 与 z。"""
        batch_size = 2
        layer.reset_memory(batch_size)

        assert layer.W_mem is not None, "W_mem should be initialized"
        assert layer.z is not None, "z should be initialized"
        assert torch.allclose(layer.W_mem, torch.zeros_like(layer.W_mem))
        assert torch.allclose(layer.z, torch.zeros_like(layer.z))
        assert layer.W_mem.shape[0] == batch_size

    def test_associative_layer_associate_first_chunk(self, layer):
        """验证首个 chunk（memory 为空）时 associate 返回全零检索结果。"""
        B, S, H = 2, 128, 256
        layer.reset_memory(B)
        hidden_states = torch.randn(B, S, H)

        out = layer.associate(hidden_states, input_is_sbh=False)

        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_associative_layer_tbptt_detach(self, layer):
        """验证 TBPTT 模式下 update_mem 会切断跨 chunk 梯度（buffer 无 grad_fn）。"""
        B, S, H = 2, 128, 256
        layer.reset_memory(B)
        hidden_states = torch.randn(B, S, H, requires_grad=True)

        layer.update_mem(hidden_states, input_is_sbh=False)

        assert layer.W_mem.grad_fn is None
        assert layer.z.grad_fn is None

    def test_associative_layer_memory_flow(self, layer):
        """验证多次 update_mem 后 W_mem 会累积更新（不再保持初始全零）。"""
        B, S, H = 2, 128, 256
        layer.reset_memory(B)

        # W_mv is initialized to zero; use non-zero weights to observe updates.
        layer.W_mv.weight.data.normal_()
        initial_W_mem = layer.W_mem.clone()

        for _ in range(3):
            hidden_states = torch.randn(B, S, H)
            layer.update_mem(hidden_states, input_is_sbh=False)

        assert not torch.allclose(layer.W_mem, initial_W_mem)
