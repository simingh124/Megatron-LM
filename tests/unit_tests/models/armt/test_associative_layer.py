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
