import math
import os
import tempfile

import pytest
import torch

from examples.armt.tools.convert_baseline_to_armt import (
    ARMTCheckpointConfig,
    convert_baseline_to_armt,
)


def _write_baseline(tmpdir, hidden_size):
    baseline_state = {
        "embedding.word_embeddings.weight": torch.randn(100, hidden_size)
    }
    path = os.path.join(tmpdir, "baseline.pt")
    torch.save(baseline_state, path)
    return path


def test_memory_embeddings_saved():
    """验证 baseline->ARMT 转换后，checkpoint 中包含 memory_embeddings 且 shape 正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 64
        baseline = _write_baseline(tmpdir, hidden_size)
        armt_path = os.path.join(tmpdir, "armt.pt")

        config = ARMTCheckpointConfig(
            num_mem_tokens=8,
            hidden_size=hidden_size,
            num_layers=2,
            d_mem=hidden_size,
            armt_n_heads=2,
            gating=False,
        )
        convert_baseline_to_armt(baseline, armt_path, config)

        armt_state = torch.load(armt_path, map_location="cpu")
        assert "memory_embeddings" in armt_state
        assert armt_state["memory_embeddings"].shape == (8, hidden_size)


def test_W_mem_z_not_saved():
    """验证 runtime buffer（W_mem/z）不会被写入 checkpoint（应当不存在于 state_dict keys）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 64
        baseline = _write_baseline(tmpdir, hidden_size)
        armt_path = os.path.join(tmpdir, "armt.pt")

        config = ARMTCheckpointConfig(
            num_mem_tokens=8,
            hidden_size=hidden_size,
            num_layers=1,
            d_mem=hidden_size,
            armt_n_heads=2,
            gating=False,
        )
        convert_baseline_to_armt(baseline, armt_path, config)

        armt_state = torch.load(armt_path, map_location="cpu")
        keys = set(armt_state.keys())
        assert not any("W_mem" in k for k in keys)
        assert not any(k.endswith(".z") for k in keys)


def test_baseline_to_armt_conversion():
    """验证转换工具会按 Megatron 默认方案补齐 recurrent_memory_layer 权重初始化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 32
        baseline = _write_baseline(tmpdir, hidden_size)
        armt_path = os.path.join(tmpdir, "armt.pt")

        config = ARMTCheckpointConfig(
            num_mem_tokens=4,
            hidden_size=hidden_size,
            num_layers=1,
            d_mem=hidden_size,
            armt_n_heads=1,
            gating=False,
        )
        convert_baseline_to_armt(baseline, armt_path, config)

        armt_state = torch.load(armt_path, map_location="cpu")
        assert "decoder.layers.0.recurrent_memory_layer.W_mq.weight" in armt_state
        assert "decoder.layers.0.recurrent_memory_layer.W_mv.weight" in armt_state
        assert float(
            armt_state["decoder.layers.0.recurrent_memory_layer.W_mq.weight"].float().std()
        ) == pytest.approx(config.init_method_std, rel=0.3)
        assert float(
            armt_state["decoder.layers.0.recurrent_memory_layer.W_mv.weight"].float().std()
        ) == pytest.approx(config.init_method_std, rel=0.3)
        assert float(
            armt_state["decoder.layers.0.recurrent_memory_layer.W_mo.weight"].float().std()
        ) == pytest.approx(config.init_method_std / math.sqrt(2 * config.num_layers), rel=0.3)
        assert torch.allclose(
            armt_state["decoder.layers.0.recurrent_memory_layer.W_mb.bias"],
            torch.zeros(config.armt_n_heads),
        )


def test_baseline_to_armt_conversion_supports_explicit_head_dim():
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 32
        baseline = _write_baseline(tmpdir, hidden_size)
        armt_path = os.path.join(tmpdir, "armt.pt")

        config = ARMTCheckpointConfig(
            num_mem_tokens=4,
            hidden_size=hidden_size,
            num_layers=1,
            d_mem=30,
            armt_n_heads=3,
            armt_head_dim=5,
            gating=True,
        )
        convert_baseline_to_armt(baseline, armt_path, config)

        armt_state = torch.load(armt_path, map_location="cpu")
        assert armt_state["decoder.layers.0.recurrent_memory_layer.W_mv.weight"].shape == (
            15,
            hidden_size,
        )
        assert armt_state["decoder.layers.0.recurrent_memory_layer.W_mo.weight"].shape == (
            hidden_size,
            15,
        )
        assert armt_state["decoder.layers.0.recurrent_memory_layer.W_mb.weight"].shape == (
            15,
            hidden_size,
        )
