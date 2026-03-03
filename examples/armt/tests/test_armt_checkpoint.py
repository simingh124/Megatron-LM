import os
import tempfile

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
    """验证转换工具会补齐 associative_layer 权重，并保证 W_mv 为全零初始化（residual-friendly）。"""
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
        assert "decoder.layers.0.associative_layer.W_mq.weight" in armt_state
        assert "decoder.layers.0.associative_layer.W_mv.weight" in armt_state
        assert torch.allclose(
            armt_state["decoder.layers.0.associative_layer.W_mv.weight"],
            torch.zeros(hidden_size, hidden_size),
        )
