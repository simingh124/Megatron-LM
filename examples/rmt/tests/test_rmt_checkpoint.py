import os
import tempfile

import torch

from examples.rmt.tools.convert_baseline_to_rmt import (
    RMTCheckpointConfig,
    convert_baseline_to_rmt,
)


def _write_baseline(tmpdir, hidden_size):
    baseline_state = {
        "embedding.word_embeddings.weight": torch.randn(100, hidden_size)
    }
    path = os.path.join(tmpdir, "baseline.pt")
    torch.save(baseline_state, path)
    return path


def test_memory_embeddings_saved():
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 64
        baseline = _write_baseline(tmpdir, hidden_size)
        rmt_path = os.path.join(tmpdir, "rmt.pt")

        config = RMTCheckpointConfig(
            num_mem_tokens=8,
            hidden_size=hidden_size,
        )
        convert_baseline_to_rmt(baseline, rmt_path, config)

        rmt_state = torch.load(rmt_path, map_location="cpu")
        assert "memory_embeddings" in rmt_state
        assert rmt_state["memory_embeddings"].shape == (8, hidden_size)


def test_runtime_memory_state_not_saved():
    with tempfile.TemporaryDirectory() as tmpdir:
        hidden_size = 64
        baseline = _write_baseline(tmpdir, hidden_size)
        rmt_path = os.path.join(tmpdir, "rmt.pt")

        config = RMTCheckpointConfig(
            num_mem_tokens=4,
            hidden_size=hidden_size,
        )
        convert_baseline_to_rmt(baseline, rmt_path, config)

        rmt_state = torch.load(rmt_path, map_location="cpu")
        assert "memory_state" not in rmt_state
