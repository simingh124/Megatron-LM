"""Convert baseline GPT checkpoint to RMT checkpoint format."""

import argparse
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RMTCheckpointConfig:
    num_mem_tokens: int
    hidden_size: int


def _get_emb_std(state: Dict) -> float:
    if "embedding.word_embeddings.weight" in state:
        return float(state["embedding.word_embeddings.weight"].std().item())
    return 0.02


def convert_baseline_to_rmt(baseline_ckpt_path: str, rmt_ckpt_path: str, config: RMTCheckpointConfig):
    baseline = torch.load(baseline_ckpt_path, map_location="cpu")
    if isinstance(baseline, dict) and "model" in baseline:
        baseline_state = baseline["model"]
        wrapper = True
    else:
        baseline_state = baseline
        wrapper = False

    rmt_state = dict(baseline_state)
    emb_std = _get_emb_std(baseline_state)
    rmt_state["memory_embeddings"] = torch.randn(config.num_mem_tokens, config.hidden_size) * emb_std

    if wrapper:
        torch.save({"model": rmt_state}, rmt_ckpt_path)
    else:
        torch.save(rmt_state, rmt_ckpt_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert baseline GPT ckpt to RMT ckpt")
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--rmt-ckpt", type=str, required=True)
    parser.add_argument("--num-mem-tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = RMTCheckpointConfig(
        num_mem_tokens=args.num_mem_tokens,
        hidden_size=args.hidden_size,
    )
    convert_baseline_to_rmt(args.baseline_ckpt, args.rmt_ckpt, config)
