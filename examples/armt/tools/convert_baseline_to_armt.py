"""Convert baseline GPT checkpoint to ARMT checkpoint format."""

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ARMTCheckpointConfig:
    num_mem_tokens: int
    hidden_size: int
    num_layers: int
    d_mem: int
    armt_n_heads: int
    gating: bool
    armt_head_dim: Optional[int] = None


def _get_emb_std(state: Dict) -> float:
    if "embedding.word_embeddings.weight" in state:
        return float(state["embedding.word_embeddings.weight"].std().item())
    return 0.02


def _get_armt_value_dim(config: ARMTCheckpointConfig) -> int:
    if config.armt_n_heads <= 0:
        raise ValueError("armt_n_heads must be > 0")
    if config.armt_head_dim is None:
        if config.hidden_size % config.armt_n_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by armt_n_heads when armt_head_dim is omitted"
            )
        return config.hidden_size
    if config.armt_head_dim <= 0:
        raise ValueError("armt_head_dim must be > 0")
    return config.armt_n_heads * config.armt_head_dim


def convert_baseline_to_armt(
    baseline_ckpt_path: str, armt_ckpt_path: str, config: ARMTCheckpointConfig
):
    baseline = torch.load(baseline_ckpt_path, map_location="cpu")
    if isinstance(baseline, dict) and "model" in baseline:
        baseline_state = baseline["model"]
        wrapper = True
    else:
        baseline_state = baseline
        wrapper = False

    armt_state = dict(baseline_state)

    emb_std = _get_emb_std(baseline_state)
    value_dim = _get_armt_value_dim(config)
    armt_state["memory_embeddings"] = torch.randn(
        config.num_mem_tokens, config.hidden_size
    ) * emb_std

    for layer_idx in range(config.num_layers):
        prefix = f"decoder.layers.{layer_idx}.recurrent_memory_layer."

        armt_state[prefix + "W_mq.weight"] = torch.empty(
            config.d_mem, config.hidden_size
        )
        torch.nn.init.trunc_normal_(armt_state[prefix + "W_mq.weight"], std=0.02)

        armt_state[prefix + "W_mk.weight"] = torch.empty(
            config.d_mem, config.hidden_size
        )
        torch.nn.init.trunc_normal_(armt_state[prefix + "W_mk.weight"], std=0.02)

        armt_state[prefix + "W_mv.weight"] = torch.zeros(value_dim, config.hidden_size)
        armt_state[prefix + "W_mo.weight"] = torch.empty(config.hidden_size, value_dim)
        torch.nn.init.trunc_normal_(armt_state[prefix + "W_mo.weight"], std=0.02)

        if config.gating:
            out_dim = value_dim
        else:
            out_dim = config.armt_n_heads

        armt_state[prefix + "W_mb.weight"] = torch.empty(out_dim, config.hidden_size)
        torch.nn.init.trunc_normal_(armt_state[prefix + "W_mb.weight"], std=0.02)
        armt_state[prefix + "W_mb.bias"] = torch.zeros(out_dim)

    if wrapper:
        torch.save({"model": armt_state}, armt_ckpt_path)
    else:
        torch.save(armt_state, armt_ckpt_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert baseline GPT ckpt to ARMT ckpt")
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--armt-ckpt", type=str, required=True)
    parser.add_argument("--num-mem-tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--armt-d-mem", dest="armt_d_mem", type=int, default=None)
    parser.add_argument("--armt-n-heads", type=int, default=1)
    parser.add_argument("--armt-head-dim", type=int, default=None)
    parser.add_argument("--armt-gating", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = ARMTCheckpointConfig(
        num_mem_tokens=args.num_mem_tokens,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        d_mem=args.armt_d_mem or args.hidden_size,
        armt_n_heads=args.armt_n_heads,
        armt_head_dim=args.armt_head_dim,
        gating=args.armt_gating,
    )
    convert_baseline_to_armt(args.baseline_ckpt, args.armt_ckpt, config)
