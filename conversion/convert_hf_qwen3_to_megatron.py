#!/usr/bin/env python3
"""
Convert a HuggingFace Qwen3 checkpoint to Megatron-Core torch_dist checkpoint.
This script is self-contained and does NOT depend on Megatron-Bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint


_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, required=True, help="HuggingFace model directory.")
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint directory.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor model parallel size.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline model parallel size.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target dtype for Megatron parameters.",
    )
    return parser.parse_args()


def _dtype_from_str(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _load_hf_config(hf_path: Path) -> dict:
    config_path = hf_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HF config.json not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_rope_theta(hf_config: dict) -> float:
    if "rope_theta" in hf_config:
        return float(hf_config["rope_theta"])
    rope_params = hf_config.get("rope_parameters", {})
    if "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    return 10000.0


def _get_rotary_percent(hf_config: dict) -> float:
    if "partial_rotary_factor" in hf_config:
        return float(hf_config["partial_rotary_factor"])
    rope_params = hf_config.get("rope_parameters", {})
    if "partial_rotary_factor" in rope_params:
        return float(rope_params["partial_rotary_factor"])
    return 1.0


def _load_hf_weights(hf_path: Path) -> Dict[str, torch.Tensor]:
    weight_files = sorted(hf_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(hf_path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No HF weight files found under: {hf_path}")

    state_dict: Dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            shard = load_file(str(weight_file), device="cpu")
        else:
            shard = torch.load(weight_file, map_location="cpu")
        state_dict.update(shard)
    return state_dict


def _split_along_dim(tensor: torch.Tensor, dim: int, tp_size: int, tp_rank: int) -> torch.Tensor:
    if tp_size == 1:
        return tensor
    size = tensor.shape[dim]
    if size % tp_size != 0:
        raise ValueError(f"Cannot split dim {dim} of size {size} into {tp_size} shards.")
    shard_size = size // tp_size
    start = tp_rank * shard_size
    end = start + shard_size
    return tensor.narrow(dim, start, shard_size).contiguous()


def _merge_qkv_weights(
    config: TransformerConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups or head_num
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    hidden_size = config.hidden_size

    q_reshaped = q.view(head_num, head_size, hidden_size)
    k_reshaped = k.view(num_query_groups, head_size, hidden_size)
    v_reshaped = v.view(num_query_groups, head_size, hidden_size)

    qkv_weights = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        qkv_weights.extend([q_group, k_group, v_group])

    qkv = torch.cat(qkv_weights, dim=0)
    if q.numel() + k.numel() + v.numel() != qkv.numel():
        raise ValueError("QKV merge produced incorrect number of elements.")

    return qkv.reshape([-1, hidden_size])


def _copy_param(param: torch.Tensor, weight: torch.Tensor) -> None:
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch for {param.shape} vs {weight.shape}")
    param.data.copy_(weight.to(dtype=param.dtype, device=param.device))


def _load_megatron_weights(
    model: GPTModel,
    hf_weights: Dict[str, torch.Tensor],
    config: TransformerConfig,
    tp_rank: int,
    tp_size: int,
    tie_word_embeddings: bool,
) -> None:
    for name, param in model.named_parameters():
        if name == "embedding.word_embeddings.weight":
            weight = hf_weights["model.embed_tokens.weight"]
            shard = _split_along_dim(weight, 0, tp_size, tp_rank)
            _copy_param(param, shard)
            continue

        if name == "output_layer.weight":
            if "lm_head.weight" in hf_weights:
                weight = hf_weights["lm_head.weight"]
            else:
                weight = hf_weights["model.embed_tokens.weight"]
            if tie_word_embeddings:
                weight = hf_weights["model.embed_tokens.weight"]
            shard = _split_along_dim(weight, 0, tp_size, tp_rank)
            _copy_param(param, shard)
            continue

        if name == "decoder.final_layernorm.weight":
            weight = hf_weights["model.norm.weight"]
            _copy_param(param, weight)
            continue

        match = _LAYER_RE.match(name)
        if not match:
            continue

        layer_idx = match.group(1)
        sub_name = match.group(2)

        if sub_name == "self_attention.linear_qkv.weight":
            q = hf_weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
            k = hf_weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
            v = hf_weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
            merged = _merge_qkv_weights(config, q, k, v)
            shard = _split_along_dim(merged, 0, tp_size, tp_rank)
            _copy_param(param, shard)
            continue

        if sub_name == "self_attention.linear_qkv.layer_norm_weight":
            weight = hf_weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
            _copy_param(param, weight)
            continue

        if sub_name == "self_attention.q_layernorm.weight":
            weight = hf_weights[f"model.layers.{layer_idx}.self_attn.q_norm.weight"]
            _copy_param(param, weight)
            continue

        if sub_name == "self_attention.k_layernorm.weight":
            weight = hf_weights[f"model.layers.{layer_idx}.self_attn.k_norm.weight"]
            _copy_param(param, weight)
            continue

        if sub_name == "self_attention.linear_proj.weight":
            weight = hf_weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            shard = _split_along_dim(weight, 1, tp_size, tp_rank)
            _copy_param(param, shard)
            continue

        if sub_name == "mlp.linear_fc1.weight":
            gate = hf_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            up = hf_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
            if gate.shape != up.shape:
                raise ValueError("Gate and up projection shapes do not match.")
            # For GLU/SwiGLU, each TP rank stores interleaved [gate, up] portions.
            # See `megatron.core.transformer.mlp.MLP` (fc1_stride=2).
            if tp_size == 1:
                merged = torch.cat([gate, up], dim=0)
            else:
                gate_shard = _split_along_dim(gate, 0, tp_size, tp_rank)
                up_shard = _split_along_dim(up, 0, tp_size, tp_rank)
                merged = torch.cat([gate_shard, up_shard], dim=0)
            _copy_param(param, merged)
            continue

        if sub_name == "mlp.linear_fc1.layer_norm_weight":
            weight = hf_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
            _copy_param(param, weight)
            continue

        if sub_name == "mlp.linear_fc2.weight":
            weight = hf_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            shard = _split_along_dim(weight, 1, tp_size, tp_rank)
            _copy_param(param, shard)
            continue


def _copy_tokenizer_assets(hf_path: Path, iter_dir: Path) -> None:
    tokenizer_dir = iter_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    for name in candidates:
        src = hf_path / name
        if src.is_file():
            shutil.copy2(src, tokenizer_dir / name)


def _add_qk_layernorm_extra_state(model_state: dict, num_layers: int) -> None:
    extra_state_keys = []
    for key in model_state.keys():
        if key.endswith("self_attention.q_layernorm.weight"):
            extra_state_keys.append(("q", key))
        elif key.endswith("self_attention.k_layernorm.weight"):
            extra_state_keys.append(("k", key))

    for kind, key in extra_state_keys:
        match = _LAYER_RE.match(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        extra_key = key.replace(".weight", "._extra_state")
        if extra_key in model_state:
            continue
        sharded_offsets = [(0, layer_idx, num_layers)]
        sharded_obj = make_sharded_object_for_checkpoint(
            torch.empty(0),
            f"decoder.layers.self_attention.{kind}_layernorm._extra_state",
            sharded_offsets=sharded_offsets,
        )
        model_state[extra_key] = sharded_obj


def _init_distributed(tp: int, pp: int) -> None:
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)


def main() -> None:
    args = _parse_args()
    hf_path = Path(args.hf_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _init_distributed(args.tp, args.pp)

    if dist.get_world_size() != args.tp * args.pp:
        raise ValueError(
            f"WORLD_SIZE mismatch: expected {args.tp * args.pp}, got {dist.get_world_size()}"
        )

    hf_config = _load_hf_config(hf_path)
    dtype = _dtype_from_str(args.dtype)

    num_layers = int(hf_config["num_hidden_layers"])
    hidden_size = int(hf_config["hidden_size"])
    ffn_hidden_size = int(hf_config["intermediate_size"])
    num_attention_heads = int(hf_config["num_attention_heads"])
    num_query_groups = int(hf_config.get("num_key_value_heads", num_attention_heads))
    kv_channels = int(hf_config.get("head_dim", hidden_size // num_attention_heads))
    vocab_size = int(hf_config["vocab_size"])
    seq_length = int(hf_config["max_position_embeddings"])
    layernorm_epsilon = float(hf_config.get("rms_norm_eps", 1e-6))
    rope_theta = _get_rope_theta(hf_config)
    rotary_percent = _get_rotary_percent(hf_config)
    tie_word_embeddings = bool(hf_config.get("tie_word_embeddings", True))

    use_cpu_init = not torch.cuda.is_available()
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        hidden_dropout=float(hf_config.get("hidden_dropout", 0.0)),
        attention_dropout=float(hf_config.get("attention_dropout", 0.0)),
        normalization="RMSNorm",
        layernorm_epsilon=layernorm_epsilon,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=use_cpu_init,
    )
    if hasattr(config, "rotary_base"):
        config.rotary_base = rope_theta
    if hasattr(config, "rotary_percent"):
        config.rotary_percent = rotary_percent
    if hasattr(config, "params_dtype"):
        config.params_dtype = dtype
    if hasattr(config, "bf16"):
        config.bf16 = dtype == torch.bfloat16
    if hasattr(config, "fp16"):
        config.fp16 = dtype == torch.float16

    # IMPORTANT: Build the Megatron model with the same TE-backed layer spec used in training.
    # Otherwise, even if shapes/keys match, weight layouts (e.g. fused LN+Linear) can differ and
    # the converted checkpoint will behave like random initialization.
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
    )

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
        pre_process=is_pp_first_stage(pp_group),
        post_process=is_pp_last_stage(pp_group),
        share_embeddings_and_output_weights=tie_word_embeddings,
        position_embedding_type="rope",
        rotary_base=rope_theta,
        rotary_percent=rotary_percent,
    )
    model.eval()

    hf_weights = _load_hf_weights(hf_path)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    _load_megatron_weights(
        model=model,
        hf_weights=hf_weights,
        config=config,
        tp_rank=tp_rank,
        tp_size=args.tp,
        tie_word_embeddings=tie_word_embeddings,
    )

    iter_dir = out_dir / "iter_0000000"
    iter_dir.mkdir(parents=True, exist_ok=True)
    model_state = model.sharded_state_dict(
        metadata={"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    )
    _add_qk_layernorm_extra_state(model_state, num_layers)
    state_dict = {"model": model_state, "iteration": 0}
    dist_checkpointing.save(
        sharded_state_dict=state_dict,
        checkpoint_dir=str(iter_dir),
        sharded_strategy=("torch_dist", 1),
    )

    if dist.get_rank() == 0:
        tracker = out_dir / "latest_checkpointed_iteration.txt"
        tracker.write_text("0", encoding="utf-8")
        _copy_tokenizer_assets(hf_path, iter_dir)

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
