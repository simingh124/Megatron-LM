import os
import random

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.rmt.rmt_model import RMTModel
from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs",
)


def _init_distributed(world_size: int):
    if dist.is_initialized():
        return

    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(29500 + random.randint(1, 1000))

    dist.init_process_group(backend="nccl", init_method="env://")


def _build_model(tp_size: int, seq_len: int):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.float32,
    )
    config.position_embedding_type = "rope"
    config.multi_latent_attention = False

    layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        normalization=None,
    )

    return RMTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=32000,
        max_sequence_length=seq_len,
        num_mem_tokens=4,
        tbptt_mode=True,
        pre_process=True,
        post_process=True,
        parallel_output=True,
    )


def _run_single_step(tp_size: int, seq_len: int, skip_read_memory_from_first_chunk: bool = False):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)

    model = _build_model(tp_size, seq_len).cuda()
    model.train()
    if skip_read_memory_from_first_chunk:
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)

    batch_size = 2
    tokens = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
    labels = tokens.clone()
    loss_mask = torch.ones(batch_size, seq_len, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.triu(
        torch.ones(1, 1, seq_len, seq_len, device="cuda", dtype=torch.bool),
        diagonal=1,
    )

    loss = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
    )
    if loss.dim() != 0:
        loss = loss.mean()
    loss.backward()

    assert torch.isfinite(loss).all()
    destroy_model_parallel()


def _finalize_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class TestRMTTraining:
    @requires_gpu
    def test_rmt_single_step(self):
        _init_distributed(world_size=1)
        try:
            _run_single_step(tp_size=1, seq_len=64)
        finally:
            _finalize_distributed()

    @requires_gpu
    def test_rmt_single_step_without_read_memory_prefix(self):
        _init_distributed(world_size=1)
        try:
            _run_single_step(
                tp_size=1,
                seq_len=64,
                skip_read_memory_from_first_chunk=True,
            )
        finally:
            _finalize_distributed()

    @requires_multi_gpu
    def test_rmt_tp(self):
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 2:
            pytest.skip("TP=2 smoke test requires torchrun with WORLD_SIZE>=2")
        _init_distributed(world_size=world_size)
        try:
            _run_single_step(tp_size=2, seq_len=64)
        finally:
            _finalize_distributed()
