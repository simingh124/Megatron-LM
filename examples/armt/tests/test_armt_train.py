import os
import random
import math
from importlib.util import find_spec

import torch
import torch.distributed as dist
import pytest

from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.armt.armt_model import ARMTModel
from megatron.core.models.armt.armt_layer_specs import get_armt_layer_spec

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs",
)

HAS_FLA = find_spec("fla") is not None
HAS_CAUSAL_CONV1D = find_spec("causal_conv1d") is not None


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


def _build_model(
    tp_size: int,
    seq_len: int,
    *,
    recurrent_chunk_size: int = 64,
    full_attn_window_size: int | None = None,
    recurrent_tbptt_mode: bool = True,
    recurrent_memory_backend: str = "associative",
    recurrent_gdn_use_fla_kernel: bool = True,
    recurrent_gdn_use_causal_conv1d: bool = True,
):
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

    layer_spec = get_armt_layer_spec(
        transformer_impl="local",
        num_mem_tokens=4,
        d_mem=64,
        armt_n_heads=1,
        nu=3,
        use_denom=True,
        gating=False,
        correction=True,
        tbptt_mode=recurrent_tbptt_mode,
        recurrent_chunk_size=recurrent_chunk_size,
        full_attn_window_size=full_attn_window_size,
        recurrent_memory_backend=recurrent_memory_backend,
        recurrent_gdn_use_fla_kernel=recurrent_gdn_use_fla_kernel,
        recurrent_gdn_use_causal_conv1d=recurrent_gdn_use_causal_conv1d,
        recurrent_gdn_conv_kernel_size=2,
        recurrent_gdn_key_head_dim=16,
        recurrent_gdn_value_head_dim=16,
        recurrent_gdn_num_key_heads=4,
        recurrent_gdn_num_value_heads=4,
    )

    model = ARMTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=32000,
        max_sequence_length=seq_len,
        num_mem_tokens=4,
        recurrent_chunk_size=recurrent_chunk_size,
        full_attn_window_size=full_attn_window_size,
        pre_process=True,
        post_process=True,
        parallel_output=True,
    )
    return model


def _run_single_step(
    tp_size: int,
    seq_len: int,
    *,
    recurrent_chunk_size: int = 64,
    full_attn_window_size: int | None = None,
    skip_read_memory_from_first_chunk: bool = True,
    recurrent_memory_backend: str = "associative",
    recurrent_gdn_use_fla_kernel: bool = True,
    recurrent_gdn_use_causal_conv1d: bool = True,
):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)

    model = _build_model(
        tp_size,
        seq_len,
        recurrent_chunk_size=recurrent_chunk_size,
        full_attn_window_size=full_attn_window_size,
        recurrent_memory_backend=recurrent_memory_backend,
        recurrent_gdn_use_fla_kernel=recurrent_gdn_use_fla_kernel,
        recurrent_gdn_use_causal_conv1d=recurrent_gdn_use_causal_conv1d,
    ).cuda()
    model.train()
    model.set_current_chunk_is_first(True)
    model.set_skip_read_memory_for_current_chunk(skip_read_memory_from_first_chunk)

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


def _run_chunked_train_steps(
    *,
    seq_len: int = 64,
    recurrent_chunk_size: int,
    full_attn_window_size: int | None,
    steps: int = 10,
):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)

    model = _build_model(
        1,
        seq_len,
        recurrent_chunk_size=recurrent_chunk_size,
        full_attn_window_size=full_attn_window_size,
        recurrent_tbptt_mode=False,
        recurrent_memory_backend="associative",
    ).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 2
    generator = torch.Generator(device="cuda")
    generator.manual_seed(4321)
    batches = [
        torch.randint(0, 32000, (batch_size, seq_len), generator=generator, device="cuda")
        for _ in range(steps)
    ]
    losses = []

    for tokens in batches:
        labels = tokens.clone()
        loss_mask = torch.ones(batch_size, seq_len, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        model.reset_all_memory()
        total_loss = None
        total_weighted_loss = None
        total_tokens = 0

        for start in range(0, seq_len, recurrent_chunk_size):
            end = min(start + recurrent_chunk_size, seq_len)
            chunk_tokens = tokens[:, start:end]
            chunk_labels = labels[:, start:end]
            chunk_loss_mask = loss_mask[:, start:end]
            chunk_num_tokens = int(chunk_loss_mask.sum().item())
            total_tokens += chunk_num_tokens
            position_ids = torch.arange(start, end, device="cuda").unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.triu(
                torch.ones(1, 1, end - start, end - start, device="cuda", dtype=torch.bool),
                diagonal=1,
            )

            model.set_current_chunk_is_first(start == 0)
            model.set_skip_read_memory_for_current_chunk(start == 0)
            model.set_current_chunk_start_position(start)

            chunk_loss = model(
                chunk_tokens,
                position_ids,
                attention_mask,
                labels=chunk_labels,
                loss_mask=chunk_loss_mask,
            )
            if chunk_loss.dim() != 0:
                chunk_loss = chunk_loss.mean()
            total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
            weighted_chunk_loss = chunk_loss.detach().float() * chunk_num_tokens
            total_weighted_loss = (
                weighted_chunk_loss
                if total_weighted_loss is None
                else total_weighted_loss + weighted_chunk_loss
            )

        assert total_loss is not None
        assert total_weighted_loss is not None
        total_loss.backward()
        optimizer.step()
        losses.append((total_weighted_loss / total_tokens).cpu())

    destroy_model_parallel()
    return losses


def _finalize_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class TestARMTTraining:
    @requires_gpu
    @pytest.mark.parametrize(
        ("recurrent_memory_backend", "recurrent_gdn_use_fla_kernel", "recurrent_gdn_use_causal_conv1d"),
        [
            ("associative", True, True),
            pytest.param(
                "gated_deltanet",
                True,
                True,
                marks=pytest.mark.skipif(
                    not (HAS_FLA and HAS_CAUSAL_CONV1D),
                    reason="Need FLA and causal_conv1d for fused GDN path",
                ),
            ),
            ("gated_deltanet", False, False),
            pytest.param(
                "gated_deltanet",
                True,
                False,
                marks=pytest.mark.skipif(not HAS_FLA, reason="Need FLA for fused GDN path"),
            ),
        ],
    )
    def test_armt_single_step(
        self,
        recurrent_memory_backend,
        recurrent_gdn_use_fla_kernel,
        recurrent_gdn_use_causal_conv1d,
    ):
        """集成测试：单卡（TP=1）下完成一次真实 forward/backward，loss 为有限值。"""
        _init_distributed(world_size=1)
        try:
            _run_single_step(
                tp_size=1,
                seq_len=64,
                skip_read_memory_from_first_chunk=True,
                recurrent_memory_backend=recurrent_memory_backend,
                recurrent_gdn_use_fla_kernel=recurrent_gdn_use_fla_kernel,
                recurrent_gdn_use_causal_conv1d=recurrent_gdn_use_causal_conv1d,
            )
        finally:
            _finalize_distributed()

    @requires_gpu
    def test_armt_single_step_with_first_chunk_read(self):
        _init_distributed(world_size=1)
        try:
            _run_single_step(tp_size=1, seq_len=64, skip_read_memory_from_first_chunk=False)
        finally:
            _finalize_distributed()

    @requires_multi_gpu
    def test_armt_tp(self):
        """集成测试：TP=2 下完成一次真实 forward/backward，验证 TP 环境可跑通。"""
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 2:
            pytest.skip("TP=2 test requires launching pytest with WORLD_SIZE>=2, e.g. under torchrun.")
        _init_distributed(world_size=world_size)
        try:
            _run_single_step(tp_size=2, seq_len=64, skip_read_memory_from_first_chunk=True)
        finally:
            _finalize_distributed()

    @requires_gpu
    def test_armt_windowed_full_attention_10_step_losses_are_finite(self):
        _init_distributed(world_size=1)
        try:
            losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=32,
                steps=10,
            )
        finally:
            _finalize_distributed()

        assert len(losses) == 10
        assert all(torch.isfinite(loss).item() for loss in losses)
        expected_random_ce = math.log(32000)
        assert all(5.0 < float(loss) < expected_random_ce + 5.0 for loss in losses)

    @requires_gpu
    def test_armt_equal_window_matches_legacy_losses_over_10_steps(self):
        _init_distributed(world_size=1)
        try:
            legacy_losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=None,
                steps=10,
            )
            equal_window_losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=16,
                steps=10,
            )
        finally:
            _finalize_distributed()

        assert len(legacy_losses) == len(equal_window_losses) == 10
        for legacy_loss, equal_window_loss in zip(legacy_losses, equal_window_losses):
            assert torch.allclose(legacy_loss, equal_window_loss, atol=1e-5, rtol=1e-4)
