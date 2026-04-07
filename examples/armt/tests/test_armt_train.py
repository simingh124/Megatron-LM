import math
import os
import random
import re
import subprocess
from importlib.util import find_spec
from pathlib import Path

import torch
import torch.distributed as dist
import pytest

from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import Float16Module
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

requires_8_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="Need at least 8 GPUs",
)

HAS_FLA = find_spec("fla") is not None
HAS_CAUSAL_CONV1D = find_spec("causal_conv1d") is not None
HAS_FLASH_ATTN = find_spec("flash_attn") is not None

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = (
    REPO_ROOT / "playground" / "rmt" / "qwen3_0p6b_armt_cross_attn_0324_fs_wo_tbptt.sh"
)
_ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+\s+\|.*?"
    r"elapsed time per iteration \(ms\):\s*([0-9.+\-Ee]+)\s*\|.*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.+\-Ee]+)\s*\|.*?"
    r"lm loss:\s*([0-9.+\-Ee]+)\s*\|.*?"
    r"number of skipped iterations:\s*(\d+)\s*\|.*?"
    r"number of nan iterations:\s*(\d+)\s*\|"
)


def _unwrap_model(model):
    return model.module if isinstance(model, Float16Module) else model


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
    armt_windowed_full_attn_backend: str = "native",
    armt_equal_window_full_attn_path: str = "legacy",
    params_dtype: torch.dtype = torch.float32,
    recurrent_slot_num_slots: int = 8,
    recurrent_slot_num_heads: int = 4,
    recurrent_slot_head_dim: int = 16,
    recurrent_slot_read_attn_backend: str = "sdpa",
):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=256,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=params_dtype,
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
        armt_windowed_full_attn_backend=armt_windowed_full_attn_backend,
        armt_equal_window_full_attn_path=armt_equal_window_full_attn_path,
        recurrent_gdn_conv_kernel_size=2,
        recurrent_gdn_key_head_dim=16,
        recurrent_gdn_value_head_dim=16,
        recurrent_gdn_num_key_heads=4,
        recurrent_gdn_num_value_heads=4,
        recurrent_slot_num_slots=recurrent_slot_num_slots,
        recurrent_slot_num_heads=recurrent_slot_num_heads,
        recurrent_slot_head_dim=recurrent_slot_head_dim,
        recurrent_slot_read_attn_backend=recurrent_slot_read_attn_backend,
    )

    model = ARMTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=32000,
        max_sequence_length=seq_len,
        num_mem_tokens=4,
        recurrent_chunk_size=recurrent_chunk_size,
        full_attn_window_size=full_attn_window_size,
        armt_equal_window_full_attn_path=armt_equal_window_full_attn_path,
        pre_process=True,
        post_process=True,
        parallel_output=True,
    )
    if params_dtype == torch.float16:
        config.fp16 = True
        config.bf16 = False
        model = Float16Module(config, model)
    elif params_dtype == torch.bfloat16:
        config.fp16 = False
        config.bf16 = True
        model = Float16Module(config, model)
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
    recurrent_slot_num_slots: int = 8,
    recurrent_slot_num_heads: int = 4,
    recurrent_slot_head_dim: int = 16,
    recurrent_slot_read_attn_backend: str = "sdpa",
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
        recurrent_slot_num_slots=recurrent_slot_num_slots,
        recurrent_slot_num_heads=recurrent_slot_num_heads,
        recurrent_slot_head_dim=recurrent_slot_head_dim,
        recurrent_slot_read_attn_backend=recurrent_slot_read_attn_backend,
    ).cuda()
    model.train()
    runtime_model = _unwrap_model(model)
    runtime_model.set_current_chunk_is_first(True)
    runtime_model.set_skip_read_memory_for_current_chunk(skip_read_memory_from_first_chunk)

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
    armt_windowed_full_attn_backend: str = "native",
    armt_equal_window_full_attn_path: str = "legacy",
    params_dtype: torch.dtype = torch.float32,
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
        armt_windowed_full_attn_backend=armt_windowed_full_attn_backend,
        armt_equal_window_full_attn_path=armt_equal_window_full_attn_path,
        params_dtype=params_dtype,
    ).cuda()
    model.train()
    runtime_model = _unwrap_model(model)
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
        runtime_model.reset_all_memory()
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

            runtime_model.set_current_chunk_is_first(start == 0)
            runtime_model.set_skip_read_memory_for_current_chunk(start == 0)
            runtime_model.set_current_chunk_start_position(start)

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


def _parse_iteration_metrics(output: str):
    metrics = []
    for match in _ITERATION_RE.finditer(output):
        metrics.append(
            {
                "iteration": int(match.group(1)),
                "elapsed_ms": float(match.group(2)),
                "throughput": float(match.group(3)),
                "loss": float(match.group(4)),
                "skipped": int(match.group(5)),
                "nan": int(match.group(6)),
            }
        )
    return metrics


def _run_cross_attn_launcher(read_attn_backend: str, exit_interval: int):
    artifact_root = (
        REPO_ROOT / "codex_assets" / "armt_cross_attn_flash_compare" / read_attn_backend
    )
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_TEST_TRAIN_RUN": "1",
            "EXIT_INTERVAL": str(exit_interval),
            "ENABLE_TEE_LOG": "0",
            "MASTER_PORT": str(19000 + random.randint(1, 2000)),
            "CHECKPOINT_PATH": str(artifact_root / "checkpoints"),
            "TENSORBOARD_LOGS_PATH": str(artifact_root / "tensorboard"),
            "LOG_DIR": str(artifact_root / "logs"),
            "RECURRENT_SLOT_READ_ATTN_BACKEND": read_attn_backend,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _assert_launcher_health(output: str, returncode: int, expected_steps: int, backend: str):
    assert returncode == 0, f"{backend} launcher failed:\n{output[-12000:]}"

    metrics = _parse_iteration_metrics(output)
    assert len(metrics) == expected_steps, (
        f"{backend} launcher expected {expected_steps} iteration logs, got {len(metrics)}:\n"
        f"{output[-12000:]}"
    )

    for expected_iteration, metric in enumerate(metrics, start=1):
        assert metric["iteration"] == expected_iteration, metric
        assert torch.isfinite(torch.tensor(metric["loss"])), metric
        assert torch.isfinite(torch.tensor(metric["throughput"])), metric
        assert torch.isfinite(torch.tensor(metric["elapsed_ms"])), metric
        assert metric["loss"] > 0.0, metric
        assert metric["throughput"] > 0.0, metric
        assert metric["elapsed_ms"] > 0.0, metric
        assert metric["skipped"] == 0, metric
        assert metric["nan"] == 0, metric

    return metrics


def _compute_loss_relative_error(flash_metric, sdpa_metric):
    return abs(flash_metric["loss"] - sdpa_metric["loss"]) / max(
        abs(sdpa_metric["loss"]),
        1e-12,
    )


def _format_loss_comparison_summary(flash_metrics, sdpa_metrics):
    if len(flash_metrics) != len(sdpa_metrics):
        raise ValueError(
            f"flash/sdpa metric length mismatch: {len(flash_metrics)} != {len(sdpa_metrics)}"
        )

    lines = [
        "flash vs sdpa loss comparison",
        "iteration | flash_loss | sdpa_loss | relative_error",
    ]
    for flash_metric, sdpa_metric in zip(flash_metrics, sdpa_metrics):
        relative_error = _compute_loss_relative_error(flash_metric, sdpa_metric)
        lines.append(
            f'{flash_metric["iteration"]} | '
            f'{flash_metric["loss"]:.6f} | '
            f'{sdpa_metric["loss"]:.6f} | '
            f"{relative_error:.10f}"
        )

    return "\n".join(lines)


def test_format_loss_comparison_summary_includes_losses_and_relative_error():
    flash_metrics = [
        {"iteration": 1, "loss": 2.0, "elapsed_ms": 100.0, "throughput": 10.0, "skipped": 0, "nan": 0},
        {"iteration": 2, "loss": 1.0, "elapsed_ms": 120.0, "throughput": 12.0, "skipped": 0, "nan": 0},
    ]
    sdpa_metrics = [
        {"iteration": 1, "loss": 2.002, "elapsed_ms": 110.0, "throughput": 11.0, "skipped": 0, "nan": 0},
        {"iteration": 2, "loss": 0.999, "elapsed_ms": 118.0, "throughput": 12.5, "skipped": 0, "nan": 0},
    ]

    summary = _format_loss_comparison_summary(flash_metrics, sdpa_metrics)

    assert "iteration | flash_loss | sdpa_loss | relative_error" in summary
    assert "1 | 2.000000 | 2.002000 | 0.0009990010" in summary
    assert "2 | 1.000000 | 0.999000 | 0.0010010010" in summary


class TestARMTTraining:
    @requires_gpu
    @pytest.mark.parametrize(
        ("recurrent_memory_backend", "recurrent_gdn_use_fla_kernel", "recurrent_gdn_use_causal_conv1d"),
        [
            ("associative", True, True),
            ("cross_attn_slots", True, True),
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

    @requires_gpu
    def test_armt_equal_window_forced_window_path_10_step_losses_are_finite(self):
        _init_distributed(world_size=1)
        try:
            losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=16,
                steps=10,
                armt_equal_window_full_attn_path="window",
            )
        finally:
            _finalize_distributed()

        assert len(losses) == 10
        assert all(torch.isfinite(loss).item() for loss in losses)

    @requires_gpu
    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="Need flash-attn for ARMT windowed backend test")
    def test_armt_windowed_flash_attn_backend_matches_native_losses_over_10_steps(self):
        params_dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        )
        _init_distributed(world_size=1)
        try:
            native_losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=32,
                steps=10,
                armt_windowed_full_attn_backend="native",
                params_dtype=params_dtype,
            )
            flash_attn_losses = _run_chunked_train_steps(
                recurrent_chunk_size=16,
                full_attn_window_size=32,
                steps=10,
                armt_windowed_full_attn_backend="flash_attn",
                params_dtype=params_dtype,
            )
        finally:
            _finalize_distributed()

        assert len(native_losses) == len(flash_attn_losses) == 10
        for native_loss, flash_attn_loss in zip(native_losses, flash_attn_losses):
            assert torch.allclose(native_loss, flash_attn_loss, atol=1e-3, rtol=1.5e-3)

    @requires_8_gpu
    def test_cross_attn_launcher_flash_matches_sdpa_loss_for_10_steps(self):
        flash_result = _run_cross_attn_launcher(read_attn_backend="flash", exit_interval=10)
        flash_metrics = _assert_launcher_health(
            flash_result.stdout,
            flash_result.returncode,
            expected_steps=10,
            backend="flash",
        )

        sdpa_result = _run_cross_attn_launcher(read_attn_backend="sdpa", exit_interval=10)
        sdpa_metrics = _assert_launcher_health(
            sdpa_result.stdout,
            sdpa_result.returncode,
            expected_steps=10,
            backend="sdpa",
        )

        comparison_summary = _format_loss_comparison_summary(flash_metrics, sdpa_metrics)
        print(comparison_summary, flush=True)

        for flash_metric, sdpa_metric in zip(flash_metrics, sdpa_metrics):
            relative_error = _compute_loss_relative_error(flash_metric, sdpa_metric)
            assert relative_error <= 1e-3, {
                "flash_loss": flash_metric["loss"],
                "sdpa_loss": sdpa_metric["loss"],
                "relative_error": relative_error,
                "iteration": flash_metric["iteration"],
                "comparison_summary": comparison_summary,
            }
