from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.armt.armt_self_attention import ARMTSelfAttention
from megatron.core.models.armt.monitoring import finalize_metric_primitives


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _build_minimal_windowed_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_mem_tokens: int,
    recurrent_chunk_size: int,
    full_attn_window_size: int,
    backend: str = "native",
):
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.num_mem_tokens = num_mem_tokens
    attention.num_read_mem_tokens = 0
    attention.recurrent_chunk_size = recurrent_chunk_size
    attention.full_attn_window_size = full_attn_window_size
    attention.armt_windowed_full_attn_backend = backend
    attention.armt_read_memory_mode = "none"
    attention._use_windowed_full_attention = True
    attention._current_chunk_start_position = 0
    attention._history_key_cache = None
    attention._history_value_cache = None
    attention._collect_monitoring_for_current_iteration = True
    attention._monitoring_stats = {}
    attention.num_attention_heads_per_partition = query.shape[2]
    attention.num_query_groups_per_partition = query.shape[2]
    attention.hidden_size_per_attention_head = query.shape[3]
    attention.layer_number = 1
    attention.core_attention = SimpleNamespace(softmax_offset=None)
    attention.pg_collection = SimpleNamespace(cp=None)
    attention.config = SimpleNamespace(
        attention_dropout=0.0,
        sequence_parallel=True,
        softmax_scale=None,
        apply_query_key_layer_scaling=False,
        attention_output_gate=False,
    )
    attention.get_query_key_value_tensors = lambda *args, **kwargs: (query, key, value)
    attention.linear_proj = lambda x: (x, None)
    attention.training = False
    return attention


def test_equal_window_defaults_to_legacy_te_path():
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.recurrent_chunk_size = 8
    attention.full_attn_window_size = 8
    attention.armt_equal_window_full_attn_path = "legacy"

    ARMTSelfAttention._configure_windowed_full_attention_mode(attention)

    assert attention._use_windowed_full_attention is False


def test_equal_window_can_be_forced_onto_window_path():
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.recurrent_chunk_size = 8
    attention.full_attn_window_size = 8
    attention.armt_equal_window_full_attn_path = "window"

    ARMTSelfAttention._configure_windowed_full_attention_mode(attention)

    assert attention._use_windowed_full_attention is True


def test_windowed_attention_uses_scaled_dot_product_attention():
    seq_with_mem = 3
    batch_size = 1
    num_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="native",
    )

    def _fake_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        del attn_mask, dropout_p, is_causal, scale
        assert query.shape == (batch_size, num_heads, seq_with_mem, head_dim)
        assert key.shape == (batch_size, num_heads, seq_with_mem, head_dim)
        return torch.zeros_like(query)

    with patch(
        "megatron.core.models.armt.armt_self_attention.F.scaled_dot_product_attention",
        side_effect=_fake_sdpa,
    ) as sdpa_mock:
        output, bias = attention.forward(
            hidden_states=torch.randn(seq_with_mem, batch_size, num_heads * head_dim),
            attention_mask=None,
        )

    sdpa_mock.assert_called_once()
    assert output.shape == (seq_with_mem, batch_size, num_heads * head_dim)
    assert bias is None


def test_windowed_attention_split_excludes_read_prefix_and_write_suffix_from_real_tokens():
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.num_mem_tokens = 1
    attention.num_read_mem_tokens = 2
    attention.armt_read_memory_mode = "shared"
    attention._skip_read_memory_for_current_chunk = False

    tensor = torch.arange(12, dtype=torch.float32).view(6, 1, 1, 2)

    read_part, real_part, write_part = attention._split_read_real_and_memory_tokens(tensor)

    assert torch.equal(read_part, tensor[:2])
    assert torch.equal(real_part, tensor[2:5])
    assert torch.equal(write_part, tensor[5:])


def test_windowed_attention_history_update_defers_concatenation():
    chunk_one = torch.randn(2, 1, 2, 4)
    chunk_two = torch.randn(2, 1, 2, 4)
    attention = _build_minimal_windowed_attention(
        query=chunk_one,
        key=chunk_one,
        value=chunk_one,
        num_mem_tokens=0,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="native",
    )

    attention._update_history_kv_cache(chunk_one, chunk_one)

    with patch("megatron.core.models.armt.armt_self_attention.torch.cat") as cat_mock:
        attention._update_history_kv_cache(chunk_two, chunk_two)

    cat_mock.assert_not_called()


def test_windowed_attention_history_selection_keeps_recent_real_tokens():
    chunk_one = torch.full((2, 1, 1, 1), 1.0)
    chunk_two = torch.full((2, 1, 1, 1), 2.0)
    chunk_three = torch.full((2, 1, 1, 1), 3.0)
    attention = _build_minimal_windowed_attention(
        query=chunk_one,
        key=chunk_one,
        value=chunk_one,
        num_mem_tokens=0,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="native",
    )

    attention._update_history_kv_cache(chunk_one, chunk_one)
    attention._update_history_kv_cache(chunk_two, chunk_two)
    attention._update_history_kv_cache(chunk_three, chunk_three)

    history_key, history_value = attention._select_history_kv(current_real_seq_len=0)

    expected = torch.cat([chunk_two, chunk_three], dim=0)
    assert torch.equal(history_key, expected)
    assert torch.equal(history_value, expected)


def test_windowed_attention_matches_reference_softmax_path():
    seq_with_mem = 3
    batch_size = 2
    num_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, dtype=torch.float32)
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, dtype=torch.float32)
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, dtype=torch.float32)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="native",
    )

    output = attention._compute_windowed_attention(query, key, value)

    scale = attention._get_softmax_scale()
    reference_scores = torch.einsum(
        "lbhd,sbhd->bhls",
        query,
        key,
    ) * scale
    reference_mask = attention._build_rectangular_causal_mask(
        seq_with_mem,
        seq_with_mem,
        query.device,
    )
    reference_scores = reference_scores.masked_fill(
        reference_mask.unsqueeze(0).unsqueeze(0),
        -10000.0,
    )
    reference_probs = torch.softmax(reference_scores, dim=-1)
    reference_output = torch.einsum(
        "bhls,sbhd->lbhd",
        reference_probs,
        value,
    ).reshape(seq_with_mem, batch_size, num_heads * head_dim)

    assert torch.allclose(output, reference_output, atol=1e-6, rtol=1e-5)


def test_read_prefix_attention_mass_metric_uses_context_queries_only():
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.num_mem_tokens = 1
    attention.num_read_mem_tokens = 2
    attention.armt_read_memory_mode = "shared"
    attention._skip_read_memory_for_current_chunk = False
    attention._use_windowed_full_attention = False
    attention._collect_monitoring_for_current_iteration = True
    attention._monitoring_stats = {}
    attention.num_attention_heads_per_partition = 1
    attention.num_query_groups_per_partition = 1
    attention.hidden_size_per_attention_head = 1
    attention.layer_number = 1
    attention.core_attention = SimpleNamespace(softmax_offset=None)
    attention.pg_collection = SimpleNamespace(cp=None)
    attention.config = SimpleNamespace(
        attention_dropout=0.0,
        sequence_parallel=False,
        softmax_scale=1.0,
        apply_query_key_layer_scaling=False,
        attention_output_gate=False,
    )

    query = torch.tensor([[[[0.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]]])
    key = torch.tensor([[[[2.0]]], [[[0.0]]], [[[-2.0]]], [[[-2.0]]], [[[-2.0]]]])
    value = torch.zeros_like(key)
    attention.get_query_key_value_tensors = lambda *args, **kwargs: (query, key, value)

    attention.track_read_prefix_attention_mass(
        hidden_states=torch.zeros(5, 1, 1),
        attention_mask=None,
        rotary_pos_emb=None,
    )

    metrics = finalize_metric_primitives(attention.consume_monitoring_primitives())
    expected = (
        torch.softmax(torch.tensor([2.0, 0.0, -2.0]), dim=0)[:2].sum()
        + torch.softmax(torch.tensor([2.0, 0.0, -2.0, -2.0]), dim=0)[:2].sum()
    ) / 2.0
    assert float(metrics["armt/read_prefix/attn_mass_from_context_mean"]) == pytest.approx(
        float(expected)
    )


def test_read_prefix_attention_mass_metric_offsets_read_slice_after_history():
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.num_mem_tokens = 1
    attention.num_read_mem_tokens = 2
    attention.armt_read_memory_mode = "shared"
    attention._skip_read_memory_for_current_chunk = False
    attention._use_windowed_full_attention = True
    attention._collect_monitoring_for_current_iteration = True
    attention._monitoring_stats = {}
    attention.num_attention_heads_per_partition = 1
    attention.num_query_groups_per_partition = 1
    attention.hidden_size_per_attention_head = 1
    attention.layer_number = 1
    attention.core_attention = SimpleNamespace(softmax_offset=None)
    attention.pg_collection = SimpleNamespace(cp=None)
    attention.config = SimpleNamespace(
        attention_dropout=0.0,
        sequence_parallel=False,
        softmax_scale=1.0,
        apply_query_key_layer_scaling=False,
        attention_output_gate=False,
    )

    query = torch.tensor([[[[0.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]]])
    key = torch.tensor([[[[2.0]]], [[[0.0]]], [[[-2.0]]], [[[-2.0]]], [[[-2.0]]]])
    value = torch.zeros_like(key)
    history_key = torch.tensor([[[[5.0]]]])
    history_value = torch.zeros_like(history_key)
    attention.get_query_key_value_tensors = lambda *args, **kwargs: (query, key, value)
    attention._select_history_kv = lambda current_real_seq_len: (history_key, history_value)

    attention.track_read_prefix_attention_mass(
        hidden_states=torch.zeros(5, 1, 1),
        attention_mask=None,
        rotary_pos_emb=None,
    )

    metrics = finalize_metric_primitives(attention.consume_monitoring_primitives())
    expected = (
        torch.softmax(torch.tensor([5.0, 2.0, 0.0, -2.0]), dim=0)[1:3].sum()
        + torch.softmax(torch.tensor([5.0, 2.0, 0.0, -2.0, -2.0]), dim=0)[1:3].sum()
    ) / 2.0
    assert float(metrics["armt/read_prefix/attn_mass_from_context_mean"]) == pytest.approx(
        float(expected)
    )


@requires_gpu
def test_windowed_attention_flash_backend_uses_flash_attn_func():
    seq_with_mem = 3
    batch_size = 1
    num_heads = 2
    head_dim = 4
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="flash_attn",
    )

    def _fake_flash_attn(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False):
        del dropout_p, softmax_scale
        assert causal is True
        assert query.shape == (batch_size, seq_with_mem, num_heads, head_dim)
        assert key.shape == (batch_size, seq_with_mem, num_heads, head_dim)
        return torch.zeros_like(query)

    with patch(
        "megatron.core.models.armt.armt_self_attention._get_flash_attn_func",
        return_value=_fake_flash_attn,
    ) as flash_mock:
        with patch(
            "megatron.core.models.armt.armt_self_attention.F.scaled_dot_product_attention"
        ) as sdpa_mock:
            output, bias = attention.forward(
                hidden_states=torch.randn(
                    seq_with_mem,
                    batch_size,
                    num_heads * head_dim,
                    device="cuda",
                ),
                attention_mask=None,
            )

    flash_mock.assert_called_once()
    sdpa_mock.assert_not_called()
    assert output.shape == (seq_with_mem, batch_size, num_heads * head_dim)
    assert bias is None


def test_windowed_attention_flash_backend_rejects_softmax_offset():
    seq_with_mem = 3
    batch_size = 1
    num_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="flash_attn",
    )
    attention.core_attention = SimpleNamespace(softmax_offset=torch.tensor(0.0))

    with pytest.raises(RuntimeError, match="softmax_offset"):
        attention._compute_windowed_attention(query, key, value)


def test_windowed_attention_flash_backend_rejects_cpu_tensors():
    seq_with_mem = 3
    batch_size = 1
    num_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="flash_attn",
    )

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        attention._compute_windowed_attention(query, key, value)


@requires_gpu
def test_windowed_attention_flash_backend_rejects_fp32_tensors():
    seq_with_mem = 3
    batch_size = 1
    num_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda")
    key = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda")
    value = torch.randn(seq_with_mem, batch_size, num_heads, head_dim, device="cuda")
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="flash_attn",
    )

    with pytest.raises(RuntimeError, match="requires fp16/bf16"):
        attention._compute_windowed_attention(query, key, value)


def test_windowed_attention_native_backend_repeats_kv_for_gqa():
    seq_with_mem = 3
    batch_size = 1
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 4
    query = torch.randn(seq_with_mem, batch_size, num_query_heads, head_dim)
    key = torch.randn(seq_with_mem, batch_size, num_kv_heads, head_dim)
    value = torch.randn(seq_with_mem, batch_size, num_kv_heads, head_dim)
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="native",
    )
    attention.num_attention_heads_per_partition = num_query_heads
    attention.num_query_groups_per_partition = num_kv_heads

    def _fake_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        del attn_mask, dropout_p, is_causal, scale
        assert query.shape == (batch_size, num_query_heads, seq_with_mem, head_dim)
        assert key.shape == (batch_size, num_query_heads, seq_with_mem, head_dim)
        assert value.shape == (batch_size, num_query_heads, seq_with_mem, head_dim)
        return torch.zeros_like(query)

    with patch(
        "megatron.core.models.armt.armt_self_attention.F.scaled_dot_product_attention",
        side_effect=_fake_sdpa,
    ):
        output = attention._compute_windowed_attention(query, key, value)

    assert output.shape == (seq_with_mem, batch_size, num_query_heads * head_dim)


@requires_gpu
def test_windowed_attention_flash_backend_matches_reference_with_gqa():
    seq_with_mem = 3
    batch_size = 2
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    query = torch.randn(
        seq_with_mem,
        batch_size,
        num_query_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    key = torch.randn(
        seq_with_mem,
        batch_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value = torch.randn(
        seq_with_mem,
        batch_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    attention = _build_minimal_windowed_attention(
        query=query,
        key=key,
        value=value,
        num_mem_tokens=1,
        recurrent_chunk_size=2,
        full_attn_window_size=4,
        backend="flash_attn",
    )
    attention.num_attention_heads_per_partition = num_query_heads
    attention.num_query_groups_per_partition = num_kv_heads
    attention.hidden_size_per_attention_head = head_dim

    flash_output = attention._compute_windowed_attention(query, key, value)

    scale = attention._get_softmax_scale()
    repeated_key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=2)
    repeated_value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=2)
    reference_mask = attention._build_rectangular_causal_mask(
        seq_with_mem,
        seq_with_mem,
        query.device,
    )
    reference_output = F.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        repeated_key.permute(1, 2, 0, 3),
        repeated_value.permute(1, 2, 0, 3),
        attn_mask=~reference_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).permute(2, 0, 1, 3).contiguous()
    reference_output = reference_output.view(
        seq_with_mem,
        batch_size,
        num_query_heads * head_dim,
    )

    assert torch.allclose(flash_output.float(), reference_output.float(), atol=2e-3, rtol=2e-3)
