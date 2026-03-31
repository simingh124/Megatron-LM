from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.models.armt.armt_self_attention import ARMTSelfAttention


def _build_minimal_windowed_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_mem_tokens: int,
    recurrent_chunk_size: int,
    full_attn_window_size: int,
):
    attention = ARMTSelfAttention.__new__(ARMTSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.num_mem_tokens = num_mem_tokens
    attention.recurrent_chunk_size = recurrent_chunk_size
    attention.full_attn_window_size = full_attn_window_size
    attention._use_windowed_full_attention = True
    attention._current_chunk_start_position = 0
    attention._history_key_cache = None
    attention._history_value_cache = None
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
