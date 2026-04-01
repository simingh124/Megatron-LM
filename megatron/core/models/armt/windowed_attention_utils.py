"""Utilities for deciding when ARMT should use the windowed full-attention path."""

from typing import Optional


def should_use_windowed_full_attention(
    *,
    recurrent_chunk_size: Optional[int],
    full_attn_window_size: Optional[int],
    equal_window_full_attn_path: str = "legacy",
) -> bool:
    if recurrent_chunk_size is None or full_attn_window_size is None:
        return False
    if full_attn_window_size > recurrent_chunk_size:
        return True
    if full_attn_window_size < recurrent_chunk_size:
        return False
    if equal_window_full_attn_path == "window":
        return True
    return False
