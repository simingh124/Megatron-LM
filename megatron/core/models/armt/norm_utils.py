"""Shared normalization helpers for ARMT recurrent memory backends."""

from typing import Optional

import torch
import torch.nn as nn


def build_recurrent_norm(
    hidden_size: int,
    *,
    normalization: str,
    eps: float,
    dtype: Optional[torch.dtype],
) -> nn.Module:
    if normalization == "LayerNorm":
        return nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)
    if normalization == "RMSNorm":
        return nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
    raise ValueError(
        f"Unsupported recurrent memory normalization: {normalization}. "
        "Only LayerNorm and RMSNorm are supported."
    )
