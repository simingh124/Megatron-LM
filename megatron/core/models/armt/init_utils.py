"""Megatron-aligned initialization helpers for ARMT-specific modules."""

from typing import Callable

import torch
import torch.nn as nn


def init_parameter(
    parameter: nn.Parameter,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    *,
    perform_initialization: bool,
) -> None:
    if not perform_initialization:
        return
    with torch.no_grad():
        init_method(parameter)


def init_linear_weight_and_bias(
    linear: nn.Linear,
    weight_init_method: Callable[[torch.Tensor], torch.Tensor],
    *,
    perform_initialization: bool,
) -> None:
    if not perform_initialization:
        return
    with torch.no_grad():
        weight_init_method(linear.weight)
        if linear.bias is not None:
            linear.bias.zero_()
