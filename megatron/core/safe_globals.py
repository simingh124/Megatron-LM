# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from argparse import Namespace
from io import BytesIO
from pathlib import PosixPath
from signal import Signals
from types import SimpleNamespace

import torch
from numpy import dtype, ndarray
from numpy.core.multiarray import _reconstruct
try:
    # NumPy 2.x
    from numpy.dtypes import UInt32DType  # type: ignore
except Exception:
    # NumPy 1.x does not expose dtype classes under numpy.dtypes.
    # For safe globals registration, using the base dtype type is sufficient.
    UInt32DType = dtype  # type: ignore

from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig
from megatron.core.rerun_state_machine import RerunDiagnostic, RerunMode, RerunState
from megatron.core.transformer.enums import AttnBackend, CudaGraphScope

SAFE_GLOBALS = [
    SimpleNamespace,
    PosixPath,
    _reconstruct,
    ndarray,
    dtype,
    UInt32DType,
    Namespace,
    AttnBackend,
    CudaGraphScope,
    ModelType,
    OptimizerConfig,
    RerunDiagnostic,
    RerunMode,
    RerunState,
    BytesIO,
    Signals,
]


def register_safe_globals():
    """Register megatron-core safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])
