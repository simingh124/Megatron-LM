"""Helpers for selecting recurrent memory backends used by ARMT."""

from .associative_layer import AssociativeLayer
from .gated_deltanet_memory import GatedDeltaNetMemory

SUPPORTED_RECURRENT_MEMORY_BACKENDS = ("associative", "gated_deltanet")


def build_recurrent_memory_backend(backend: str, **kwargs):
    if backend == "associative":
        return AssociativeLayer(**kwargs)
    if backend == "gated_deltanet":
        return GatedDeltaNetMemory(**kwargs)
    raise ValueError(f"Unsupported recurrent memory backend: {backend}")
