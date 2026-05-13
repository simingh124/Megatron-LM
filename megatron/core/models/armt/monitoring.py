"""Monitoring helpers for ARMT TensorBoard metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import torch

_RATIO_EPSILON = 1e-8
_ARMT_TENSORBOARD_TRACKER: Dict[str, "MetricPrimitive"] = {}


@dataclass(frozen=True)
class MetricPrimitive:
    """A reduce-friendly metric representation."""

    kind: str
    numerator: torch.Tensor
    denominator: torch.Tensor


def _to_scalar_tensor(value: torch.Tensor | float | int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.tensor(float(value), dtype=torch.float32)

    if tensor.numel() != 1:
        raise ValueError(f"Expected a scalar tensor, got shape {tuple(tensor.shape)}")

    return tensor.reshape(()).to(dtype=torch.float32)


def build_mean_metric(sum_value: torch.Tensor, count: torch.Tensor) -> MetricPrimitive:
    return MetricPrimitive("mean", _to_scalar_tensor(sum_value), _to_scalar_tensor(count))


def build_ratio_metric(numerator: torch.Tensor, denominator: torch.Tensor) -> MetricPrimitive:
    return MetricPrimitive("ratio", _to_scalar_tensor(numerator), _to_scalar_tensor(denominator))


def build_ratio_of_means_metric(
    numerator_sum: torch.Tensor,
    numerator_count: torch.Tensor,
    denominator_sum: torch.Tensor,
    denominator_count: torch.Tensor,
) -> MetricPrimitive:
    return MetricPrimitive(
        "ratio_of_means",
        torch.stack(
            (
                _to_scalar_tensor(numerator_sum),
                _to_scalar_tensor(numerator_count),
            )
        ),
        torch.stack(
            (
                _to_scalar_tensor(denominator_sum),
                _to_scalar_tensor(denominator_count),
            )
        ),
    )


def build_rms_metric(square_sum: torch.Tensor, count: torch.Tensor) -> MetricPrimitive:
    return MetricPrimitive("rms", _to_scalar_tensor(square_sum), _to_scalar_tensor(count))


def build_std_metric(
    sum_value: torch.Tensor,
    square_sum: torch.Tensor,
    count: torch.Tensor,
) -> MetricPrimitive:
    return MetricPrimitive(
        "std",
        torch.stack((_to_scalar_tensor(sum_value), _to_scalar_tensor(square_sum))),
        _to_scalar_tensor(count),
    )


def merge_metric_primitives(
    destination: MutableMapping[str, MetricPrimitive],
    source: Mapping[str, MetricPrimitive],
) -> MutableMapping[str, MetricPrimitive]:
    for name, primitive in source.items():
        current = destination.get(name)
        if current is None:
            destination[name] = primitive
            continue
        if current.kind != primitive.kind:
            raise ValueError(
                f"Mismatched metric kinds for {name}: {current.kind} vs {primitive.kind}"
            )
        destination[name] = MetricPrimitive(
            kind=current.kind,
            numerator=current.numerator + primitive.numerator,
            denominator=current.denominator + primitive.denominator,
        )

    return destination


def finalize_metric_primitive(primitive: MetricPrimitive) -> torch.Tensor:
    numerator = primitive.numerator
    denominator = primitive.denominator

    if primitive.kind == "mean":
        return numerator / torch.clamp(denominator, min=1.0)
    if primitive.kind == "ratio":
        return numerator / (denominator + _RATIO_EPSILON)
    if primitive.kind == "ratio_of_means":
        if numerator.numel() != 2 or denominator.numel() != 2:
            raise ValueError(
                "ratio_of_means expects [sum, count] tensors for numerator and denominator"
            )
        numerator_mean = numerator.reshape(-1)[0] / torch.clamp(numerator.reshape(-1)[1], min=1.0)
        denominator_mean = denominator.reshape(-1)[0] / torch.clamp(
            denominator.reshape(-1)[1], min=1.0
        )
        return numerator_mean / (denominator_mean + _RATIO_EPSILON)
    if primitive.kind == "rms":
        return torch.sqrt(numerator / torch.clamp(denominator, min=1.0))
    if primitive.kind == "std":
        if numerator.numel() != 2:
            raise ValueError("std expects [sum, square_sum] tensors for numerator")
        count = torch.clamp(denominator, min=1.0)
        sum_value, square_sum = numerator.reshape(-1)
        mean = sum_value / count
        variance = square_sum / count - mean * mean
        return torch.sqrt(torch.clamp(variance, min=0.0))

    raise ValueError(f"Unsupported metric kind: {primitive.kind}")


def finalize_metric_primitives(
    primitives: Mapping[str, MetricPrimitive]
) -> Dict[str, torch.Tensor]:
    return {
        name: finalize_metric_primitive(primitive)
        for name, primitive in primitives.items()
    }


def clear_armt_tensorboard_metrics() -> None:
    _ARMT_TENSORBOARD_TRACKER.clear()


def accumulate_armt_tensorboard_metrics(primitives: Mapping[str, MetricPrimitive]) -> None:
    merge_metric_primitives(_ARMT_TENSORBOARD_TRACKER, primitives)


def publish_armt_tensorboard_metrics(primitives: Mapping[str, MetricPrimitive]) -> None:
    clear_armt_tensorboard_metrics()
    accumulate_armt_tensorboard_metrics(primitives)


def _get_metric_primitive_device(primitive: MetricPrimitive) -> torch.device:
    numerator_device = primitive.numerator.device
    denominator_device = primitive.denominator.device
    if numerator_device != denominator_device:
        raise ValueError(
            "ARMT metric primitives require numerator and denominator on the same device, "
            f"got {numerator_device} vs {denominator_device}."
        )
    return numerator_device


def _flatten_metric_primitives(
    primitives: Mapping[str, MetricPrimitive],
) -> Dict[torch.device, Tuple[torch.Tensor, list[tuple[str, str, tuple[int, ...], int, tuple[int, ...], int]]]]:
    grouped_buffers: Dict[torch.device, torch.Tensor] = {}
    grouped_metadata: Dict[
        torch.device,
        list[tuple[str, str, tuple[int, ...], int, tuple[int, ...], int]],
    ] = {}
    grouped_total_numel: Dict[torch.device, int] = {}

    for name, primitive in primitives.items():
        device = _get_metric_primitive_device(primitive)
        grouped_metadata.setdefault(device, [])
        grouped_total_numel[device] = grouped_total_numel.get(device, 0) + primitive.numerator.numel()
        grouped_total_numel[device] += primitive.denominator.numel()
        grouped_metadata[device].append(
            (
                name,
                primitive.kind,
                tuple(primitive.numerator.shape),
                primitive.numerator.numel(),
                tuple(primitive.denominator.shape),
                primitive.denominator.numel(),
            )
        )

    for device, total_numel in grouped_total_numel.items():
        grouped_buffers[device] = torch.empty(total_numel, device=device, dtype=torch.float32)

    grouped_offsets = {device: 0 for device in grouped_buffers}
    for device, metadata in grouped_metadata.items():
        flat_buffer = grouped_buffers[device]
        cursor = grouped_offsets[device]
        for name, _kind, _numerator_shape, numerator_numel, _denominator_shape, denominator_numel in metadata:
            primitive = primitives[name]
            flat_buffer[cursor : cursor + numerator_numel].copy_(
                primitive.numerator.to(dtype=torch.float32).reshape(-1)
            )
            cursor += numerator_numel
            flat_buffer[cursor : cursor + denominator_numel].copy_(
                primitive.denominator.to(dtype=torch.float32).reshape(-1)
            )
            cursor += denominator_numel
        grouped_offsets[device] = cursor

    return {
        device: (grouped_buffers[device], grouped_metadata[device]) for device in grouped_buffers
    }


def consume_armt_tensorboard_metrics(
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    finalize_on_this_rank: bool = True,
) -> Dict[str, torch.Tensor]:
    if not _ARMT_TENSORBOARD_TRACKER:
        return {}

    primitives = dict(_ARMT_TENSORBOARD_TRACKER)
    clear_armt_tensorboard_metrics()

    should_reduce = (
        reduce_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not finalize_on_this_rank:
        for flat_buffer, _metadata in _flatten_metric_primitives(primitives).values():
            if should_reduce:
                torch.distributed.all_reduce(flat_buffer, group=reduce_group)
        return {}

    finalized_metrics: Dict[str, torch.Tensor] = {}
    for flat_buffer, metadata in _flatten_metric_primitives(primitives).values():
        if should_reduce:
            torch.distributed.all_reduce(flat_buffer, group=reduce_group)

        cursor = 0
        for (
            name,
            kind,
            numerator_shape,
            numerator_numel,
            denominator_shape,
            denominator_numel,
        ) in metadata:
            numerator = flat_buffer[cursor : cursor + numerator_numel]
            cursor += numerator_numel
            denominator = flat_buffer[cursor : cursor + denominator_numel]
            cursor += denominator_numel
            finalized_metrics[name] = finalize_metric_primitive(
                MetricPrimitive(
                    kind=kind,
                    numerator=numerator.reshape(numerator_shape),
                    denominator=denominator.reshape(denominator_shape),
                )
            )

    return finalized_metrics
