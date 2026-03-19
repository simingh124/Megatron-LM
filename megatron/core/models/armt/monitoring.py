"""Monitoring helpers for ARMT TensorBoard metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

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


def consume_armt_tensorboard_metrics(
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[str, torch.Tensor]:
    if not _ARMT_TENSORBOARD_TRACKER:
        return {}

    primitives = dict(_ARMT_TENSORBOARD_TRACKER)
    clear_armt_tensorboard_metrics()

    reduced_primitives: Dict[str, MetricPrimitive] = {}
    should_reduce = (
        reduce_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    for name, primitive in primitives.items():
        reduced_values = torch.stack(
            (
                primitive.numerator.to(dtype=torch.float32),
                primitive.denominator.to(dtype=torch.float32),
            )
        )
        if should_reduce:
            torch.distributed.all_reduce(reduced_values, group=reduce_group)
        reduced_primitives[name] = MetricPrimitive(
            kind=primitive.kind,
            numerator=reduced_values[0],
            denominator=reduced_values[1],
        )

    return finalize_metric_primitives(reduced_primitives)
