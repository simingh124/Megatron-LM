from __future__ import annotations

import gzip
import json
from pathlib import Path
from re import sub
from typing import Any, Callable, Literal

import torch

PytorchProfilerTraceFormat = Literal["tensorboard", "perfetto"]
_PERFETTO_TRACE_EVENT_CATEGORIES = {
    "cpu_op",
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
}


def _sanitize_trace_id(trace_id: str | None) -> str:
    if trace_id is None:
        return "trace"

    sanitized = sub(r"[^0-9A-Za-z._-]+", "-", str(trace_id)).strip("-_.")
    return sanitized or "trace"


def build_perfetto_trace_path(
    trace_dir: str,
    profile_step_start: int,
    profile_step_end: int,
    rank: int,
    trace_id: str | None,
    use_gzip: bool,
) -> Path:
    display_step_start = profile_step_start + 1
    suffix = ".json.gz" if use_gzip else ".json"
    trace_name = (
        f"perfetto_step{display_step_start}-{profile_step_end}"
        f"_rank{rank}_{_sanitize_trace_id(trace_id)}{suffix}"
    )
    return Path(trace_dir) / trace_name


def build_pytorch_profiler_trace_handler(
    trace_dir: str,
    trace_format: PytorchProfilerTraceFormat,
    use_gzip: bool,
    profile_step_start: int,
    profile_step_end: int,
    rank: int,
) -> Callable[[Any], None]:
    if trace_format == "tensorboard":
        return torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=use_gzip)

    def _handle_trace(profiler: Any) -> None:
        trace_path = build_perfetto_trace_path(
            trace_dir=trace_dir,
            profile_step_start=profile_step_start,
            profile_step_end=profile_step_end,
            rank=rank,
            trace_id=profiler.get_trace_id() if hasattr(profiler, "get_trace_id") else None,
            use_gzip=use_gzip,
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        raw_trace_path = trace_path.parent / f".{trace_path.name}.raw.json"
        profiler.export_chrome_trace(str(raw_trace_path))

        filtered_trace = _load_trace_payload(raw_trace_path)
        filtered_trace = _filter_perfetto_trace_payload(filtered_trace)
        _write_trace_payload(trace_path, filtered_trace, use_gzip=use_gzip)
        raw_trace_path.unlink()

    return _handle_trace


def _load_trace_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_perfetto_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered_trace_events = []
    for event in payload.get("traceEvents", []):
        phase = event.get("ph")
        category = event.get("cat") or ""
        if phase == "M":
            filtered_trace_events.append(
                {key: event[key] for key in ("name", "ph", "pid", "tid", "ts", "args") if key in event}
            )
            continue
        if category not in _PERFETTO_TRACE_EVENT_CATEGORIES:
            continue
        filtered_trace_events.append(
            {key: event[key] for key in ("name", "cat", "ph", "pid", "tid", "ts", "dur") if key in event}
        )

    return {
        key: payload[key]
        for key in ("schemaVersion", "displayTimeUnit", "baseTimeNanoseconds")
        if key in payload
    } | {"traceEvents": filtered_trace_events}


def _write_trace_payload(path: Path, payload: dict[str, Any], use_gzip: bool) -> None:
    if use_gzip:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
