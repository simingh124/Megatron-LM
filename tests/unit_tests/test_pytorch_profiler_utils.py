import gzip
import json

from megatron.training.pytorch_profiler_utils import (
    build_perfetto_trace_path,
    build_pytorch_profiler_trace_handler,
)


class _FakeProfiler:
    def __init__(self, trace_id: str, payload: dict):
        self._trace_id = trace_id
        self._payload = payload
        self.exported_paths = []

    def get_trace_id(self):
        return self._trace_id

    def export_chrome_trace(self, path: str):
        self.exported_paths.append(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._payload, handle)


def test_build_perfetto_trace_path_uses_human_step_numbers_and_rank():
    path = build_perfetto_trace_path(
        trace_dir="/tmp/traces",
        profile_step_start=2,
        profile_step_end=5,
        rank=0,
        trace_id="trace/id",
        use_gzip=True,
    )

    assert path.name == "perfetto_step3-5_rank0_trace-id.json.gz"


def test_perfetto_trace_handler_gzips_single_chrome_trace(tmp_path):
    handler = build_pytorch_profiler_trace_handler(
        trace_dir=str(tmp_path),
        trace_format="perfetto",
        use_gzip=True,
        profile_step_start=2,
        profile_step_end=5,
        rank=3,
    )
    payload = {
        "schemaVersion": 1,
        "traceEvents": [
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "main"}},
            {"name": "aten::mm", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2, "ts": 10, "dur": 5, "args": {"Input Dims": []}},
            {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "pid": 1, "tid": 2, "ts": 12, "dur": 3},
            {"name": "void kernel", "cat": "kernel", "ph": "X", "pid": 7, "tid": 9, "ts": 15, "dur": 4, "args": {"grid": [1, 1, 1]}},
            {"name": "ProfilerStep#2", "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 8, "dur": 10},
        ],
    }
    profiler = _FakeProfiler(trace_id="trace/id", payload=payload)

    handler(profiler)

    output_path = tmp_path / "perfetto_step3-5_rank3_trace-id.json.gz"
    assert output_path.exists()
    assert profiler.exported_paths == [str(tmp_path / ".perfetto_step3-5_rank3_trace-id.json.gz.raw.json")]
    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == {
            "schemaVersion": 1,
            "traceEvents": [
                {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "main"}},
                {"name": "aten::mm", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2, "ts": 10, "dur": 5},
                {"name": "void kernel", "cat": "kernel", "ph": "X", "pid": 7, "tid": 9, "ts": 15, "dur": 4},
            ],
        }
