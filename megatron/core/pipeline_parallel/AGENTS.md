# ARMT TBPTT Schedule Module (`megatron/core/pipeline_parallel`)

## Core Function
- 在不启用 pipeline parallel（PP=1）的前提下，为 ARMT 提供 TBPTT 的 forward/backward 调度实现。

## Directory Structure (ARMT-related)
- `schedules.py`: 默认 forward/backward schedules（当前不自动切到 ARMT TBPTT）。
- `armt_schedules.py`:
  - `chunk_data(...)`: 按 `armt_chunk_size` 切分 batch（含 `attention_mask` 的方阵切片）
  - `armt_forward_backward_no_pipelining(...)`: TBPTT 调度主函数

## Key Data Flow
1. `armt_forward_backward_no_pipelining(...)`（作为 `forward_backward_func` 运行）：
   - 从 `data_iterator` 获取 raw batch（TP rank 广播兼容：`get_batch_on_this_tp_rank`）
   - 强制 `packed_seq_params=None`（v1 不支持 packed seq）
   - `chunk_data(...)` 切 chunk；可选 `--no-loss-from-first-chunk` 将首 chunk `loss_mask` 置 0
   - 每个 microbatch 开始：若模型支持 `reset_all_memory()` 则重置状态
   - 每个 chunk：`forward_step_func()` -> `loss_func()` -> 按 token 比例缩放 loss -> 需要时 `backward_step()`
   - 使用 `no_sync` 避免非最后一步的梯度 all-reduce；最后调用 `finalize_model_grads_func`

## Dev Notes
- 入口约束：PP 必须为 1（`examples/armt/armt_args.py` 校验；TBPTT 仅实现 no-pipeline 路径）。
- batch 约束：
  - `--no-loss-from-first-chunk` 需要 `batch["loss_mask"]` 存在且为 tensor，否则显式报错。
  - packed sequences 直接 `NotImplementedError`（保持 fail-fast）。
- 修改 chunk 语义/切片规则/skip-backward 逻辑后，更新单测：
  - `tests/unit_tests/models/armt/test_armt_scheduler.py`
