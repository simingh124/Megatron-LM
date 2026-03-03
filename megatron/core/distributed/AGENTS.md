# Distributed Training Runtime (`megatron/core/distributed`)

## Core Function
- DDP/FSDP 包装、梯度 buffer、finalize grads（all-reduce / reduce-scatter），支撑 Megatron 的 DP 训练。

## Directory Structure
- `distributed_data_parallel.py`: `DistributedDataParallel`（Megatron DDP）
- `finalize_model_grads.py`: `finalize_model_grads_*`（用于 schedule 收尾）
- `param_and_grad_buffer.py`: 参数/梯度连续 buffer（性能关键）
- `fsdp/`, `torch_fully_sharded_data_parallel*.py`: Torch FSDP 支持

## Key Data Flow
1. model 构建后被 wrap 为 DDP（config 提供 `no_sync_func/finalize_model_grads_func`）
2. schedule 在累积期进入 `no_sync`，最后一次退出并调用 `finalize_model_grads_func`
3. optimizer step 基于已 finalize 的 grads 更新参数

## Dev Notes (Custom schedules)
- 自定义 schedule（如 ARMT TBPTT）必须：
  - 对非最后一步使用 `config.no_sync_func()`，避免提前 all-reduce
  - 在所有 backward 完成后调用 `config.finalize_model_grads_func(...)`
- `force_all_reduce` 用于保存/检查主梯度时覆盖 reduce-scatter 行为。
