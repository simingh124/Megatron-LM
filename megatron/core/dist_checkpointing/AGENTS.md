# Distributed Checkpointing (`megatron/core/dist_checkpointing`)

## Core Function
- `torch_dist` 分布式 checkpoint：按 TP/PP/DP 分片保存/加载模型与优化器状态；提供策略与 strictness 语义。

## Directory Structure
- `core.py`: `save()/load()` 入口
- `mapping.py`: `ShardedTensor/ShardedObject` 描述与 offsets
- `serialization.py`: 读写策略选择（default load/save strategy）
- `strategies/`: fully-parallel 等实现
- `tensor_aware_state_dict.py`: tensor-aware state dict（加速/一致性）

## Key Data Flow (ARMT run)
1. `training/checkpointing.py`：
   - `args.ckpt_format == "torch_dist"` -> `dist_checkpointing.save/load`
   - `strict=args.dist_ckpt_strictness`
2. baseline -> ARMT：checkpoint metadata 不含 ARMT 新参数
   - `log_unexpected/log_all`：加载时丢弃 unexpected keys（但仍记录）
3. 保存时写 `common_state_dict`（iteration/args 等）+ sharded model/optim state

## Dev Notes
- 修改参数命名/shape/sharding 后检查 strictness 是否仍可兼容旧 ckpt。
- “新增参数”场景建议先用 `log_*` strictness 验证，再考虑 conversion/tool 补齐。
