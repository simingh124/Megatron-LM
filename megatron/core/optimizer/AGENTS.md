# Optimizers (`megatron/core/optimizer`)

## Core Function
- 构建/封装 optimizer（Adam 等）、分布式 optimizer、梯度裁剪与（fp16）grad scaling。

## Directory Structure
- `optimizer.py`: `get_megatron_optimizer(...)` 主入口
- `optimizer_config.py`: 配置结构（lr/weight_decay/betas/...）
- `distrib_optimizer.py`: `DistributedOptimizer`（param buffer + reduce-scatter）
- `clip_grads.py`: grad norm / clipping
- `grad_scaler.py`: fp16 loss scaling（bf16 通常不需要）

## Key Data Flow
1. `training.py` 根据 args 创建 optimizer + scheduler
2. schedule/backward 产 grads -> finalize -> `optimizer.step()`
3. checkpointing 保存/加载 optimizer state（本脚本常用 `--no-load-optim`）

## Dev Notes
- ARMT 新参数会自动进入 optimizer；若做 param group 特化需显式覆盖。
- TBPTT schedule 按 token 比例缩放 loss，避免 chunk 长度偏置；修改该逻辑需同步看 grad clip 行为。
