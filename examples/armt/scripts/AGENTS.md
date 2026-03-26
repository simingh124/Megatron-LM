# ARMT Script Samples (`examples/armt/scripts`)

## Core Function
- 提供最小可运行的启动脚本样例（偏“用法演示/冒烟”，非生产模板）。

## Directory Structure
- `train_armt_3b.sh`: 直接调用 `python examples/armt/train.py ...` 的 CLI 示例。

## Key Data Flow
1. 组装 ARMT 关键 flags（`--use-recurrent-model-schedule/--num-mem-tokens/--armt-chunk-size`）
2. 透传到 `examples/armt/train.py`，进入 Megatron 训练入口。

## Dev Notes
- 脚本内容需要与 `examples/armt/README.md`、`examples/armt/armt_args.py` 保持一致。
- 复杂生产训练（多机、多日志、ckpt 严格性等）优先参考 `playground/rmt/` 下的模板脚本。
