# ARMT Launcher Module (`playground/rmt`)

## Core Function
- 提供 ARMT 训练启动脚本模板，组装分布式参数、模型参数、ARMT 参数、数据与 checkpoint 参数。
- 统一入口命令：`torchrun ... examples/armt/train.py ...`。

## Directory Structure
- `qwen3_0p6b_armt_*.sh`: ARMT 训练脚本变体（`TP`、`armt_d_mem`、`armt_n_heads`、`num_mem_tokens` 等）。
- `qwen3_0p6b_baseline_*.sh`: baseline 对照脚本。

## Related Modules (Downstream)
- `examples/armt/`: ARMT 训练入口、参数/约束、工具与测试。
- `megatron/core/models/armt/`: `ARMTModel/ARMTLayer/AssociativeLayer` 实现。
- `megatron/core/pipeline_parallel/`: `recurrent_schedules.py`（TBPTT 调度实现）。
- `tests/unit_tests/models/armt/`: CPU 单测（约束/模型/层/调度）。
- `examples/armt/tests/`: GPU 冒烟集成测试 + checkpoint 转换测试。

## Key Data Flow
1. 读取环境变量，确定分布式拓扑（`GPUS_PER_NODE/NUM_NODES/NODE_RANK`）。
2. 拼装参数组：`DISTRIBUTED_ARGS`、`MODEL_ARGS`、`ARMT_ARGS`、`TRAINING_ARGS`、`DATA_ARGS`、`CKPT_AND_LOG_ARGS`。
3. 将 `DATASET_PATH`、`TOKENIZER_DIR`、`LOAD_CHECKPOINT_PATH` 传入 `examples/armt/train.py`。
4. 由 `train.py` 进入 Megatron `pretrain()`；TBPTT 会经由 `schedules.py` 选择 `recurrent_forward_backward_no_pipelining`。

## Dev Notes
- `MODEL_ARGS` 必须与 `--load` checkpoint 结构一致（层数、hidden、TP/PP 等）。
- ARMT TBPTT 依赖 `--armt-chunk-size`；`forward_step` 期望输入 batch 已在 GPU（推荐用 `recurrent_schedules.py` 调度）。
- 从 baseline checkpoint 启动 ARMT 时，建议保持 `--ckpt-format torch_dist` + `--dist-ckpt-strictness log_all/log_unexpected`。
- 冒烟调试优先用 `EXIT_INTERVAL=1`，减少无效长跑。
