# Megatron Core Runtime (`megatron/core`)

## Core Function
- Megatron “核心运行时”：并行拓扑/进程组、模型基座（Transformer/GPT）、数据集与 checkpoint 的公共能力入口。
- 训练 ARMT 时这里负责：TP/DP 相关 primitive、mask/position 约定、schedule 所需的 unwrap/config 工具。

## Directory Structure (Training-critical)
- `parallel_state.py`: 初始化/查询 TP/PP/DP/CP 进程组与 rank 信息（训练/ckpt/collective 都依赖）。
- `process_groups_config.py`: `ProcessGroupCollection`（将多组 pg 作为一个集合向下传递）。
- `utils.py`: `unwrap_model/get_model_config/get_model_type/...`（schedule/ckpt/训练 loop 常用）。
- `num_microbatches_calculator.py`: microbatch 计算与 gradient accumulation 相关逻辑。
- 关键子目录（已分散建档）：
  - `models/`（`gpt/`, `armt/`, `common/`）
  - `transformer/`, `tensor_parallel/`, `distributed/`, `optimizer/`
  - `pipeline_parallel/`（含 ARMT TBPTT schedule）
  - `datasets/`, `tokenizers/`, `dist_checkpointing/`

## Key Data Flow (ARMT Training)
1. `megatron/training/initialize.py` 初始化 torch.distributed + `parallel_state.initialize_model_parallel(...)`
2. `examples/armt/train.py` 构建 `ARMTModel`（依赖 `core/models/*` + `core/transformer/*`）
3. 训练 loop 调 `forward_backward_func`：
   - 默认来自 `pipeline_parallel/schedules.py`
   - ARMT/RMT TBPTT 由 `pipeline_parallel/recurrent_schedules.py` 提供 no-pipeline 实现
4. `training/checkpointing.py` 在 `--ckpt-format torch_dist` 下走 `core/dist_checkpointing`

## Dev Notes
- ARMT v1 约束（PP=1/CP=1、rope/yarn、禁 FP8/packed seq）由 `examples/armt/armt_args.py` 校验，但落点涉及 `parallel_state`/mask/tokenizer/ckpt 多处契约。
- 自定义 schedule（如 TBPTT）通常需要 `utils.unwrap_model()` 获取原始模型，并使用 `distributed.finalize_model_grads` 完成梯度收尾。
