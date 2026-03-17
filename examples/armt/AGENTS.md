# ARMT Entrypoint Module (`examples/armt`)

## Core Function
- 提供 ARMT 训练入口 `train.py`，将 ARMT 模型/调度接入标准 Megatron GPT 训练流程。
- 定义 ARMT 参数与 v1 约束：`armt_args.py`。
- 提供参考配置/脚本、checkpoint 转换工具与测试用例。

## Directory Structure
- `train.py`: `torchrun/python` 入口；调用 `megatron.training.pretrain(...)`。
- `armt_args.py`: `add_armt_args()` + `validate_armt_constraints()`。
- `tools/`: checkpoint 相关工具（baseline -> ARMT）。
- `tests/`: GPU 集成冒烟 + checkpoint 转换测试。
- `configs/`, `scripts/`: 最小可运行样例（不被 `train.py` 自动加载）。
- `README.md`: v1 约束与用法。

## Key Data Flow (Launcher -> Train -> Core)
1. 启动：`torchrun ... examples/armt/train.py --use-armt-tbptt ...`
2. 参数：
   - `pretrain(..., extra_args_provider=add_armt_args)` 将 ARMT 参数注入全局 args。
   - `model_provider()` 里 `validate_armt_constraints(args)` 做硬约束检查。
3. 模型：
   - `get_armt_layer_spec(...)` 产出 ARMT layer spec
   - `ARMTModel(...)` 构建带 memory tokens 的 GPT 变体
4. 调度（TBPTT）：
   - 实现：`megatron/core/pipeline_parallel/recurrent_schedules.py:recurrent_forward_backward_no_pipelining`
   - `schedules.py` 会在 `use_recurrent_tbptt` 打开时自动选用。

## Batch Contract (forward_step)
- 关键字段：`tokens/labels/loss_mask/attention_mask/position_ids`
- v1 固定：`packed_seq_params` 必须为 `None`（packed sequence 不支持）。
- `forward_step` 期望输入 batch 已在 GPU；推荐由 TBPTT schedule 内部调用 `get_batch_on_this_tp_rank()` 负责 `.cuda()` 与 TP broadcast。
- `--armt-chunk-size`：按序列维切 TBPTT chunks；`--no-loss-from-first-chunk` 会将首 chunk 的 `loss_mask` 置 0。

## Dev Notes
- v1 约束（与单测一致）：
  - `PP=1`、`CP=1`
  - `position_embedding_type in {rope,yarn}`
  - 禁用 FP8；禁用 packed sequence
  - 开启 `sequence_parallel` 时需要满足整除约束（见 `armt_args.py`）。
- 从 baseline ckpt 启动 ARMT：
  - baseline 没有 ARMT 新参数；分布式 ckpt 推荐 `--dist-ckpt-strictness log_*` 以丢弃 unexpected keys。
  - 或用 `tools/convert_baseline_to_armt.py` 预先补齐 ARMT 参数（非 torch_dist 分布式 ckpt）。
- 修改 ARMT 语义后同步更新：
  - `tests/unit_tests/models/armt/`（CPU 单测）
  - `examples/armt/tests/`（GPU 冒烟/转换测试）
