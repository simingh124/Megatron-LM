# Megatron Training Runtime (`megatron/training`)

## Core Function
- 训练入口：`pretrain()`（build model/datasets/optimizer，运行 train/eval，保存 ckpt）。
- 初始化与全局状态：`initialize_megatron()` + `global_vars`（args/tokenizer/timers/writers）。

## Directory Structure (High-signal)
- `training.py`: `pretrain()`, `train_step()`，evaluate/checkpoint 主循环。
- `initialize.py`: `initialize_megatron()`（parse/validate args，init dist/mpu，seed）。
- `arguments.py`: CLI/YAML 参数定义与校验（ARMT 通过 `extra_args_provider` 注入）。
- `checkpointing.py`: `save_checkpoint()/load_checkpoint()`（内部调用 `core/dist_checkpointing`）。
- `global_vars.py`: `_GLOBAL_ARGS/_GLOBAL_TOKENIZER/...`（`get_args()/get_tokenizer()`）。

## Key Data Flow (ARMT)
1. `examples/armt/train.py` -> `pretrain(..., extra_args_provider=add_armt_args, model_provider=...)`
2. `initialize_megatron()` -> `set_global_variables(args)`（会构建 legacy tokenizer 并设置 `args.padded_vocab_size`）
3. `train_valid_test_datasets_provider()`（来自 `pretrain_gpt.py`）构建 datasets + dataloaders
4. `train_step()` 调 `forward_backward_func(...)`；ARMT TBPTT 需要用 `armt_schedules.armt_forward_backward_no_pipelining`

## Dev Notes
- ARMT 接入点：`extra_args_provider`（新增 flags）+ `model_provider`（构建 `ARMTModel`）+ 自定义 schedule（TBPTT）。
- `--ckpt-format torch_dist` 时 load/save 走 `core/dist_checkpointing`；baseline -> ARMT 新参数会触发 unexpected keys，需 `--dist-ckpt-strictness log_*`。
- batch 的 `.cuda()`/TP broadcast 由 schedule/forward_step 协议决定；ARMT TBPTT schedule 内部调用 `get_batch_on_this_tp_rank()`。
