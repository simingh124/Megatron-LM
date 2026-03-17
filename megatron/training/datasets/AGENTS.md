# Training Data Samplers & Wrappers (`megatron/training/datasets`)

## Core Function
- 为 core datasets 构建 PyTorch DataLoader + 分布式 sampler（按 DP rank 切分/打乱）。
- 提供可选的 FIM/SFT dataset 包装。

## Directory Structure
- `data_samplers.py`: `build_pretraining_data_loader()` + `MegatronPretrainingSampler/*RandomSampler`
- `fim_dataset.py`: `GPTFIMDataset*`
- `sft_dataset.py`: `SFTDataset`

## Key Data Flow
1. `training.py` 构建 iterators 时调用 `build_pretraining_data_loader(dataset, consumed_samples)`
2. DataLoader -> yield batch dict（`GPTDataset` 默认 keys：`tokens/labels/loss_mask/position_ids` + 可选 `attention_mask`）
3. schedule/forward_step 决定 batch broadcast & `.cuda()`（ARMT/RMT TBPTT 由 `core/pipeline_parallel/recurrent_schedules.py` 做）

## Dev Notes
- sampler 以 `micro_batch_size * DP` 为一个“全局 step”，每个 DP rank 取连续片段；`consumed_samples` 影响 resume。
- `--dataloader-type=external` 会直通 dataset（自带 sampler/collate）。
- FIM 与 `--legacy-tokenizer` 不兼容（见 `arguments.py` 约束）。
