# Core Datasets (`megatron/core/datasets`)

## Core Function
- 提供可复用的 pretrain datasets（GPT/BERT/T5/...）与索引格式（mmap/indexed_dataset），支持 blend、cache、mask/position_id 生成。

## Directory Structure (GPT path)
- `gpt_dataset.py`: `GPTDataset`（输出 `tokens/labels/loss_mask/position_ids` + 可选 `attention_mask`）
- `blended_megatron_dataset_builder.py`: `BlendedMegatronDatasetBuilder.build()`（按 blend/split 构建 train/valid/test）
- `blended_megatron_dataset_config.py`: 配置结构（seq_len、tokenizer、mask flags、cache 等）
- `indexed_dataset.py`: mmap/indexed 二进制数据读取
- `megatron_tokenizer.py`: legacy tokenizer 抽象（供 training tokenizer 继承）

## Key Data Flow (ARMT run)
1. `pretrain_gpt.py:core_gpt_dataset_config_from_args(args)`：
   - tokenizer：默认走 `core/tokenizers`（`--legacy-tokenizer` 才用 `global_vars.get_tokenizer()`）
   - `--data-path/--split/...` -> `blend`/`blend_per_split`
2. `BlendedMegatronDatasetBuilder(...).build()` -> `GPTDataset.__getitem__` 产 batch（CPU）
3. `training/datasets.build_pretraining_data_loader()` 负责 sampler/DataLoader
4. TBPTT：`recurrent_schedules.chunk_data()` 依据 `seq_length` 切 `tokens/labels/loss_mask/position_ids`，并对 `attention_mask[...,s:e,s:e]` 做方阵切片

## Dev Notes
- `--create-attention-mask-in-dataloader` 决定是否返回 `attention_mask`；TBPTT 可无此字段（模型需能处理）。
- `loss_mask` 会在 labels==pad 时置 0；若使用 `--no-loss-from-first-chunk` 必须保证 batch 含 `loss_mask`。
- packed sequences 相关字段（`cu_seqlens/max_seqlen`）来自 `get_batch_on_this_tp_rank`，ARMT v1 明确禁用 `packed_seq_params`。
