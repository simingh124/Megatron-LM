# Core Tokenizers (`megatron/core/tokenizers`)

## Core Function
- MCore tokenizer 抽象：`MegatronTokenizer`（统一 HF/SentencePiece/TikToken/Null），主要供 core datasets 构建时使用。

## Directory Structure
- `megatron_tokenizer.py`: `MegatronTokenizer.from_pretrained(...)` + metadata 驱动加载
- `base_tokenizer.py`: 抽象基类
- `text/utils/build_tokenizer.py`: 根据 `args.tokenizer_type` 选择 library/path/kwargs
- `text/libraries/huggingface_tokenizer.py`: `HuggingFaceTokenizer` 实现

## Key Data Flow
1. `pretrain_gpt.py:core_gpt_dataset_config_from_args()`：
   - `args.legacy_tokenizer=False`（默认）-> `text/utils/build_tokenizer.build_tokenizer(args)`
2. `MegatronTokenizer.from_pretrained(tokenizer_path, metadata)` -> 提供 `tokenize()/detokenize()/eod/...`
3. tokenizer 注入 `GPTDatasetConfig(tokenizer=...)`，dataset 用它计算 EOD、mask、position_ids

## Dev Notes
- 与 `megatron/training/tokenizer` 并存：training side 仍会构建 legacy tokenizer 并设置 `args.padded_vocab_size`；dataset side 默认用这里的 tokenizer。
- `HuggingFaceTokenizer` 依赖 `transformers`；`--trust-remote-code` 会影响可复现。
- core build_tokenizer 不会自动设置 `args.padded_vocab_size`；训练侧依赖该值构建模型 vocab（ARMT/GPT）。
