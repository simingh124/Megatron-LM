# Legacy Tokenizer Plumbing (`megatron/training/tokenizer`)

## Core Function
- 构建 “legacy” tokenizer（`global_vars.get_tokenizer()`），并设置 `args.padded_vocab_size`（训练/模型 vocab 对齐）。

## Directory Structure
- `tokenizer.py`: `build_tokenizer()`（按 `--tokenizer-type` 分派），`_vocab_size_with_padding()`
- `bert_tokenization.py`, `gpt2_tokenization.py`: 传统 tokenizer 实现
- `multimodal_tokenizer.py`, `sft_tokenizer.py`: 多模态 / SFT 包装

## Key Data Flow
1. `initialize_megatron()` -> `set_global_variables()` -> `_build_tokenizer()` -> `build_tokenizer(args)`
2. 产物挂到 `_GLOBAL_TOKENIZER`，并写回 `args.padded_vocab_size`

## Dev Notes (ARMT)
- dataset tokenizer 默认走 `megatron/core/tokenizers`；这里只负责全局 tokenizer 与 `padded_vocab_size`。
- `HuggingFaceTokenizer`（这里的 `_HuggingFaceTokenizer`）使用 `transformers.AutoTokenizer.from_pretrained()`；`--trust-remote-code` 会影响可复现/安全。
- padding：`padded_vocab_size = ceil(vocab / (make_vocab_size_divisible_by*TP)) * (...)`；必须与 `--vocab-size/--make-vocab-size-divisible-by`、checkpoint 一致。
