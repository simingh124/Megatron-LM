# Shared Model Components (`megatron/core/models/common`)

## Core Function
- GPT/ARMT 等模型共用的 embedding、language module 抽象与运行时计划结构。

## Directory Structure
- `embeddings/`: token embedding + RoPE/Yarn/relative pos embedding。
- `language_module/`: `LanguageModule` 基类（forward/loss 约定）。
- `model_chunk_schedule_plan.py`: schedule plan 数据结构（启用 overlap/plan 时使用）。

## Key Data Flow
- `GPTModel/ARMTModel` 在 `_preprocess()` 中构建 `decoder_input + rotary_pos_emb`（见 `embeddings/`）。

## Dev Notes
- ARMT 会改变序列长度（追加 `num_mem_tokens`）；任何依赖 `seq_len` 的 embedding/positional 逻辑都必须可在运行时重算。
