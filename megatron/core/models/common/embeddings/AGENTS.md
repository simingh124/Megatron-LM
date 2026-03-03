# Positional & Token Embeddings (`megatron/core/models/common/embeddings`)

## Core Function
- 提供 token embedding + position embedding（RoPE/Yarn/relative），被 `GPTModel` 与 `ARMTModel` 复用。

## Directory Structure
- `language_model_embedding.py`: token embedding（word + position，含 dropout）
- `rotary_pos_embedding.py`: RoPE 实现（`rotary_pos_emb(seq_len, ...)`）
- `yarn_rotary_pos_embedding.py`: Yarn RoPE scaling
- `rope_utils.py`: RoPE 计算辅助
- `relative_pos_embedding.py`: 相对位置（非 ARMT v1 主要路径）

## Key Data Flow (ARMT)
1. `GPTModel/ARMTModel._preprocess()` 调 `self.rotary_pos_emb(seq_len, ...)`
2. ARMT 追加 memory tokens 后以 `new_seq_len = seq_len + num_mem_tokens` 重新生成 rotary embeddings

## Dev Notes
- `position_embedding_type`：
  - `rope`: 通常返回 RoPE embedding（以及实现相关的 cos/sin 变体）
  - `yarn`: 可能返回 `(rotary_pos_emb, ...)`；调用方需按返回值协议处理
- 修改返回值结构/shape 时同步检查 `GPTModel._preprocess()` 与 `ARMTModel._preprocess()`。
