# Transformer Core (`megatron/core/transformer`)

## Core Function
- MCore Transformer 基础组件：attention/MLP/layer/block + `TransformerConfig`，供 GPT/ARMT layer specs 组装。

## Directory Structure (High-signal)
- `transformer_config.py`: `TransformerConfig`（并行/精度/attention 选项）
- `transformer_layer.py`, `transformer_block.py`: layer/block 组合与 forward
- `attention.py`, `dot_product_attention.py`, `mlp.py`: 核心算子
- `spec_utils.py`: 根据 layer spec 构建模块（ARMT/GPT 都用 spec）
- `moe/`: MoE 相关（ARMT 可选继承；v1 主要走 dense）

## Key Data Flow
1. `get_*_layer_spec(...)` -> spec（module types + init）  
2. `GPTModel/ARMTModel` 通过 spec 构建 transformer layers  
3. forward：`tokens + position_ids + attention_mask` -> hidden_states -> logits/loss

## Dev Notes
- ARMT layer spec 会在标准 TransformerLayer 中插入 `ARMTLayer`（见 `models/armt`）；需保持输入/输出 shape 与 mask dtype（bool）契约。
- attention mask 的 shape 假设需与 TBPTT chunking 兼容（可能为 `None` 或 `[b,1,s,s]`）。
