# Model Zoo (`megatron/core/models`)

## Core Function
- MCore 模型集合与后端选择：以 `TransformerConfig + layer_spec` 组装模型（GPT/ARMT 等）。
- ARMT 训练以 `gpt/` 为基座，在 `armt/` 注入 memory tokens 与 associative layer。

## Directory Structure (ARMT path)
- `gpt/`: `GPTModel`（decoder-only 基座）+ specs。
- `armt/`: `ARMTModel/ARMTLayer/AssociativeLayer`（memory tokens + TBPTT 相关 reset/detach 语义）。
- `common/`: embeddings（RoPE/Yarn）与 `LanguageModule` 抽象。
- `backends.py`: Transformer Engine / Torch 等实现选择（与 `--transformer-impl`/spec 相关）。

## Key Data Flow
1. `examples/armt/train.py`：
   - `get_armt_layer_spec(...)` 选择/组装 layer spec
   - `ARMTModel(config, transformer_layer_spec, ...)`
2. `ARMTModel` 复用 `GPTModel` 的 transformer stack，但在 `_preprocess()` 里：
   - 拼接 `num_mem_tokens`
   - 重算 RoPE/Yarn positional embedding
   - 输出前 strip memory tokens（保证 logits/labels 对齐）

## Dev Notes
- checkpoint 与 spec/shape 强绑定：转换（`conversion/`）与训练必须使用一致的 layer spec，否则表现接近随机初始化。
- `vocab_size` 一律用 `args.padded_vocab_size`（TP padding 对齐），不要直接用原始 vocab。
