# ARMT Core Model Module (`megatron/core/models/armt`)

## Core Function
- 在 `GPTModel/TransformerLayer` 基础上实现 ARMT v1：
  - 追加每层可写 memory tokens（associative memory）
  - 支持 TBPTT（跨 chunk 梯度断开）语义

## Directory Structure
- `armt_model.py`: `ARMTModel(GPTModel)`，负责拼接/剥离 memory tokens 与 attention mask 适配。
- `armt_layer.py`: `ARMTLayer(TransformerLayer)`，封装“检索 -> 注意力/MLP -> 写入”的层逻辑。
- `associative_layer.py`: `AssociativeLayer`（W_mem/z 状态 + DPFP 特征映射）。
- `armt_layer_specs.py`: `get_armt_layer_spec(...)`，将 GPT layer spec 替换为 `ARMTLayer`。

## Key Data Flow
- `ARMTModel.forward(...)`
  1. `GPTModel._preprocess(...)` 得到 `decoder_input/padding_mask/...`
  2. `decoder_input` 末尾拼接 `memory_embeddings`（S -> S+M）
  3. 扩展 `attention_mask` 到 `(S+M, S+M)` 并重新生成 RoPE/Yarn embedding
  4. `self.decoder(...)` 执行 ARMT layers
  5. 输出隐藏态剥离 memory token 部分（S+M -> S），再走 `_postprocess`（LM head/loss）
- `ARMTLayer.forward(...)`
  1. `recurrent_memory_layer.associate()` 检索记忆并残差相加
  2. `TransformerLayer.forward()`（注意力 + FFN）
  3. 取尾部 `M` 个 memory token hidden 做 `recurrent_memory_layer.update_mem()` 写入
- `AssociativeLayer`
  - 状态：`W_mem`（memory matrix）与可选 `z`（denom）；按 batch 动态 reshape/zero
  - TBPTT：`tbptt_mode=True` 时 update 使用 `detach()`，避免跨 chunk 梯度与 in-place 版本冲突

## Dev Notes
- v1 假设：
  - packed sequence 不支持（`PackedSeqParams` 必须为 `None`）
  - 默认使用 SBH layout；memory tokens 追加在序列末尾
- 并行注意点：
  - 开启 `sequence_parallel` 时会对序列维 gather/scatter（memory token 拼接与写入都依赖）。
- checkpoint 命名约定：
  - 顶层：`memory_embeddings`
  - 每层：`decoder.layers.{i}.recurrent_memory_layer.*`
  - 变更命名/shape 时同步更新转换工具与测试。
