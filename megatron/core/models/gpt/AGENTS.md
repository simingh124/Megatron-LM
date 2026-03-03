# GPT Base Model (`megatron/core/models/gpt`)

## Core Function
- Decoder-only GPT 实现：`GPTModel` + layer/module specs（TE/Torch），作为 `ARMTModel` 的基座。

## Directory Structure
- `gpt_model.py`: `GPTModel`（embedding -> transformer blocks -> lm head）
- `gpt_layer_specs.py`: `get_gpt_layer_*_spec`（选择 attention/mlp/layernorm 实现）
- `*_module_specs.py`: MoE/实验 attention 变体 specs

## Key Data Flow (ARMT inherits)
1. `examples/armt/train.py:model_provider()` 构建 `ARMTModel(GPTModel subclass)`
2. `GPTModel._preprocess()`：
   - token -> embedding（见 `models/common/embeddings`）
   - build rotary pos emb（rope/yarn）
3. transformer blocks（见 `core/transformer`）-> logits / loss
4. ARMT 覆写 `_preprocess()`：在序列维拼接 memory tokens，并重新计算 rotary embedding；最后在输出前 strip memory tokens

## Dev Notes
- vocab：训练侧必须使用 `args.padded_vocab_size`（与 TP padding 对齐）。
- position embedding：ARMT v1 仅支持 `rope/yarn`（见 `examples/armt/armt_args.py`）。
- sequence_parallel=True 时，ARMT 需要 `tensor_parallel.gather/scatter` 来拼接/裁剪 memory tokens；修改相关逻辑需同步考虑 TP 分片约束。
