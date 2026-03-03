# HF -> Megatron Checkpoint Conversion (`conversion`)

## Core Function
- 将 HuggingFace Qwen3 权重转换为 Megatron-Core `torch_dist` 分布式 checkpoint（供 ARMT/MLM 训练 `--load`）。

## Directory Structure
- `convert_hf_qwen3_to_megatron.py`: 主转换器（初始化 TP/PP + weight mapping + `dist_checkpointing.save`）。
- `qwen3_hf2mlm.sh`: 参考封装（`torchrun --nproc_per_node=tp*pp`）。

## Key Data Flow
1. `torchrun ... convert_hf_qwen3_to_megatron.py --hf-path ... --out ... --tp ... --pp ...`
2. `_init_distributed()` -> `parallel_state.initialize_model_parallel(tp, pp, CP=1)`
3. 读取 HF `config.json` + weights（`*.safetensors`/`pytorch_model*.bin`）
4. 构建 `TransformerConfig` + `GPTModel(layer_spec=TE spec)`，按 TP shard 拆分/合并权重（QKV merge）
5. `dist_checkpointing.save(...)` 写入 `out/`（含 metadata）

## Dev Notes
- `layer_spec` 必须与训练一致，否则 layout 不匹配会导致效果接近随机初始化。
- 期望 `WORLD_SIZE == tp*pp`；默认 `CP=1`。
- baseline -> ARMT 会新增参数；torch_dist load 时需用 `--dist-ckpt-strictness log_*` 允许丢弃 unexpected keys（见 `examples/armt/train.py`）。
