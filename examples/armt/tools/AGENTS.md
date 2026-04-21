# ARMT Checkpoint Tools (`examples/armt/tools`)

## Core Function
- 将 baseline GPT checkpoint 转为包含 ARMT 新参数的 checkpoint（便于从非 ARMT ckpt 启动训练）。

## Directory Structure
- `convert_baseline_to_armt.py`
  - `ARMTCheckpointConfig`: 转换所需结构参数（`num_layers/hidden_size/...`）
  - `convert_baseline_to_armt(...)`: 主转换逻辑

## Key Data Flow (convert_baseline_to_armt)
1. 读取 baseline：支持两种格式
   - 直接 `state_dict`
   - wrapper：`{"model": state_dict}`
2. 拷贝 baseline 权重 -> `armt_state`
3. 新增/补齐 ARMT 参数：
   - `memory_embeddings`: `[num_mem_tokens, hidden_size]`（std 参考词嵌入）
   - 每层 `decoder.layers.{i}.recurrent_memory_layer.*`：
     - `W_mq/W_mk`: `trunc_normal_`
     - `W_mv`: 全零初始化（residual-friendly）
     - `W_mb`: `gating` 决定输出维（`hidden_size` vs `armt_n_heads`）
4. 保持 wrapper 结构（若输入是 wrapper，则输出也是 wrapper）。

## Dev Notes
- 该工具面向 `torch.save` 的单文件 ckpt；不处理 `torch_dist` 分布式 checkpoint 目录格式。
- runtime buffers（如 `W_mem/z`）不应进入 checkpoint（实现侧应保持 `persistent=False`）。
- 若修改 ARMT 参数命名/层级，请同步更新：
  - `megatron/core/models/armt/`（state_dict key）
  - `examples/armt/tests/test_armt_checkpoint.py`
