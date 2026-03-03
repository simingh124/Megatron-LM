# Tensor Parallel Primitives (`megatron/core/tensor_parallel`)

## Core Function
- TP / sequence-parallel 的 gather/scatter、并行 Linear、RNG tracker、TP-friendly cross entropy。

## Directory Structure
- `layers.py`: Column/RowParallelLinear 等
- `mappings.py`: gather/scatter/reduce-scatter
- `random.py`: CUDA RNG tracker（model-parallel seed）
- `cross_entropy.py`: vocab-parallel loss

## Key Data Flow (ARMT)
- `ARMTModel._preprocess()` 在 `sequence_parallel=True` 时：
  - gather -> concat memory tokens -> scatter（保持 sequence-parallel 分片）

## Dev Notes
- v1 SP 安全约束（见 `examples/armt/armt_args.py`）：
  - `seq_length % armt_chunk_size == 0`
  - `(armt_chunk_size + num_mem_tokens) % TP == 0`
