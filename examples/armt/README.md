# ARMT (Associative Recurrent Memory Transformer)

Experimental ARMT training entrypoint for Megatron-LM.

## V1 Constraints

- Pipeline parallel size must be 1.
- Context parallel size must be 1.
- Packed sequence is not supported.
- Position embedding type must be RoPE or Yarn.
- FP8 is not supported (TE bf16/fp16 is allowed).
- Sequence parallel is supported in safety mode:
  - `seq_length % armt_chunk_size == 0` (unless padding is implemented)
  - `(armt_chunk_size + num_mem_tokens) % TP == 0`

## Usage

```bash
python examples/armt/train.py \
  --use-armt-tbptt \
  --num-mem-tokens 16 \
  --armt-chunk-size 512 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8
```
