# ARMT (Associative Recurrent Memory Transformer)

Experimental ARMT training entrypoint for Megatron-LM.

## V1 Constraints

- Pipeline parallel size must be 1.
- Context parallel size must be 1.
- Packed sequence is not supported.
- Position embedding type must be RoPE or Yarn.
- FP8 is not supported (TE bf16/fp16 is allowed).
- Sequence parallel is supported in safety mode:
  - `seq_length % recurrent_chunk_size == 0` (unless padding is implemented)
  - `(recurrent_chunk_size + num_mem_tokens) % TP == 0`

Legacy ARMT flag names such as `--use-armt-tbptt` and `--armt-chunk-size` remain accepted as
compatibility aliases.

## Usage

```bash
python examples/armt/train.py \
  --use-recurrent-tbptt \
  --num-mem-tokens 16 \
  --recurrent-chunk-size 512 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8
```
