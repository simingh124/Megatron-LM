#!/usr/bin/env bash
set -euo pipefail

python examples/armt/train.py \
  --use-armt-tbptt \
  --num-mem-tokens 16 \
  --armt-chunk-size 512 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8
