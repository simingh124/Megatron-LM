#!/usr/bin/env bash
set -euo pipefail

python examples/armt/train.py \
  --use-recurrent-model-schedule \
  --num-mem-tokens 16 \
  --recurrent-chunk-size 512 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8
