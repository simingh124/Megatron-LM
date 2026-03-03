#!/bin/bash
set -ex

# Qwen3-0.6B continual training (torch_dist ckpt), FineWeb-Edu 2025.
#
# Style reference:
#   code_repo/Megatron-LM-OPC/playground/opc_v18_pt_0b5_baseline.sh
#
# Parameters reference:
# - /mnt/step3-abla/siming/ckpts/Qwen3-0.6B-Base/config.json
# - /mnt/step3-abla/siming/ckpts/mlm/qwen3_0p6b_tp2_pp1_torch_dist/iter_0000000/run_config.yaml
#
# Only distributed settings are configurable via env vars:
#   GPUS_PER_NODE, NUM_NODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
#
# Exit interval:
#   Set EXIT_INTERVAL=1 to stop after 1 iter.

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ========== Fixed paths ==========
ROOT="/mnt/step3-abla/siming"
MEGATRON_ROOT="${ROOT}/code_repo/Megatron-LM"
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH}"

PRETRAIN_SCRIPT_PATH="${MEGATRON_ROOT}/pretrain_gpt.py"
LOAD_CHECKPOINT_PATH="${ROOT}/ckpts/mlm/qwen3_0p6b_tp2_pp1_torch_dist"
TOKENIZER_DIR="${ROOT}/tokenizers/qwen3_tokenizer"

CHECKPOINT_PATH="${ROOT}/exp_logs/checkpoints/qwen3_0p6b_mvp"
TENSORBOARD_LOGS_PATH="${ROOT}/outputs/tensorboard/qwen3_0p6b_ct_fineweb_2025"
# DATA_CACHE_PATH="${ROOT}/outputs/data_cache/fineweb_edu_2025"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"
mkdir -p "$(dirname "${TENSORBOARD_LOGS_PATH}")"
if [[ -n "${DATA_CACHE_PATH}" ]]; then
  mkdir -p "${DATA_CACHE_PATH}"
fi

if [[ ! -f "${PRETRAIN_SCRIPT_PATH}" ]]; then
  echo "ERROR: pretrain_gpt.py not found: ${PRETRAIN_SCRIPT_PATH}" >&2
  exit 1
fi
if ! command -v torchrun >/dev/null 2>&1; then
  echo "ERROR: torchrun not found in PATH" >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_DIR}" ]]; then
  echo "ERROR: tokenizer dir not found: ${TOKENIZER_DIR}" >&2
  exit 1
fi
if [[ ! -d "${LOAD_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: ckpt root dir not found: ${LOAD_CHECKPOINT_PATH}" >&2
  exit 1
fi

# ========== Distributed training setup ==========
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((${GPUS_PER_NODE} * ${NUM_NODES}))

# ========== Data ==========
# FineWeb-Edu merged-by-year, 2025 (token count: 104357702010).
DATASET_PATH="
104357702010 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2025
"

# ========== Fixed model parameters ==========
# From config.json + run_config.yaml (must match ckpt).
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1

NUM_LAYERS=28
HIDDEN_SIZE=1024
FFN_HIDDEN_SIZE=3072
NUM_ATTN_HEADS=16
NUM_QUERY_GROUPS=8
KV_CHANNELS=128

SEQ_LENGTH=32768
MAX_POSITION_EMBEDDINGS=32768

VOCAB_SIZE=151936
MAKE_VOCAB_SIZE_DIVISIBLE_BY=128

# RoPE must match the checkpoint/hf config (Qwen3 uses rope_theta=1e6).
# If rotary_base mismatches, checkpoint can still "load successfully" but the forward pass
# position encoding is inconsistent and the loss will look close to random (≈ ln(vocab)).
ROTARY_BASE=1000000
ROTARY_PERCENT=1.0

NORM_EPS=1e-6

# ========== Fixed training parameters ==========
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2

# Continual training schedule (token-based -> iters).
# Note: This corresponds to training on all FineWeb-Edu 2025 tokens; adjust in-script if you want a shorter run.
TRAIN_TOKENS=104357702010
LR_DECAY_TOKENS=${TRAIN_TOKENS}
WARMUP_TOKENS=0

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))

LR=1e-5
MIN_LR=1e-5

# EXIT_INTERVAL=1

DISTRIBUTED_ARGS=(
  --nproc_per_node ${GPUS_PER_NODE}
  --nnodes ${NUM_NODES}
  --node_rank ${NODE_RANK}
  --master_addr ${MASTER_ADDR}
  --master_port ${MASTER_PORT}
)

MODEL_ARGS=(
  --use-mcore-models
  --tensor-model-parallel-size ${TP_SIZE}
  --context-parallel-size ${CP_SIZE}
  --pipeline-model-parallel-size ${PP_SIZE}
  --num-layers ${NUM_LAYERS}
  --hidden-size ${HIDDEN_SIZE}
  --ffn-hidden-size ${FFN_HIDDEN_SIZE}
  --num-attention-heads ${NUM_ATTN_HEADS}
  --group-query-attention
  --num-query-groups ${NUM_QUERY_GROUPS}
  --kv-channels ${KV_CHANNELS}
  --seq-length ${SEQ_LENGTH}
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
  --position-embedding-type rope
  --use-rotary-position-embeddings
  --rotary-base ${ROTARY_BASE}
  --rotary-percent ${ROTARY_PERCENT}
  --normalization RMSNorm
  --norm-epsilon ${NORM_EPS}
  --qk-layernorm
  --swiglu
  --disable-bias-linear
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --bf16
  --vocab-size ${VOCAB_SIZE}
  --make-vocab-size-divisible-by ${MAKE_VOCAB_SIZE_DIVISIBLE_BY}
)

TRAINING_ARGS=(
  --micro-batch-size ${MICRO_BATCH_SIZE}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --train-iters ${TRAIN_ITERS}
  --lr-decay-iters ${LR_DECAY_ITERS}
  --lr-warmup-iters ${LR_WARMUP_ITERS}
  --lr ${LR}
  --min-lr ${MIN_LR}
  --lr-decay-style constant
  --clip-grad 1.0
  --weight-decay 0.01
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.999
  --adam-eps 1e-8
  --seed 1234
)

DATA_ARGS=(
  --data-path "${DATASET_PATH}"
  --split "100,0,0"
  --num-workers 2
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_DIR}"
)
if [[ -n "${DATA_CACHE_PATH}" ]]; then
  DATA_ARGS+=(--data-cache-path "${DATA_CACHE_PATH}")
fi

CKPT_AND_LOG_ARGS=(
  --ckpt-format torch_dist
  # Fail-fast / visibility for distributed checkpoint key mismatches.
  # Options: assume_ok_unexpected | log_unexpected | log_all | raise_unexpected | raise_all | return_unexpected | return_all | ignore_all
  # Recommendation: start with log_all, switch to raise_all when debugging.
  --dist-ckpt-strictness log_all
  --load "${LOAD_CHECKPOINT_PATH}"
  --save "${CHECKPOINT_PATH}"
  --no-load-optim
  --no-load-rng
  --no-save-optim
  --no-save-rng
  --log-interval 1
  --eval-interval 1000000000
  --eval-iters 0
  --save-interval 1000
  --tensorboard-dir "${TENSORBOARD_LOGS_PATH}"
  --distributed-timeout-minutes 60
)

EXTRA_ARGS=()
# Optional: quick debug exit (e.g., EXIT_INTERVAL=1 to stop after 1 iter).
if [[ -n "${EXIT_INTERVAL:-}" ]]; then
  EXTRA_ARGS+=(--exit-interval "${EXIT_INTERVAL}")
fi

echo "ROOT=${ROOT}"
echo "MEGATRON_ROOT=${MEGATRON_ROOT}"
echo "LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE} (GPUS_PER_NODE=${GPUS_PER_NODE}, NUM_NODES=${NUM_NODES})"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NODE_RANK=${NODE_RANK}"
echo "TRAIN_ITERS=${TRAIN_ITERS} (TRAIN_TOKENS=${TRAIN_TOKENS})"

torchrun ${DISTRIBUTED_ARGS[@]} \
  "${PRETRAIN_SCRIPT_PATH}" \
  ${MODEL_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${CKPT_AND_LOG_ARGS[@]} \
  ${EXTRA_ARGS[@]}