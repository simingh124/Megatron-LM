#!/bin/bash
set -ex

# Qwen3-0.6B ARMT continual training from torch_dist ckpt.
#
# Verification goal:
# - Training can start from converted checkpoint.
# - ARMT-specific params are newly created and kept random-initialized.
# - Loss is not close to random guess (≈ ln(vocab)).
#
# Distributed settings are configurable via env vars:
#   GPUS_PER_NODE, NUM_NODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
#
# Exit interval:
#   Set EXIT_INTERVAL=1 to stop after 1 iter.
#
# Optional:
#   ENABLE_TEST_TRAIN_RUN=1    add --test-train-run and disable output_logs tee by default

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ========== For test ==========
ENABLE_TEST_TRAIN_RUN=${ENABLE_TEST_TRAIN_RUN:-0}

# ========== Distributed runtime toggles ==========
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-1}  # shard optimizer state across DP ranks
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}  # overlap gradient reduction with backward
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-1}  # overlap parameter gather with forward
USE_NCCL_UB=${USE_NCCL_UB:-0}  # enable NCCL user buffers for comm
LOG_THROUGHPUT=${LOG_THROUGHPUT:-1}  # print throughput metrics in logs


# ========== Distributed training setup ==========
GPUS_PER_NODE=${PROC_PER_NODE:-8}
NUM_NODES=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-9899}
WORLD_SIZE=$((${GPUS_PER_NODE} * ${NUM_NODES}))

# ========== Fixed paths ==========
ROOT="/mnt/step3-abla/siming"
MEGATRON_ROOT="${ROOT}/code_repo/Megatron-LM"
VENV_PYTHON=${VENV_PYTHON:-"${ROOT}/.venv/bin/python"}
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH}"

# Read from file name without extension
EXP_NAME=$(basename "${BASH_SOURCE[0]}" ".sh")

PRETRAIN_SCRIPT_PATH="${MEGATRON_ROOT}/examples/armt/train.py"
LOAD_CHECKPOINT_PATH="${ROOT}/ckpts/mlm/qwen3_0p6b_tp2_pp1_torch_dist"
TOKENIZER_DIR="${ROOT}/tokenizers/qwen3_tokenizer"

CHECKPOINT_PATH="${ROOT}/exp_logs/checkpoints/rmt_qwen/${EXP_NAME}"
TENSORBOARD_LOGS_PATH="${ROOT}/exp_logs/tensorboard/rmt_qwen/${EXP_NAME}"
LOG_DIR="${ROOT}/exp_logs/output_logs/rmt_qwen/${EXP_NAME}"
mkdir -p "$(dirname "${CHECKPOINT_PATH}")"
mkdir -p "$(dirname "${TENSORBOARD_LOGS_PATH}")"

# ========== Optional terminal+file logging ==========
# Default off for smoke tests. Set ENABLE_TEE_LOG=1 to enable explicitly.
if [[ "${ENABLE_TEST_TRAIN_RUN}" == "1" ]]; then
  ENABLE_TEE_LOG=${ENABLE_TEE_LOG:-0}
else
  ENABLE_TEE_LOG=${ENABLE_TEE_LOG:-1}
fi
if [[ "${ENABLE_TEE_LOG}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  LOG_TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/train_${LOG_TS}.log"
  if [[ "${NUM_NODES}" -gt 1 ]]; then
    LOG_FILE="${LOG_DIR}/train_${LOG_TS}.node${NODE_RANK}.log"
  fi
  if [[ -z "${__SCRIPT_TEE_ACTIVE:-}" ]]; then
    export __SCRIPT_TEE_ACTIVE=1
    exec > >(tee "${LOG_FILE}") 2>&1
  fi
  echo "LOG_FILE=${LOG_FILE}"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: venv python not found or not executable: ${VENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAIN_SCRIPT_PATH}" ]]; then
  echo "ERROR: ARMT train entrypoint not found: ${PRETRAIN_SCRIPT_PATH}" >&2
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

# ========== Data ==========
# FineWeb-Edu merged-by-year, 2013-2025.
DATASET_PATH="
22715400849 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2013 \
92435520518 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2014 \
108810954465 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2015 \
99634946152 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2016 \
168671396875 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2017 \
156577020935 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2018 \
152974268028 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2019 \
123847804696 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2020 \
144518165251 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2021 \
111847170240 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2022 \
112493210234 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2023 \
168470576123 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2024 \
104357702010 ${ROOT}/pt_data/fineweb_edu_by_year_merged/2025
"

# ========== Fixed model parameters ==========
# Must match the checkpoint.
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1

NUM_LAYERS=28
HIDDEN_SIZE=1024
FFN_HIDDEN_SIZE=3072
NUM_ATTN_HEADS=16
NUM_QUERY_GROUPS=8
KV_CHANNELS=128

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=32768

VOCAB_SIZE=151936
MAKE_VOCAB_SIZE_DIVISIBLE_BY=128

ROTARY_BASE=1000000
ROTARY_PERCENT=1.0

NORM_EPS=1e-6

# ========== ARMT parameters ==========
NUM_MEM_TOKENS=16
ARMT_CHUNK_SIZE=512
ARMT_N_HEADS=32
ARMT_HEAD_SIZE=64
ARMT_D_MEM=$(( ${ARMT_N_HEADS} * ${ARMT_HEAD_SIZE} ))

# ========== Fixed training parameters (smoke-friendly defaults) ==========
MICRO_BATCH_SIZE=30
GLOBAL_BATCH_SIZE=480
NUM_WORKERS=${NUM_WORKERS:-32}


TRAIN_TOKENS=100000000000
LR_DECAY_TOKENS=${TRAIN_TOKENS}
WARMUP_TOKENS=$(( 1000 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LENGTH} ))

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))

if [[ "${OVERLAP_PARAM_GATHER}" == "1" && "${USE_DISTRIBUTED_OPTIMIZER}" != "1" ]]; then
  echo "ERROR: OVERLAP_PARAM_GATHER=1 requires USE_DISTRIBUTED_OPTIMIZER=1" >&2
  exit 1
fi

if [[ "${OVERLAP_PARAM_GATHER}" == "1" && "${OVERLAP_GRAD_REDUCE}" != "1" ]]; then
  echo "ERROR: OVERLAP_PARAM_GATHER=1 requires OVERLAP_GRAD_REDUCE=1" >&2
  exit 1
fi

LR=5e-4
MIN_LR=1e-5

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

ARMT_ARGS=(
  --use-armt-tbptt
  --num-mem-tokens ${NUM_MEM_TOKENS}
  --armt-chunk-size ${ARMT_CHUNK_SIZE}
  --armt-n-heads ${ARMT_N_HEADS}
  --armt-d-mem ${ARMT_D_MEM}
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
  --weight-decay 0.1
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --seed 42
)

DATA_ARGS=(
  --data-path "${DATASET_PATH}"
  --split "100,0,0"
  --num-workers ${NUM_WORKERS}
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_DIR}"
)

CKPT_AND_LOG_ARGS=(
  --ckpt-format torch_dist
  # Important: baseline ckpt doesn't have ARMT params; drop those "unexpected" keys.
  --dist-ckpt-strictness log_all
  --load "${LOAD_CHECKPOINT_PATH}"
  --save "${CHECKPOINT_PATH}"
  --no-load-optim
  --no-load-rng
  # --no-save-optim
  # --no-save-rng
  --log-interval 1
  --eval-interval 1000000000
  --eval-iters 0
  --save-interval 2000
  --tensorboard-dir "${TENSORBOARD_LOGS_PATH}"
  --distributed-timeout-minutes 60
)

EXTRA_ARGS=()
if [[ "${USE_DISTRIBUTED_OPTIMIZER}" == "1" ]]; then
  EXTRA_ARGS+=(--use-distributed-optimizer)
fi
if [[ "${OVERLAP_GRAD_REDUCE}" == "1" ]]; then
  EXTRA_ARGS+=(--overlap-grad-reduce)
fi
if [[ "${OVERLAP_PARAM_GATHER}" == "1" ]]; then
  EXTRA_ARGS+=(--overlap-param-gather)
fi
if [[ "${USE_NCCL_UB}" == "1" ]]; then
  EXTRA_ARGS+=(--use-nccl-ub)
fi
if [[ "${LOG_THROUGHPUT}" == "1" ]]; then
  EXTRA_ARGS+=(--log-throughput)
fi

if [[ -n "${EXIT_INTERVAL:-}" ]]; then
  EXTRA_ARGS+=(--exit-interval "${EXIT_INTERVAL}")
fi
if [[ "${ENABLE_TEST_TRAIN_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--test-train-run)
fi

echo "ROOT=${ROOT}"
echo "MEGATRON_ROOT=${MEGATRON_ROOT}"
echo "VENV_PYTHON=${VENV_PYTHON}"
echo "LOAD_CHECKPOINT_PATH=${LOAD_CHECKPOINT_PATH}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE} (GPUS_PER_NODE=${GPUS_PER_NODE}, NUM_NODES=${NUM_NODES})"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NODE_RANK=${NODE_RANK}"
echo "TRAIN_TOKENS=${TRAIN_TOKENS} TRAIN_ITERS=${TRAIN_ITERS}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "NUM_MEM_TOKENS=${NUM_MEM_TOKENS} ARMT_CHUNK_SIZE=${ARMT_CHUNK_SIZE} ARMT_D_MEM=${ARMT_D_MEM}"
echo "ENABLE_TEST_TRAIN_RUN=${ENABLE_TEST_TRAIN_RUN}"

echo "NUM_WORKERS=${NUM_WORKERS}"
echo "USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER} OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE} OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER}"
echo "USE_NCCL_UB=${USE_NCCL_UB} LOG_THROUGHPUT=${LOG_THROUGHPUT}"

${VENV_PYTHON} -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} \
  "${PRETRAIN_SCRIPT_PATH}" \
  ${ARMT_ARGS[@]} \
  ${MODEL_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${CKPT_AND_LOG_ARGS[@]} \
  ${EXTRA_ARGS[@]}
