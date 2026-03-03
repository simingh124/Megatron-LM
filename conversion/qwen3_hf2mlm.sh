#!/bin/bash
set -euo pipefail

ROOT="/mnt/step3-abla/siming"
MEGATRON_ROOT="${ROOT}/code_repo/Megatron-LM"
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH-}"

TP_SIZE=1
PP_SIZE=1
MP_SIZE=$((TP_SIZE * PP_SIZE))
echo "TP_SIZE: ${TP_SIZE}, PP_SIZE: ${PP_SIZE}, MP_SIZE: ${MP_SIZE}"

HF_PATH="${ROOT}/ckpts/Qwen3-0.6B-Base"
OUT_PATH="${ROOT}/ckpts/mlm/qwen3_0p6b_tp${TP_SIZE}_pp${PP_SIZE}_torch_dist"

if [ ! -d "${HF_PATH}" ]; then
  echo "Error: HF path not found: ${HF_PATH}"
  exit 1
fi

if [ -d "${OUT_PATH}" ]; then
  rm -rf "${OUT_PATH}"
fi

torchrun --nproc_per_node=${MP_SIZE} "${MEGATRON_ROOT}/conversion/convert_hf_qwen3_to_megatron.py" \
  --hf-path "${HF_PATH}" \
  --out "${OUT_PATH}" \
  --tp ${TP_SIZE} --pp ${PP_SIZE} --dtype bf16
