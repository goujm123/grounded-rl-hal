#!/bin/bash
set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set to the repository data directory before launching AMBER SFT.}"

DATASET="${DATASET:-amber_train_MCTS_20260708_223000_full_linearized_chains}"
EVAL_DATASET="${EVAL_DATASET:-amber_val_MCTS_20260708_223000_full_linearized_chains}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-$DATA_ROOT/checkpoints/sft}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_ROOT}"
DEFAULT_CONFIG_PATH="${DEFAULT_CONFIG_PATH:-examples/qwen2_5vl_amber_full_sft.yaml}"

LR="${LR:-1e-6}"
WD="${WD:-0.01}"
EPOCHS="${EPOCHS:-1}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
DEFAULT_BS="${DEFAULT_BS:-1}"
DEFAULT_GRAD_ACC="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_GRAD_ACC:-4}}"
MAX_STEPS="${MAX_STEPS:-none}"
SAVE_STEPS_DEFAULT="${SAVE_STEPS:-100}"

TEMPLATE="${TEMPLATE:-qwen2_vl}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TAG_MODEL="${TAG_MODEL:-qwen2_5_vl_7b_amber_mcts_20260708}"
TAG_BASE="${TAG:-${TAG_MODEL}_full_sft_${DATASET}}"
RANDOM_SUFFIX="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
TAG="${TAG_BASE}_$(date +'%Y%m%d_%H%M%S')_${RANDOM_SUFFIX}"

echo "DATASET: ${DATASET}"
echo "EVAL_DATASET: ${EVAL_DATASET}"
echo "MODEL: ${MODEL}"
echo "TEMPLATE: ${TEMPLATE}"
echo "TAG: ${TAG}"
echo "DEFAULT_CONFIG_PATH: ${DEFAULT_CONFIG_PATH}"
echo "IMAGE_DIR: ${IMAGE_DIR}"

TOTAL_BS=$((DEFAULT_BS * NNODES * GPUS_PER_NODE * DEFAULT_GRAD_ACC))
RUN_OUTPUT_DIR="${OUTPUT_DIR_BASE}/${TAG}_lr${LR}_wd${WD}_bs${TOTAL_BS}_epochs${EPOCHS}/"

echo "NNODES: ${NNODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "DEFAULT_GRAD_ACC: ${DEFAULT_GRAD_ACC}"
echo "TOTAL_BS: ${TOTAL_BS}"
echo "OUTPUT_DIR: ${RUN_OUTPUT_DIR}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "SAVE_STEPS: ${SAVE_STEPS_DEFAULT}"

bash examples/train_sft.sh \
  "${DATASET}" \
  "${LR}" \
  "${WD}" \
  "${DEFAULT_BS}" \
  "${EPOCHS}" \
  "${MODEL}" \
  "${RUN_OUTPUT_DIR}" \
  "${SAVE_STEPS_DEFAULT}" \
  "${TAG}" \
  "${DEFAULT_GRAD_ACC}" \
  "${DEFAULT_CONFIG_PATH}" \
  "${TEMPLATE}" \
  "${EVAL_DATASET}" \
  "${MAX_STEPS}" \
  "${IMAGE_DIR}"