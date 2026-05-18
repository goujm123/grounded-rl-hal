#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
OFFLINE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$OFFLINE_ROOT" || exit 1

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
export DATA_ROOT

# replace with the names of the datasets for train/val as logged in data/dataset_info.json
DATASET="${DATASET:-amber_train_MCTS_linearized_chains}"
EVAL_DATASET="${EVAL_DATASET:-amber_val_MCTS_linearized_chains}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/checkpoints/sft}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_ROOT}"
DEFAULT_CONFIG_PATH="${DEFAULT_CONFIG_PATH:-examples/qwen2vl_full_sft.yaml}"

LR=("${LR:-1e-6}")
WD=("${WEIGHT_DECAY:-0.01}")
EPOCHS=("${EPOCHS:-1}")
NNODES_SWEEP=("${NNODES:-1}")
DEFAULT_BS="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
DEFAULT_GRAD_ACC_SWEEP=("${GRADIENT_ACCUMULATION_STEPS:-4}")
MAX_STEPS="${MAX_STEPS:-none}"
SAVE_STEPS_DEFAULT="${SAVE_STEPS:-100}"

TEMPLATE="${TEMPLATE:-qwen2_vl}"
echo "TEMPLATE: ${TEMPLATE}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TAG_MODEL="${TAG_MODEL:-qwen2_5_vl_7b}"

# tags
TAG="${TAG_MODEL}_full_sft_${DATASET}"

RANDOM_SUFFIX=$(tr -dc 'a-zA-Z0-9' </dev/urandom | head -c8 || true)
TAG="${TAG}_$(date +'%Y%m%d_%H%M%S')_${RANDOM_SUFFIX}"

echo "TAG: $TAG"

echo "Starting combined parameter sweep"

# sweep over a single combination
combinations=(
    # Combination 1
    "${LR[0]} ${WD[0]} ${EPOCHS[0]} ${NNODES_SWEEP[0]} ${DEFAULT_GRAD_ACC_SWEEP[0]}"
)

for combo in "${combinations[@]}"; do
    lr=$(echo $combo | awk '{print $1}')
    wd=$(echo $combo | awk '{print $2}')
    epochs=$(echo $combo | awk '{print $3}')
    if [ "$NNODES_SWEEP" ]; then
        NNODES=$(echo $combo | awk '{print $4}')
        DEFAULT_GRAD_ACC=$(echo $combo | awk '{print $5}')
    fi
    echo "NNODES: ${NNODES}"
    echo "DEFAULT_GRAD_ACC: ${DEFAULT_GRAD_ACC}"
    TOTAL_BS=$((DEFAULT_BS * NNODES * 8 * DEFAULT_GRAD_ACC))
    echo "Submitting job for lr=${lr}, wd=${wd}, epochs=${epochs}, bs=${TOTAL_BS}"  # Debug statement
    OUTPUT_DIR="${OUTPUT_DIR}/${TAG}_lr${lr}_wd${wd}_bs${TOTAL_BS}_epochs${epochs}/"
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    bash examples/train_sft.sh $DATASET $lr $wd $DEFAULT_BS $epochs $MODEL $OUTPUT_DIR $SAVE_STEPS_DEFAULT $TAG $DEFAULT_GRAD_ACC $DEFAULT_CONFIG_PATH $TEMPLATE $EVAL_DATASET $MAX_STEPS $IMAGE_DIR
    sleep 1
done