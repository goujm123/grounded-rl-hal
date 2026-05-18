#!/bin/bash
set -eo pipefail

DATASET=$1
LR=$2
WD=$3
BS=$4
EPOCHS=$5
MODEL=$6
OUTPUT_DIR=$7
SAVE_STEPS=$8
TAG=$9
GRAD_ACC=${10}
DEFAULT_CONFIG_PATH=${11}
TEMPLATE=${12}
EVAL_DATASET=${13}
MAX_STEPS=${14}
IMAGE_DIR=${15}

echo "DATASET: $DATASET"
echo "LR: $LR"
echo "WD: $WD"
echo "BS: $BS"
echo "EPOCHS: $EPOCHS"
echo "MODEL: $MODEL"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SAVE_STEPS: $SAVE_STEPS"
echo "TAG: $TAG"
echo "GRAD_ACC: $GRAD_ACC"
echo "DEFAULT_CONFIG_PATH: $DEFAULT_CONFIG_PATH"
echo "TEMPLATE: $TEMPLATE"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "MAX_STEPS: $MAX_STEPS"
echo "IMAGE_DIR: $IMAGE_DIR"

export NNODES=1 #$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export FORCE_TORCHRUN=1

if [ "$MAX_STEPS" == "none" ]; then
  args=""
else
  args="--max_steps $MAX_STEPS"
fi

# Preprocess the YAML file to substitute parameters
python examples/generate_yaml.py \
  --default_yaml "$DEFAULT_CONFIG_PATH" \
  --output_yaml examples/qwen2vl_full_sft_parsed.yaml \
  --dataset "$DATASET" \
  --lr "$LR" \
  --wd "$WD" \
  --epochs "$EPOCHS" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --template "$TEMPLATE" \
  --save_steps "$SAVE_STEPS" \
  --eval_dataset "$EVAL_DATASET" \
  --media_dir "$IMAGE_DIR" \
  $args

# Command to execute
llamafactory-cli train examples/qwen2vl_full_sft_parsed.yaml