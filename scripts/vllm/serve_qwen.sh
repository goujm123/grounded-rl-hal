#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="$REPO_ROOT/vllm_logs"
FAIL_LOG="$REPO_ROOT/logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"

# check if argument is provided, otherwise use default
CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3

# # Check if checkpoint path exists
# if [ ! -e "$CHECKPOINT_PATH" ]; then
#     echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
#     exit 1
# fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"

GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-30}"
MAX_PIXELS="${VLLM_MAX_PIXELS:-4194304}"
MIN_PIXELS="${VLLM_MIN_PIXELS:-4096}"

echo "GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "LIMIT_IMAGES: $LIMIT_IMAGES"
echo "MAX_PIXELS: $MAX_PIXELS"
echo "MIN_PIXELS: $MIN_PIXELS"

mkdir -p "$LOG_DIR"

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --limit-mm-per-prompt "image=$LIMIT_IMAGES" \
    --mm-processor-kwargs "{\"max_pixels\":$MAX_PIXELS,\"min_pixels\":$MIN_PIXELS}" \
    --api-key "qwen" > "$LOG_DIR/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "$FAIL_LOG"
fi