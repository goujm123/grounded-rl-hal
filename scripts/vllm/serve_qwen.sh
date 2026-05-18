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

mkdir -p "$LOG_DIR"

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --limit-mm-per-prompt "image=30" \
    --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
    --api-key "qwen" > "$LOG_DIR/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "$FAIL_LOG"
fi