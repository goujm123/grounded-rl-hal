#!/bin/bash

# check if argument is provided, otherwise use default
CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3
CUDA_VISIBLE_DEVICES=$4

# # Check if checkpoint path exists
# if [ ! -e "$CHECKPOINT_PATH" ]; then
#     echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
#     exit 1
# fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set GPU memory utilization and other parameters, increasing available KV cache size for Qwen-72B
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"    # originally 0.9
LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-30}"
MAX_PIXELS="${VLLM_MAX_PIXELS:-12960000}"                       # originally 12960000
MIN_PIXELS="${VLLM_MIN_PIXELS:-4096}"

mkdir -p vllm_logs

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --limit-mm-per-prompt "image=$LIMIT_IMAGES" \
    --mm-processor-kwargs "{\"max_pixels\":$MAX_PIXELS,\"min_pixels\":$MIN_PIXELS}" \
    --api-key "qwen" > "vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi