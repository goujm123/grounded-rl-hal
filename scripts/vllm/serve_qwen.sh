#!/bin/bash
set -o pipefail

# check if argument is provided, otherwise use default
CHECKPOINT_PATH="${1:-Qwen/Qwen2.5-VL-72B-Instruct}"
NUM_GPUS="${2:-4}"
PORT="${3:-9011}"
export CUDA_VISIBLE_DEVICES="${4:-0,1,2,3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# # Check if checkpoint path exists
# if [ ! -e "$CHECKPOINT_PATH" ]; then
#     echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
#     exit 1
# fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set GPU memory utilization and other parameters. Keep some headroom for CUDA/NCCL cleanup.
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"   # originally none
LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-30}"
MAX_PIXELS="${VLLM_MAX_PIXELS:-4194304}"                        # originally 12960000
MIN_PIXELS="${VLLM_MIN_PIXELS:-4096}"
USE_FAST="${VLLM_USE_FAST:-true}"                               # originally none
# MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"                    # originally none
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"                    # originally none
TEE_LOGS="${VLLM_TEE_LOGS:-true}"                               # originally none

echo "VLLM_USE_FAST: $USE_FAST"
echo "VLLM_MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "VLLM_ENFORCE_EAGER: $ENFORCE_EAGER"

mkdir -p vllm_logs
LOGFILE="vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
echo "Logging vLLM output to: $LOGFILE"

VLLM_CMD=(
    vllm serve "$CHECKPOINT_PATH"
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    # --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --disable-custom-all-reduce \
    --limit-mm-per-prompt "image=$LIMIT_IMAGES" \
    --mm-processor-kwargs "{\"max_pixels\":$MAX_PIXELS,\"min_pixels\":$MIN_PIXELS,\"use_fast\":$USE_FAST}" \
    --api-key "qwen"
)

if [[ "$ENFORCE_EAGER" == "true" ]]; then
    VLLM_CMD+=(--enforce-eager)
fi

if [[ "$TEE_LOGS" == "true" ]]; then
    stdbuf -oL -eL "${VLLM_CMD[@]}" 2>&1 | tee "$LOGFILE"
    VLLM_EXIT=${PIPESTATUS[0]}
else
    stdbuf -oL -eL "${VLLM_CMD[@]}" > "$LOGFILE" 2>&1
    VLLM_EXIT=$?
fi

# Check if the command succeeded, and log a failure message if not
if [ $VLLM_EXIT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Command failed" | tee -a "logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
fi
