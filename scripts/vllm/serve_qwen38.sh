#!/usr/bin/env bash
# Serve the modern teacher independently of the legacy training environment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SERVE_ENV="${QWEN38_SERVE_ENV:-$HOME/miniconda3/envs/qwen38-serve}"
MODEL_PATH="${QWEN38_MODEL_PATH:-$HOME/models/Qwen3.8-27B}"
HOST="${QWEN38_HOST:-127.0.0.1}"
PORT="${QWEN38_PORT:-9011}"
TP_SIZE="${QWEN38_TP_SIZE:-1}"
MAX_PIXELS="${QWEN38_MAX_PIXELS:-1048576}"
MIN_PIXELS="${QWEN38_MIN_PIXELS:-65536}"
ENABLE_THINKING="${QWEN38_ENABLE_THINKING:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1
# FlashInfer builds its sampling kernel on first use. Keep its compiler matched
# to the teacher's CUDA 13 Torch wheel, even when called from grounded-rl.
export CUDA_HOME="${QWEN38_CUDA_HOME:-$SERVE_ENV}"
export PATH="$SERVE_ENV/bin:$PATH"
# Conda's CUDA 13 libraries live under targets/, while FlashInfer's linker also
# assumes the conventional CUDA_HOME/lib64 layout.
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ ! -x "$SERVE_ENV/bin/vllm" ]]; then
    echo "vLLM executable missing: $SERVE_ENV/bin/vllm" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Model config missing: $MODEL_PATH/config.json" >&2
    exit 1
fi
if [[ "$ENABLE_THINKING" != true && "$ENABLE_THINKING" != false ]]; then
    echo "QWEN38_ENABLE_THINKING must be true or false" >&2
    exit 1
fi
for value in "$PORT" "$TP_SIZE" "$MAX_PIXELS" "$MIN_PIXELS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Port, tensor parallel size, and pixel limits must be positive integers" >&2
        exit 1
    fi
done

cd -- "$REPO_ROOT"
printf 'Serving %s on http://%s:%s/v1 using GPUs %s (TP=%s)\n' \
    "$MODEL_PATH" "$HOST" "$PORT" "$CUDA_VISIBLE_DEVICES" "$TP_SIZE"

# Qwen3.8's image processor expresses pixel-area limits through size, rather
# than the max_pixels/min_pixels kwargs used by the older Qwen2.5 launcher.
# No reasoning parser: MCTS needs literal <think>/<answer> tags in content.
exec "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "${QWEN38_SERVED_MODEL_NAME:-qwen_vllm}" \
    --api-key "${QWEN38_API_KEY:-qwen}" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "${QWEN38_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${QWEN38_MAX_NUM_SEQS:-8}" \
    --gpu-memory-utilization "${QWEN38_GPU_MEMORY_UTILIZATION:-0.90}" \
    --limit-mm-per-prompt "{\"image\":${QWEN38_LIMIT_IMAGES:-8},\"video\":0}" \
    --mm-processor-kwargs "{\"size\":{\"longest_edge\":$MAX_PIXELS,\"shortest_edge\":$MIN_PIXELS}}" \
    --default-chat-template-kwargs "{\"enable_thinking\":$ENABLE_THINKING}" \
    "$@"
