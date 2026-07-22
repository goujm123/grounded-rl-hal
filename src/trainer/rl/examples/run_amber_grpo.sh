#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

: "${DATA_ROOT:?DATA_ROOT must point to the repository data directory.}"
DATA_ROOT=$(cd "$DATA_ROOT" && pwd)

RUN_MODE=${RUN_MODE:-smoke}
NUM_GPUS=${NUM_GPUS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-$DATA_ROOT/checkpoints/sft/qwen2_5_vl_7b_amber_mcts_20260708_full_sft_amber_train_MCTS_20260708_223000_full_linearized_chains_20260717_112449_0ebe2291_lr1e-6_wd0.01_bs8_epochs1}
TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/mllm_hal/rl/amber_discriminative_train.jsonl}
VAL_FILE=${VAL_FILE:-$DATA_ROOT/mllm_hal/rl/amber_discriminative_val.jsonl}
FORMAT_PROMPT=${FORMAT_PROMPT:-$SCRIPT_DIR/format_prompt/amber_discriminative_grounded_thinking.jinja}
REWARD_FILE=${REWARD_FILE:-$SCRIPT_DIR/reward_function/amber_discriminative.py}
CONFIG_PATH=${CONFIG_PATH:-$SCRIPT_DIR/config.yaml}
SAVE_PATH_BASE=${SAVE_PATH_BASE:-$DATA_ROOT/checkpoints/rl}
RESUME_FROM=${RESUME_FROM:-}
DRY_RUN=${DRY_RUN:-false}

LR=${LR:-1.0e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-2}
KL_COEF=${KL_COEF:-1.0e-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MIN_PIXELS=${MIN_PIXELS:-3136}
MAX_PIXELS=${MAX_PIXELS:-262144}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}

case "$RUN_MODE" in
  val)
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
    NUM_ROLLOUTS=${NUM_ROLLOUTS:-2}
    EXPERIENCE_MICRO_BATCH=${EXPERIENCE_MICRO_BATCH:-2}
    MAX_STEPS=${MAX_STEPS:-1}
    VAL_ONLY=true
    VAL_BEFORE_TRAIN=true
    VAL_AFTER_TRAIN=false
    VAL_FREQ=-1
    SAVE_FREQ=${SAVE_FREQ:--1}
    SAVE_AFTER_TRAIN=${SAVE_AFTER_TRAIN:-false}
    export WANDB_MODE=${WANDB_MODE:-disabled}
    ;;
  smoke)
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
    NUM_ROLLOUTS=${NUM_ROLLOUTS:-2}
    EXPERIENCE_MICRO_BATCH=${EXPERIENCE_MICRO_BATCH:-2}
    MAX_STEPS=${MAX_STEPS:-1}
    VAL_ONLY=false
    VAL_BEFORE_TRAIN=false
    VAL_AFTER_TRAIN=false
    VAL_FREQ=-1
    SAVE_FREQ=${SAVE_FREQ:-1}
    SAVE_AFTER_TRAIN=${SAVE_AFTER_TRAIN:-true}
    export WANDB_MODE=${WANDB_MODE:-disabled}
    ;;
  train)
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    NUM_ROLLOUTS=${NUM_ROLLOUTS:-4}
    EXPERIENCE_MICRO_BATCH=${EXPERIENCE_MICRO_BATCH:-4}
    MAX_STEPS=${MAX_STEPS:-100}
    VAL_ONLY=false
    VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
    VAL_AFTER_TRAIN=${VAL_AFTER_TRAIN:-true}
    VAL_FREQ=${VAL_FREQ:-25}
    SAVE_FREQ=${SAVE_FREQ:-25}
    SAVE_AFTER_TRAIN=${SAVE_AFTER_TRAIN:-true}
    ;;
  *)
    echo "RUN_MODE must be one of: val, smoke, train." >&2
    exit 2
    ;;
esac

UPDATE_MICRO_BATCH=${UPDATE_MICRO_BATCH:-1}
TOTAL_EPISODES=${TOTAL_EPISODES:-1000}
SAVE_LIMIT=${SAVE_LIMIT:-3}
VAL_GENERATIONS_TO_LOG=${VAL_GENERATIONS_TO_LOG:-10}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

if (( NUM_GPUS != 2 )); then
  echo "This launcher is validated for NUM_GPUS=2; got $NUM_GPUS." >&2
  exit 2
fi
if (( ROLLOUT_BATCH_SIZE % GLOBAL_BATCH_SIZE != 0 )); then
  echo "ROLLOUT_BATCH_SIZE must be divisible by GLOBAL_BATCH_SIZE." >&2
  exit 2
fi
if (( (ROLLOUT_BATCH_SIZE * NUM_ROLLOUTS) % EXPERIENCE_MICRO_BATCH != 0 )); then
  echo "ROLLOUT_BATCH_SIZE * NUM_ROLLOUTS must be divisible by EXPERIENCE_MICRO_BATCH." >&2
  exit 2
fi
if (( (GLOBAL_BATCH_SIZE * NUM_ROLLOUTS / NUM_GPUS) % UPDATE_MICRO_BATCH != 0 )); then
  echo "Per-device expanded GLOBAL_BATCH_SIZE must be divisible by UPDATE_MICRO_BATCH." >&2
  exit 2
fi
if (( NUM_ROLLOUTS <= 1 )); then
  echo "GRPO requires NUM_ROLLOUTS greater than one." >&2
  exit 2
fi

for required_file in "$TRAIN_FILE" "$VAL_FILE" "$FORMAT_PROMPT" "$REWARD_FILE" "$CONFIG_PATH"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Required file is missing or empty: $required_file" >&2
    exit 2
  fi
done
for required_file in config.json preprocessor_config.json tokenizer.json model.safetensors.index.json; do
  if [[ ! -s "$MODEL_PATH/$required_file" ]]; then
    echo "SFT checkpoint is missing $required_file: $MODEL_PATH" >&2
    exit 2
  fi
done

python3 - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
with (model_path / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("Checkpoint index contains no weight shards.")
missing = [shard for shard in shards if not (model_path / shard).is_file()]
if missing:
    raise SystemExit(f"Checkpoint is missing weight shards: {missing}")
print(f"Checkpoint preflight found {len(shards)} weight shards.")
PY

DATETIME=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-amber_discriminative_grpo_${RUN_MODE}_rbs${ROLLOUT_BATCH_SIZE}_n${NUM_ROLLOUTS}_lr${LR}_${DATETIME}}
SAVE_PATH="$SAVE_PATH_BASE/$RUN_NAME"
mkdir -p "$SAVE_PATH"
LOG_FILE="$SAVE_PATH/train.log"

command=(
  python3 -m verl.trainer.main
  "config=$CONFIG_PATH"
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "data.prompt_key=prompt"
  "data.answer_key=answer"
  "data.image_key=images"
  "data.image_root=$DATA_ROOT"
  "data.format_prompt=$FORMAT_PROMPT"
  "data.rollout_batch_size=$ROLLOUT_BATCH_SIZE"
  "data.val_batch_size=$VAL_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.min_pixels=$MIN_PIXELS"
  "data.max_pixels=$MAX_PIXELS"
  "algorithm.adv_estimator=grpo"
  "algorithm.disable_kl=false"
  "algorithm.use_kl_loss=true"
  "algorithm.kl_coef=$KL_COEF"
  "worker.actor.model.model_path=$MODEL_PATH"
  "worker.actor.model.trust_remote_code=true"
  "worker.actor.model.freeze_vision_tower=true"
  "worker.actor.model.enable_gradient_checkpointing=true"
  "worker.actor.global_batch_size=$GLOBAL_BATCH_SIZE"
  "worker.actor.micro_batch_size_per_device_for_update=$UPDATE_MICRO_BATCH"
  "worker.actor.micro_batch_size_per_device_for_experience=$EXPERIENCE_MICRO_BATCH"
  "worker.actor.optim.lr=$LR"
  "worker.actor.optim.weight_decay=$WEIGHT_DECAY"
  "worker.actor.optim.strategy=adamw"
  "worker.actor.fsdp.enable_full_shard=true"
  "worker.actor.fsdp.enable_cpu_offload=false"
  "worker.actor.offload.offload_params=true"
  "worker.actor.offload.offload_optimizer=true"
  "worker.actor.padding_free=true"
  "worker.rollout.n=$NUM_ROLLOUTS"
  "worker.rollout.temperature=1.0"
  "worker.rollout.top_p=0.95"
  "worker.rollout.tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
  "worker.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  "worker.rollout.limit_images=1"
  "worker.rollout.multiturn=false"
  "worker.rollout.max_model_len=2048"
  "worker.rollout.val_override_config.temperature=0.0"
  "worker.rollout.val_override_config.n=1"
  "worker.ref.fsdp.enable_cpu_offload=false"
  "worker.ref.offload.offload_params=true"
  "worker.reward.reward_function=$REWARD_FILE:amber_compute_score"
  "trainer.project_name=amber_grpo"
  "trainer.experiment_name=$RUN_NAME"
  "trainer.n_gpus_per_node=$NUM_GPUS"
  "trainer.nnodes=1"
  "trainer.total_episodes=$TOTAL_EPISODES"
  "trainer.max_steps=$MAX_STEPS"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "trainer.val_after_train=$VAL_AFTER_TRAIN"
  "trainer.val_only=$VAL_ONLY"
  "trainer.val_freq=$VAL_FREQ"
  "trainer.val_generations_to_log=$VAL_GENERATIONS_TO_LOG"
  "trainer.save_checkpoint_path=$SAVE_PATH"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.save_after_train=$SAVE_AFTER_TRAIN"
  "trainer.save_limit=$SAVE_LIMIT"
)

if [[ -n "$RESUME_FROM" ]]; then
  if [[ ! -d "$RESUME_FROM" || "$(basename "$RESUME_FROM")" != global_step_* ]]; then
    echo "RESUME_FROM must be an existing verl global_step_* directory." >&2
    exit 2
  fi
  command+=("trainer.load_checkpoint_path=$RESUME_FROM")
fi

echo "RUN_MODE: $RUN_MODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "TRAIN_FILE: $TRAIN_FILE"
echo "VAL_FILE: $VAL_FILE"
echo "SAVE_PATH: $SAVE_PATH"
printf 'COMMAND:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

cleanup() {
  status=$?
  trap - EXIT INT TERM
  ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

cd "$RL_ROOT"
"${command[@]}" 2>&1 | tee "$LOG_FILE"