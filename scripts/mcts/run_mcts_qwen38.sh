#!/usr/bin/env bash
# Generate new AMBER rollouts with the isolated Qwen3.8 teacher.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MCTS_PYTHON="${MCTS_PYTHON:-$HOME/miniconda3/envs/grounded-rl/bin/python}"
if [[ ! -x "$MCTS_PYTHON" ]]; then
    echo "MCTS Python is missing: $MCTS_PYTHON" >&2
    exit 1
fi
cd -- "$REPO_ROOT"
exec "$MCTS_PYTHON" scripts/mcts/qwen38_rollout.py "$@"
