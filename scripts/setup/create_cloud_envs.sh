#!/usr/bin/env bash
# Recreate the separate legacy training and modern Qwen teacher environments.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_EXE="$CONDA_ROOT/bin/conda"
TRAIN_ENV="$CONDA_ROOT/envs/grounded-rl"
TEACHER_ENV="$CONDA_ROOT/envs/qwen38-serve"
CONSTRAINTS="$REPO_ROOT/scripts/setup/cloud-training-constraints.in"
if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Install Miniconda at $CONDA_ROOT, or set CONDA_ROOT to its location." >&2
    exit 1
fi
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
cd -- "$REPO_ROOT"
if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
    "$CONDA_EXE" create -y -p "$TRAIN_ENV" --override-channels -c conda-forge python=3.10 pip
fi
if [[ ! -x "$TEACHER_ENV/bin/python" ]]; then
    "$CONDA_EXE" create -y -p "$TEACHER_ENV" --override-channels -c conda-forge python=3.12 pip
fi
# The toolkit metapackage alone allows newer component versions. Pin the CUDA
# version and compiler explicitly to match the legacy Torch 2.6 CUDA 12.4 wheel.
"$CONDA_EXE" install -y -p "$TRAIN_ENV" --override-channels \
    -c nvidia/label/cuda-12.4.1 -c conda-forge \
    cuda-version=12.4 cuda-toolkit=12.4.1 cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127
"$CONDA_EXE" env config vars set -p "$TRAIN_ENV" "CUDA_HOME=$TRAIN_ENV"
uv pip install --python "$TRAIN_ENV/bin/python" -c "$CONSTRAINTS" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 setuptools wheel ninja packaging
uv pip install --python "$TRAIN_ENV/bin/python" --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
CUDA_HOME="$TRAIN_ENV" PATH="$TRAIN_ENV/bin:$PATH" DS_BUILD_OPS=0 \
    uv pip install --python "$TRAIN_ENV/bin/python" --no-build-isolation \
    -c "$CONSTRAINTS" -e . -e src/trainer/rl \
    -e 'src/trainer/offline[torch,metrics]' deepspeed==0.16.9 hf_transfer nvidia-ml-py
"$CONDA_EXE" install -y -p "$TEACHER_ENV" --override-channels \
    -c nvidia/label/cuda-13.0.2 -c conda-forge \
    cuda-version=13.0 cuda-toolkit=13.0.2 'cuda-nvcc=13.0.*' 'cuda-cudart-dev=13.0.*'
"$CONDA_EXE" env config vars set -p "$TEACHER_ENV" "CUDA_HOME=$TEACHER_ENV" \
    "LIBRARY_PATH=$TEACHER_ENV/targets/x86_64-linux/lib:$TEACHER_ENV/lib"
uv pip install --python "$TEACHER_ENV/bin/python" \
    vllm==0.26.0 transformers==5.8.0 qwen-vl-utils
"$TRAIN_ENV/bin/python" -m pip check
"$TEACHER_ENV/bin/python" -m pip check
echo "Environments ready. Run scripts/setup/smoke_training.py in grounded-rl."
