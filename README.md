# ViGoRL: Visually Grounded Reinforcement Learning

This repository contains the official code implementation for the paper:

> **Grounded Reinforcement Learning for Visual Reasoning**
> Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, Katerina Fragkiadaki

[**Paper**](https://arxiv.org/abs/2505.23678) | [**Datasets**](https://huggingface.co/datasets/gsarch/vigorl_datasets) | [**Models**](https://huggingface.co/collections/gsarch/vigorl-6855655677fd6ff5864f65f2) | [**Project Page**](https://visually-grounded-rl.github.io/)

---

## Overview

Visually-Grounded Reinforcement Learning (**ViGoRL**) integrates vision-language models (VLMs) with reinforcement learning (RL) to produce spatially grounded reasoning steps explicitly anchored to visual regions. By using a novel multi-turn RL formulation, ViGoRL dynamically zooms into relevant image areas to improve visual attention, grounding, and reasoning capabilities across various visual reasoning tasks.

---

## Features

* **Supported Models**

  * Qwen2/Qwen2.5-VL
  * Easily supports additional models compatible with vLLM

* **Supported Algorithms**

  * Async, distributed Monte Carlo Tree Search (MCTS) over thought steps with VLMs (& visual grounding), and tree search linearization
  * Supervised Fine-Tuning (SFT)
  * Group Relative Policy Optimization (GRPO) reinforcement learning
  * Multi-turn GRPO with visual feedback loops
  * Async, distributed VLM evaluation 

* **Supported Tasks and Benchmarks**

  * **Spatial Reasoning:** SAT-2, BLINK, RoboSpatial
  * **Visual Search:** V\*Bench
  * **Web Grounding and Interaction:** OS-ATLAS, ScreenSpot (V2 and Pro), VisualWebArena
  * **Any vision-text dataset** in a specific format

* **Key Features**

  * Spatial grounding of textual reasoning steps to visual coordinates
  * Dynamic, zoom-in visual feedback for multi-turn reasoning
  * Efficient, async inference via vLLM and multiprocessing

---

## Requirements

* Python >= 3.9
* transformers >= 4.51.0
* flash-attn >= 2.4.3
* vllm >= 0.8.3

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Gabesarch/grounded-rl.git
cd grounded-rl
```

### Environment Setup

```bash
conda create -n grounded-rl python=3.10
conda activate grounded-rl
pip install uv
uv pip install -e .
uv pip install flash-attn --no-build-isolation
uv pip install deepspeed<=0.16.9

# RL dependencies
cd src/trainer/rl
uv pip install -e .
cd ../../..

# SFT dependencies
cd src/trainer/offline
uv pip install -e ".[torch,metrics]"
```

### Cloud server setup with Qwen3.8-27B

The cloud setup uses two conda environments because the vendored training code
depends on older vLLM worker APIs and Transformers versions:

| Environment | Purpose | Main versions |
| --- | --- | --- |
| `grounded-rl` | MCTS client, SFT, and GRPO | Python 3.10, Torch 2.6.0/CUDA 12.4, vLLM 0.8.3, Transformers 4.51.0, FlashAttention 2.7.4.post1, DeepSpeed 0.16.9 |
| `qwen38-serve` | Qwen3.8 teacher server | Python 3.12, Torch 2.11.0/CUDA 13.0, vLLM 0.26.0, Transformers 5.8.0 |

Miniconda is installed at `~/miniconda3`. Open a new Bash shell, or initialize it
in the current shell, then activate the project environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grounded-rl
python scripts/setup/smoke_training.py --output reports/setup/training-smoke.json
```

The official BF16 teacher checkpoint is at `~/models/Qwen3.8-27B`, pinned to
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its
`download-manifest.json` records all 32 upstream file hashes and sizes (55.586 GB).
The download and verification script is `~/models/download_qwen38.py`.

Start the teacher from the repository root in a separate terminal:

```bash
bash scripts/vllm/serve_qwen38.sh
# After the server is ready, from grounded-rl:
python scripts/setup/smoke_teacher.py --legacy-client
```

The launcher selects the teacher environment automatically, uses GPU 0 in BF16,
and serves `qwen_vllm` at `http://127.0.0.1:9011/v1` with API key `qwen`.
It defaults to 32,768 context tokens, eight concurrent sequences, and native
thinking disabled, preserving the MCTS client's literal `<think>`/`<answer>`
output. Stop the server with Ctrl-C before using both GPUs for training.

To distribute the teacher across both GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 QWEN38_TP_SIZE=2 bash scripts/vllm/serve_qwen38.sh
```

Use `QWEN38_MODEL_PATH`, `QWEN38_PORT`, `QWEN38_MAX_MODEL_LEN`,
`QWEN38_MAX_NUM_SEQS`, and `QWEN38_GPU_MEMORY_UTILIZATION` for other settings.
Keep the MCTS client's `PORT` equal to `QWEN38_PORT` when changing the port.
`scripts/setup/create_cloud_envs.sh` recreates both environments with their
matching CUDA compiler toolkits; it requires Miniconda and `uv`. Installed package
snapshots and validation reports are saved locally under `reports/setup/`.
Datasets and student checkpoints are configured separately through the existing
data and training scripts.

---

## Data Setup

Set the dataset path environment variable before running scripts:

```bash
export DATA_ROOT="/path/to/your/data/root"
```

### Downloading Datasets

Run the provided script to download and extract data:

```bash
python download_data.py
```

Datasets include:

* Spatial reasoning tasks (SAT-2, BLINK)
* Visual search tasks (V\*Bench)
* Web grounding and action tasks (OS-ATLAS, ScreenSpot, VisualWebArena)

---

## Model Checkpoints

Pretrained ViGoRL checkpoints are available via Hugging Face:

| Task                  | 3B Model                                                                                                                                                                                                                  | 7B Model                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Visual Search**     | [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-3b-Visual-Search), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-Multiturn-MCTS-SFT-3b-Visual-Search)                                                        | [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-7b-Visual-Search), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-Multiturn-MCTS-SFT-7b-Visual-Search) |
| **Web Grounding**     | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Web-Grounding), [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-3b-Web-Grounding), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-3b-Web-Grounding) | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Web-Grounding)                                                                                                    |
| **Web Action**        | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Web-Action)                                                                                                                                                              | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Web-Action)                                                                                                       |
| **Spatial Reasoning** | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Spatial), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-3b-Spatial)                                                                                           | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Spatial), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-7b-Spatial)                                    |

---

## Demo Scripts

We provide demo scripts to run ViGoRL checkpoints interactively on a single image. Two versions are available:

* `demo/demo_singleturn.py`: Run and visualize our singe-turn variant. Generates a single output of multiple grounded reasoning steps for a given image and query.
* `demo/demo_multiturn.py`: Run and visualize our multi-turn variant. Iteratively generates reasoning steps, dynamically feeding cropped images back to the model based on model outputs until reaching a final answer.

### Single-Turn Demo

**Usage Example:**

```bash
# Spatial reasoning example
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Spatial \
    --image demo/examples/LivingRoom.jpg \
    --query "What is above the blue lamp?"

# Web grounding example
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Web-Grounding \
    --image demo/examples/APnews.png \
    --query "Description: check sports news"

# Web action example
python demo/demo_singleturn.py \
    --model gsarch/ViGoRL-7b-Web-Action \
    --image demo/examples/osclass_page.png \
    --query "OBJECTIVE: Identify the insect in the picture. Leave a comment with the title \"Questions\" and text containing the insect's identity, with the purpose of confirming with the seller.\n\nPREVIOUS ACTIONS: \n1. type  [INPUT] [] [Questions]\n\nIMAGE:\n"
```

### Multi-Turn Demo

**Usage Example:**

```bash
# Visual search example
python demo/demo_multiturn.py \
    --model gsarch/ViGoRL-Multiturn-7b-Visual-Search \
    --image demo/examples/man.jpg \
    --query "What color is the man's shirt?" \
    --crop_offset 182 \
    --crop_size 672

# Web grounding example
python demo/demo_multiturn.py \
    --model gsarch/ViGoRL-Multiturn-3b-Web-Grounding \
    --image demo/examples/APnews.png \
    --query "Description: check sports news" \
    --draw_dot \
    --crop_offset 100 \
    --crop_size 512
```

---

## Repository Structure

```
grounded-rl
├── data/                     # Data and rollouts
├── scripts/
│   ├── evaluation/           # Evaluation scripts
│   └── mcts/                 # MCTS-related scripts
├── src/
│   ├── vlmsearch/
│   │   ├── arguments.py
│   │   └── tree_search/
│   │       ├── mcts_search.py
│   │       └── single_path_rollouts.py
│   └── trainer/
│       ├── offline/          # SFT (Llama-Factory based)
│       └── rl/               # GRPO (EasyR1 based)
└── download_data.py
```

---

## Training Pipeline

### Step 1: Run MCTS to Generate Grounded Traces

```bash
bash scripts/mcts/run_mcts_qwen72b.sh
```

* Specify dataset and vLLM port at the top of the script.

### Step 2: Linearize MCTS Trees

Single-turn:

```bash
python scripts/mcts/build_reasoning_chains_from_mcts.py
```

Multi-turn:

```bash
python scripts/mcts/build_reasoning_chains_from_mcts_multiturn.py
```

### Step 3: Supervised Fine-Tuning (SFT)

Add reasoning chains to `src/trainer/offline/data/dataset_info.json` and run:

```bash
cd src/trainer/offline
bash examples/train_qwen2_5_vl_sft.sh
```

### Step 4: Reinforcement Learning (GRPO)

Run GRPO on top of the SFT model:

```bash
cd src/trainer/rl
bash examples/run_vigorl.sh
```

Checkpoints are sharded; convert to HF format using:

```bash
python src/trainer/rl/scripts/model_merger.py
```

---

## Evaluation

Configure `MODEL` and `SYSTEM_PROMPT` variables at the top of each script:

* Multi-turn visual search & ScreenSpot:

```bash
bash scripts/evaluation/eval_multiturn.sh
```

* Spatial reasoning (BLINK, SAT-2):

```bash
bash scripts/evaluation/eval_spatial.sh
```

* Web grounding (ScreenSpot-Pro):

```bash
bash scripts/evaluation/eval_web_grounding.sh
```

---

## FAQ

### Q: I’m getting the error `KeyError: 'visual.patch_embed.proj.weight'`. What should I do?

This error typically indicates a version mismatch between the model and the installed `transformers` library. To resolve it, downgrade your `transformers` version to:

```bash
pip install transformers==4.51.0
```

### Q: I'm getting the error `ImportError: .../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: ...`. What should I do?

This indicates a problem with flash attention version compilation. To resolve, compile with an alternate flash attention version:

```bash
pip install flash-attn==2.5.1.post1 --no-build-isolation
```

---

## Citation

```bibtex
@article{sarch2025vigorl,
    title={Grounded Reinforcement Learning for Visual Reasoning},
    author={Sarch, Gabriel and Saha, Snigdha and Khandelwal, Naitik and Jain, Ayush and Tarr, Michael J and Kumar, Aviral and Fragkiadaki, Katerina},
    year={2025}
}
```

---

## Acknowledgements
This project builds on the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) projects to support visually-grounded RL, we thank all the authors for providing such a high-performance training framework.

## License

The code is released under [MIT License](LICENSE).
