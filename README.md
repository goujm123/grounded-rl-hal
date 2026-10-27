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
thinking disabled. The production rollout launcher uses literal
`<evidence>`/`<answer>` blocks. Stop the server with Ctrl-C before using both GPUs
for training.

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

### AMBER, POPE, and LongHalQA

The hallucination collection is stored under `data/hallucination/`. Its downloader
preserves official annotations, pins GitHub/Hugging Face revisions, records file
hashes, and fully decodes each image. It downloads only the 500 images referenced
by original COCO POPE, rather than the entire COCO image archive.

```bash
uv run --no-project --with requests --with gdown --with huggingface_hub \
  --with pyarrow --with Pillow scripts/data/download_hallucination_data.py --data-root data
conda activate grounded-rl
python scripts/data/prepare_hallucination_splits.py --data-root data --inspect-only
python scripts/data/prepare_hallucination_splits.py --data-root data
python scripts/data/prepare_hallucination_splits.py --data-root data --audit
```

The prepared collection freezes AMBER at approximately 80/10/10 by image cluster;
all POPE and LongHalQA variants remain test data. Identity grouping uses canonical
source IDs, file hashes, decoded RGB hashes, and conservative perceptual matching.
Perceptual candidates have contact sheets for inspection. A changed source,
policy, or image requires a new output directory; an existing manifest cannot be
silently replaced. Large crops or edits can evade perceptual matching, so the
audit records the matching criteria and their limits.

| Dataset | Train records | Dev records | Test records |
| --- | ---: | ---: | ---: |
| AMBER discriminative | 11,282 | 1,447 | 1,487 |
| Original COCO POPE, three settings | 0 | 0 | 9,000 |
| LongHalQA, seven tasks | 0 | 0 | 9,461 |

Important local files:

* `data/hallucination/raw/download-manifest.json`: source revisions, files, and checksums.
* `data/hallucination/prepared/manifest.json`: the shared sample/image/cluster assignments.
* `data/hallucination/prepared/amber/{train,dev,test}.mcts.jsonl`: binary rollout inputs.
* `data/hallucination/prepared/amber/{train,dev,test}.rl.jsonl`: GRPO inputs with the same assignments.
* `data/hallucination/prepared/evaluation/`: full POPE and LongHalQA inputs, including task-specific files and A–D choices.
* `data/hallucination/prepared/{report,audit}.json`: counts and independent image/export verification.

The downloaded LongHalQA release contains 25 identical image/question groups
with conflicting labels, involving 51 records. Official labels and all 9,461
evaluation records are preserved. `label-conflicts.json` records the affected
groups; `evaluation/longhalqa.unambiguous.jsonl` excludes those records for a
separate 9,410-record sensitivity analysis. Do not describe that filtered result
as the original benchmark score. LongHalQA's `answer` field is authoritative;
its constant `category="hallucination"` is not a binary label.

The versioned `configs/data/amber_train_review_v1.json` records automated visual
review of all 51 flagged training annotation conflicts, ten additional suspects,
and two controls. It conservatively excludes 52 records and retains 11 reviewed
records, leaving 11,230 eligible training questions. Official gold labels and
frozen split assignments remain unchanged. The overlay records the question,
image hash, rationale, confidence, and decision for each reviewed record. Its
content hash and source hashes are verified before training inference. Excluded
IDs are filtered before sampling and cannot enter reviewed SFT exports.

Prepare a balanced, stratified Qwen3.8 pilot:

```bash
python scripts/data/prepare_amber_pilot.py
bash scripts/mcts/run_mcts_qwen38.sh --dry-run
DATA_FILE=data/hallucination/pilots/amber_150_seed42_v1/pilot.mcts.jsonl \
  SAVE_TAG=PILOT_AMBER_150_QWEN38 MAX_NEW_TOKENS=256 N_ROLLOUTS_PER_NODE=1 \
  QWEN38_MAX_EVIDENCE_STEPS=1 QWEN38_EVIDENCE_MAX_NEW_TOKENS=128 \
  bash scripts/mcts/run_mcts_qwen38.sh --simulations 2 --max-depth 4 --rollout-max-steps 2
```

This selects 150 distinct image groups with 75 yes and 75 no answers across the
nine available question-type/label cells, including 32 large images and the
largest eligible image. Existence questions in this release are all negative;
there is no original positive-existence cell to sample. `selection.json` freezes
the selected IDs, quotas, dimensions, and input/review/split hashes. Changed
selection inputs require a new output directory. These bounded pilot settings
retain two children per expansion while limiting repeated observations. Record
the search budget when comparing yield or estimating a larger run.
The normalized Qwen protocol defaults to one balanced evidence observation per
trajectory, followed by a forced final answer. Evidence calls are capped at 128
tokens. This budget counts generated history only; a question or few-shot
example containing evidence tags cannot consume it. Legacy protocols retain
their previous behavior. Rollouts record the generation policy and budget hits.

The launcher validates input labels, image checksums, the frozen manifest, and
training review before starting inference. It uses the isolated teacher environment, verifies
the endpoint's model identity, disables native thinking, and requests one visible
evidence step or final answer at a time. It reuses a matching running server;
otherwise it starts one and stops only the processes it owns when finished.
Failures propagate a nonzero status. Development/test rollouts require an
explicit `--split dev|test --allow-held-out` and retain their evaluation assignment.
The old `run_mcts_qwen72b.sh` name delegates AMBER runs to this launcher.

The teacher uses literal `<evidence>` blocks for visible observations, keeping
them separate from Qwen's native thinking delimiter. SFT conversion turns those
observations into the student's established `<think>` format. Stop strings keep
each API response to one complete evidence or answer block.
Native evidence decoding also constrains coordinate parentheses to complete
numeric pairs in the 0–1000 grid. This prevents malformed or out-of-range pairs;
the later image-based critic still checks whether a point identifies its claim.

New rollout records preserve their split, image cluster, manifest/review hashes,
image geometry, timing/API counters, and raw final response. The teacher uses an
explicit normalized 0–1000 coordinate grid. RGB/PNG uploads use the same factor-32
resize policy and pixel budget as the server and a bounded process-local cache;
the original image bytes remain unchanged. SFT conversion checks every point,
converts valid points to zero-based original-image pixels, and preserves both
the source geometry and coordinate mapping. Negative, nonfinite, malformed, or
out-of-bounds coordinates are rejected even when coordinate-free observations
are allowed. Old coordinate-bearing rollouts without geometry require an
explicit `--legacy-coordinate-frame original_pixels`; their frame is never
guessed from the numbers.

The deterministic scorer requires one unambiguous answer and
compares the complete binary/choice label, rejecting substring matches such as
`unknown` for `no`, duplicate answers, and malformed output. It does not verify
visual explanations or coordinate accuracy; audit pilot reasoning separately.

Linearize a completed run using its frozen manifest (also copied into the run):

```bash
python scripts/mcts/build_reasoning_chains_from_mcts.py \
  --data-str data/mcts/YOUR_RUN_DIRECTORY --data-root data \
  --split-manifest data/hallucination/prepared/manifest.json
```

The converter validates all source points before conservative observation
deduplication, validates the retained points again, and records removed spans.
Cleanup preserves different counts, negation, colors, directional roles, and
distinct image locations. These exports are candidates with
`quality_status=pending_visual_verification`, not quality-approved training data.

The converter preserves train/dev/test assignments, writing separate train,
validation, and test files. `scripts/rl/build_amber_discriminative_rl.py` also
accepts `--split-manifest` and detects this prepared manifest by default, rather
than making an independent RL split. LongHalQA choice tasks remain evaluation
data and are not silently converted to binary training questions.
The existing coordinate filter is configurable with `--allow-no-coordinate` for
observations such as image-wide absence; coordinate presence does not establish
localization correctness.

Apply semantic quality checks before using candidate SFT for training. Keep a
matching teacher server running for this stage (for example, start
`bash scripts/vllm/serve_qwen38.sh` in a separate terminal):

```bash
python scripts/mcts/filter_reasoning_quality.py \
  --input-dir data/mcts/YOUR_RUN_DIRECTORY/reasoning_chains \
  --rollout-dir data/mcts/YOUR_RUN_DIRECTORY \
  --selection data/hallucination/pilots/amber_150_seed42_v1/selection.json \
  --output-dir data/mcts/YOUR_RUN_DIRECTORY/sft_quality_filtered \
  --cache-dir data/hallucination/quality_cache \
  --base-url http://127.0.0.1:9011/v1 --workers 4
```

The filter first answers the image question without seeing the proposed answer
or explanation. It then checks every material claim. A separate pass describes
orientation, specific lying posture, material and explicit chromatic-color
attributes with the claimed values masked; the result must agree with the literal source claim. It describes
numbered point centers without candidate text, and compares those centers with
the claimed referents using fresh full-image views and local crops. Local
part/contact phrases are bound to their point: a hoof claim cannot be approved
using a broader sheep reference. Ordinary object anchors may still identify any
visible part of that object. Gold labels, rewards,
sample IDs, and prior audit findings are never sent to the critic. Uncertainty,
unsupported assertions, wrong or imprecise referents, repeated observations,
incomplete responses, and protocol errors withhold the entire chain.

The filter revalidates exact source-chain derivation, image bytes, geometry,
review decisions, and frozen partitions. Its locked cache binds requests to the
checkpoint, prompts, inference settings, images, and schema. New filtered files
include accepted-decision provenance; `quality_decisions.jsonl` preserves every
rejection and `quality_manifest.json` records completion and output hashes.
These files record automated acceptance. A missing completion manifest indicates
an unfinished or failed filtering run. Output directories must be new; cached
critiques can be reused on retries. For a full training run omit `--selection`;
it is a pilot-membership check, not a second data split.

The critic uses the same model with separate prompts and can share its visual
errors. Calibrate it against separately reviewed training images outside the
pilot and prior review sets. Split these references by image group into a
calibration partition and a locked validation partition. Freeze references before
critic calls, tune only on calibration, and freeze the policy before validation.
Authored correct/corrupted pairs are useful challenge cases, but their error
frequencies do not estimate natural rollout error rates. Visually inspect a
stratified retained pilot sample as well; automated acceptance is not a visual
correctness guarantee.

Evaluate completed calibration ledgers without changing any critic decisions:

```bash
python scripts/mcts/evaluate_critic_calibration.py \
  --fixture reports/YOUR_REVIEW/calibration_fixture.json \
  --ledger reports/YOUR_REVIEW/critic_gpu0/quality_decisions.jsonl \
  --ledger reports/YOUR_REVIEW/critic_gpu1/quality_decisions.jsonl \
  --partition calibration --output reports/YOUR_REVIEW/calibration_evaluation.json
```

Use `--partition validation --policy-lock PATH` for locked validation. The
evaluator documents the lock schema and checks that the source hashes, model,
settings and prompt fingerprints were frozen before validation calls. It audits
exact reference coverage and the bound requests, responses and retry attempts.
False acceptances, false rejections, uncertainty and protocol failures are
reported separately; failed verification earns no bad-claim rejection credit.

If a completed visual audit flags a question, quarantine every accepted variant
of that question in a new pilot training export:

```bash
python scripts/mcts/apply_reasoning_review.py \
  --input-dir data/mcts/YOUR_RUN_DIRECTORY/sft_quality_filtered \
  --audit reports/YOUR_PILOT/visual_review/visual_audit.json \
  --output-dir data/mcts/YOUR_RUN_DIRECTORY/sft_reviewed
```

The curator binds the audit to the exact source exports and reviewed chains,
preserves the original filter decisions and audit counts, and writes a separate
exclusion ledger and completion manifest. Train on this post-audit export when
review exclusions exist. Unreviewed retained chains remain automated decisions;
quarantining known concerns does not establish their visual correctness.

Analyze the pilot after SFT conversion, supplying its saved selection and run:

```bash
python scripts/mcts/analyze_amber_pilot.py \
  --selection data/hallucination/pilots/amber_150_seed42_v1/selection.json \
  --rollouts-dir data/mcts/YOUR_RUN_DIRECTORY \
  --sft-dir data/mcts/YOUR_RUN_DIRECTORY/reasoning_chains \
  --output reports/amber_pilot/report.json
```

The report separates search yield, parsing validity, coordinate bounds, and
retention by question type and gold label. Gold-guided MCTS success is not a
held-out benchmark accuracy estimate. Visually inspect retained claims and
anchors before scaling; valid syntax and a correct final label alone do not
establish visual support.

These are project-held-out partitions. Previously trained checkpoints retain
their earlier data exposure; start a fresh student from the same public base
when comparing teachers under this split. Run regression checks with
`python -m unittest discover -s tests -v`.

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
