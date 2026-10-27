#!/usr/bin/env python3
"""Validate the legacy MCTS/SFT/GRPO installation without loading model weights.

Each check runs in an isolated process so failures do not hide independent
results. No dataset, model, distributed worker, or training job is started.
"""

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "SMOKE_RESULT="
MODULES = {
    "mcts_imports": [
        "vlmsearch.__main__",
        "vlmsearch.tree_search.mcts_search",
        "vlmsearch.models.qwen_vllm",
    ],
    "rl_fsdp_import": ["verl.workers.fsdp_workers"],
    "rl_rollout_imports": [
        "verl.workers.rollout.vllm_rollout_spmd",
        "verl.workers.rollout.multiturn.rollout_multiturn",
    ],
    "rl_trainer_imports": ["verl.trainer.main", "verl.trainer.ray_trainer"],
}


def check_versions(_args):
    from packaging.requirements import Requirement

    requirements = [
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        "vllm==0.8.3", "transformers==4.51.0", "flash-attn>=2.4.3",
        "datasets>=2.16.0,<=3.6.0", "accelerate>=1.3.0,<=1.7.0",
        "peft>=0.14.0,<=0.15.2", "trl>=0.8.6,<=0.9.6",
        "tokenizers>=0.19.0,<=0.21.1", "numpy<2", "pydantic<=2.10.6",
        "deepspeed<=0.16.9", "tensordict", "torchdata", "ray",
        "liger-kernel", "qwen-vl-utils", "hf-transfer",
        "grounded-rl", "verl", "llamafactory",
    ]
    versions = {}
    errors = []
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            versions[requirement.name] = None
            errors.append(f"Missing {requirement.name}")
            continue
        versions[requirement.name] = installed
        if not requirement.specifier.contains(installed):
            errors.append(f"{requirement.name} {installed} does not satisfy {requirement}")
    return {"passed": not errors, "versions": versions, "errors": errors}


def check_pip(_args):
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True, capture_output=True, timeout=120,
    )
    return {
        "passed": completed.returncode == 0,
        "output": (completed.stdout + completed.stderr).strip(),
    }


def check_cuda(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check nvidia-smi, driver access, and the Torch CUDA wheel")
    count = torch.cuda.device_count()
    if count < args.expected_gpus:
        raise RuntimeError(f"Expected at least {args.expected_gpus} visible GPUs, found {count}")
    devices = []
    for index in range(count):
        with torch.cuda.device(index):
            matrix = torch.eye(64, device=f"cuda:{index}", dtype=torch.bfloat16)
            output = matrix @ matrix
            torch.cuda.synchronize(index)
            torch.testing.assert_close(output, matrix, rtol=0, atol=0)
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "bf16_matmul": "passed",
            })
    return {"devices": devices, "torch_cuda": torch.version.cuda}


def check_flash_attention(_args):
    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention requires a visible CUDA GPU")
    devices = []
    for index in range(torch.cuda.device_count()):
        generator = torch.Generator(device=f"cuda:{index}").manual_seed(17)
        q, k, v = [
            torch.randn(
                (1, 16, 2, 64), device=f"cuda:{index}",
                dtype=torch.bfloat16, generator=generator,
            )
            for _ in range(3)
        ]
        output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / 8
        reference = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
        torch.cuda.synchronize(index)
        torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.03)
        devices.append({
            "index": index,
            "max_abs_error": (output.float() - reference).abs().max().item(),
        })
    return {"devices": devices, "dtype": "bfloat16"}


def check_llamafactory(_args):
    # Require the real version checks even if the interactive shell disables them.
    os.environ.pop("DISABLE_VERSION_CHECK", None)
    from llamafactory.extras.misc import check_dependencies

    check_dependencies()
    for module in ("llamafactory.hparams.parser", "llamafactory.train.sft.workflow"):
        importlib.import_module(module)
    return {"dependency_checks": "enabled", "sft_workflow_import": "passed"}


def check_dataproto(_args):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl.protocol import DataProto

    tokens = torch.arange(12, dtype=torch.long).reshape(3, 4)
    original = DataProto.from_dict(
        tensors={"input_ids": tokens, "attention_mask": torch.ones_like(tokens)},
        non_tensors={"sample_id": np.array(["a", "b", "c"], dtype=object)},
        meta_info={"source": "environment_smoke"},
    )
    if not isinstance(original.batch, TensorDict):
        raise RuntimeError("DataProto.from_dict did not create a TensorDict")
    restored = pickle.loads(pickle.dumps(original))
    restored.check_consistency()
    torch.testing.assert_close(restored.batch["input_ids"], tokens)
    np.testing.assert_array_equal(restored.non_tensor_batch["sample_id"], ["a", "b", "c"])
    if restored.meta_info != original.meta_info or len(restored) != 3:
        raise RuntimeError("DataProto serialization lost metadata or batch dimensions")
    reconstructed = DataProto.concat(restored.chunk(3))
    torch.testing.assert_close(reconstructed.batch["input_ids"], tokens)
    return {"batch_shape": list(restored.batch.batch_size), "serialization": "passed", "chunk_concat": "passed"}


CHECKS = {
    "versions": check_versions,
    "pip_check": check_pip,
    "cuda_matmul": check_cuda,
    "flash_attention": check_flash_attention,
    **{name: None for name in MODULES},
    "llamafactory": check_llamafactory,
    "dataproto_roundtrip": check_dataproto,
}

HINTS = {
    "versions": "Activate grounded-rl and reinstall using scripts/setup/cloud-training-constraints.in; keep the teacher stack in qwen38-serve.",
    "pip_check": "Resolve the package conflicts listed by pip check without upgrading the pinned Torch/vLLM/Transformers stack.",
    "cuda_matmul": "Check nvidia-smi, CUDA_VISIBLE_DEVICES, and that the installed Torch wheel includes CUDA support.",
    "flash_attention": "Use a FlashAttention wheel matching Python, Torch 2.6, CUDA 12, and torch._C._GLIBCXX_USE_CXX11_ABI.",
    "mcts_imports": "Install the repository editable and resolve the specific missing import shown in the traceback.",
    "rl_fsdp_import": "Use the vendored src/trainer/rl package with Torch 2.6, vLLM 0.8.3, and Transformers 4.51.0.",
    "rl_rollout_imports": "The vendored rollout code requires legacy vLLM APIs and qwen-vl-utils; inspect the first failed import.",
    "rl_trainer_imports": "Check Ray, torchdata, and the vendored verl installation; inspect the first failed import.",
    "llamafactory": "Install src/trainer/offline[torch,metrics] with its dependency constraints; the traceback identifies the incompatible dependency.",
    "dataproto_roundtrip": "Check TensorDict compatibility with Torch and the vendored DataProto serialization API.",
}


def run_child(args):
    captured = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if args.check in MODULES:
                loaded = [importlib.import_module(name) for name in MODULES[args.check]]
                details = {"modules": {module.__name__: module.__file__ for module in loaded}}
            else:
                details = CHECKS[args.check](args)
        result = {"passed": True, **details}
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    result["seconds"] = round(time.monotonic() - started, 2)
    if captured.getvalue():
        result["diagnostics"] = captured.getvalue()[-6000:]
    print(RESULT_PREFIX + json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180, help="Timeout per independent check")
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file")
    parser.add_argument("--check", choices=CHECKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check:
        run_child(args)
        return
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    results = {}
    for name in CHECKS:
        command = [sys.executable, str(Path(__file__).resolve()), "--check", name, "--expected-gpus", str(args.expected_gpus)]
        try:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, text=True,
                capture_output=True, timeout=args.timeout,
            )
            lines = [line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
            if not lines:
                raise RuntimeError(f"Check exited {completed.returncode} without a result: {(completed.stdout + completed.stderr)[-6000:]}")
            results[name] = json.loads(lines[-1][len(RESULT_PREFIX):])
            if completed.returncode:
                results[name].update(passed=False, exit_code=completed.returncode)
            if completed.stderr:
                results[name]["process_stderr"] = completed.stderr[-3000:]
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            results[name] = {"passed": False, "error": str(exc)}
        if not results[name]["passed"]:
            results[name]["hint"] = HINTS[name]
    passed = all(result["passed"] for result in results.values())
    report = {
        "passed": passed,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "checks": results,
        "scope": "Dependency/import/kernel smoke checks; no model weights or training job loaded",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
