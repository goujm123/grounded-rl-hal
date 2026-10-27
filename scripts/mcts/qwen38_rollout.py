#!/usr/bin/env python3
"""Run AMBER MCTS against a verified Qwen3.8 endpoint with owned-process cleanup."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from vlmsearch.data_manifest import FrozenDataManifest
from vlmsearch.data_review import ReviewedTrainingData
from vlmsearch.image_geometry import NORMALIZED_COORDINATE_INSTRUCTION


REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """You are a careful visual verifier answering a yes/no question about an image.
Check visible evidence for the queried object, attribute, count, action, or relation.
Return exactly ONE of these blocks, with no text outside it:
<evidence>A brief visible observation, with a representative coordinate (x, y) when a location helps.</evidence>
<answer> yes </answer>
<answer> no </answer>
If the user's message has no Previous evidence, start with one concise visual observation.
Previous evidence in the user's message contains your earlier observations. When it is present, answer as soon as the evidence is sufficient; add another observation only if needed.
Coordinates are evidence anchors, not proof of a claim. Check both entities for relations.
Do not infer details that are not visible. Do not repeat evidence or invent extra steps.
Never write multiple answer blocks or text after the final answer block."""
SYSTEM_PROMPT += "\n" + NORMALIZED_COORDINATE_INSTRUCTION
SYSTEM_PROMPT += "\nFor an absent object, describe the visible scene or checked regions; do not invent a point for the absent object."


class EndpointMismatch(RuntimeError):
    pass


def verify_endpoint(base_url, api_key, model_name, model_path):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise EndpointMismatch(f"Teacher /models returned HTTP {exc.code}; check the API key and endpoint") from exc
    except (urllib.error.URLError, TimeoutError):
        return False
    cards = [card for card in payload.get("data", []) if card.get("id") == model_name]
    if len(cards) != 1:
        raise EndpointMismatch(f"Endpoint does not advertise exactly one {model_name!r} model")
    advertised = cards[0].get("root")
    if not advertised or Path(advertised).expanduser().resolve() != model_path:
        raise EndpointMismatch(f"Endpoint checkpoint {advertised!r} does not match expected {str(model_path)!r}")
    return True


def validate_input(data_file, manifest_file, split, data_root=None, review=None):
    frozen = FrozenDataManifest(manifest_file, data_root or manifest_file.parent)
    if not frozen.data.get("manifest_id"):
        raise ValueError("Prepared split manifest must include its frozen manifest_id")
    seen = set()
    count = 0
    with data_file.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate MCTS input ID: {sample_id}")
            seen.add(sample_id)
            conversations = row.get("conversations", [])
            if len(conversations) != 2:
                raise ValueError(f"Expected one question and one gold label for {sample_id}")
            metadata = frozen.validate_sample(
                sample_id, row.get("image", ""), conversations[0].get("value", ""),
                conversations[1].get("value", ""), expected_split=split,
            )
            if metadata["dataset"] != "amber" or not sample_id.startswith("amber:"):
                raise ValueError("This training launcher accepts only AMBER; POPE and LongHalQA are held out")
            if review:
                if sample_id in review.excluded_ids:
                    continue
                review.validate_sample(sample_id)
            count += 1
    if not count:
        raise ValueError(f"MCTS input is empty: {data_file}")
    return count


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    # Popen(start_new_session=True) gives this child its own process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and command without starting processes")
    parser.add_argument("--no-start-server", action="store_true", default=os.environ.get("START_SERVER") == "0")
    parser.add_argument("--split", choices=("train", "dev", "test"), default=os.environ.get("DATA_SPLIT", "train"))
    parser.add_argument("--allow-held-out", action="store_true", help="Explicitly allow diagnostic rollouts on dev/test; never use them for training")
    parser.add_argument("--max-samples", type=int, default=os.environ.get("MAX_SAMPLES"))
    parser.add_argument("--processes", type=int, default=os.environ.get("NUM_PROCESSES", "4"))
    parser.add_argument("--simulations", type=int, default=os.environ.get("N_SIMULATIONS", "2"))
    parser.add_argument("--max-depth", type=int, default=os.environ.get("MAX_DEPTH", "10"))
    parser.add_argument("--rollout-max-steps", type=int, default=os.environ.get("ROLLOUT_MAX_STEPS", "5"))
    parser.add_argument("--seed", type=int, default=os.environ.get("SEED", "42"))
    parser.add_argument("--ready-timeout", type=float, default=os.environ.get("READY_TIMEOUT", "600"))
    args = parser.parse_args()
    if args.split != "train" and not args.allow_held_out:
        parser.error("dev/test are held out; use --allow-held-out only for diagnostics/evaluation")
    if os.environ.get("DATASET", "amber_discriminative") != "amber_discriminative":
        parser.error("Qwen3.8 training launcher supports AMBER discriminative; other benchmarks remain held out")
    for name in ("max_samples", "processes", "simulations", "max_depth", "rollout_max_steps", "ready_timeout"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    data_root = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
    prepared = data_root / "hallucination/prepared"
    data_file = Path(os.environ.get("DATA_FILE", prepared / f"amber/{args.split}.mcts.jsonl")).expanduser().resolve()
    manifest = Path(os.environ.get("SPLIT_MANIFEST", prepared / "manifest.json")).expanduser().resolve()
    review_path = (Path(os.environ.get("AMBER_REVIEW_OVERLAY", REPO_ROOT / "configs/data/amber_train_review_v1.json")).expanduser().resolve()
                   if args.split == "train" else None)
    model_path = Path(os.environ.get("QWEN38_MODEL_PATH", os.environ.get("MODEL", Path.home() / "models/Qwen3.8-27B"))).expanduser().resolve()
    port = os.environ.get("QWEN38_PORT", os.environ.get("PORT", "9011"))
    base_url = os.environ.get("QWEN_VLLM_BASE_URL", f"http://127.0.0.1:{port}/v1")
    api_key = os.environ.get("QWEN_VLLM_API_KEY", os.environ.get("QWEN38_API_KEY", "qwen"))
    model_name = os.environ.get("QWEN_VLLM_MODEL", os.environ.get("QWEN38_SERVED_MODEL_NAME", "qwen_vllm"))
    save_tag = os.environ.get("SAVE_TAG", f"MCTS_AMBER_QWEN38_27B_{args.split}")
    command = [
        sys.executable, "-m", "vlmsearch", "--model", "qwen_vllm", "--judge", "string_match",
        "--search_method", "mcts", "--seed", str(args.seed), "--use_python_mp",
        "--num_processes", str(args.processes), "--max_depth", str(args.max_depth),
        "--n_simulations", str(args.simulations), "--rollout_max_steps", str(args.rollout_max_steps),
        "--n_rollouts_per_node", os.environ.get("N_ROLLOUTS_PER_NODE", "2"),
        "--num_children_per_expand", os.environ.get("NUM_CHILDREN_PER_EXPAND", "2"),
        "--c_puct", os.environ.get("C_PUCT", "2.0"),
        "--temperature", os.environ.get("TEMPERATURE", "1.0"), "--top_p", os.environ.get("TOP_P", "1.0"),
        "--top_k", os.environ.get("TOP_K", "10"),
        "--max_new_tokens", os.environ.get("MAX_NEW_TOKENS", "512"),
        "--thought_token_begin", "<evidence>", "--thought_token_end", "</evidence>",
        "--checkpoint_interval", os.environ.get("CHECKPOINT_INTERVAL", "1"),
        "--save_rollouts", "--save_rollouts_dir", os.environ.get("SAVE_ROLLOUTS_DIR", "data/mcts"),
        "--system_prompt", SYSTEM_PROMPT, "--pretrained", str(model_path),
        "--image_root", str(data_root), "--data_files", str(data_file),
        "--split_manifest", str(manifest), "--data_split", args.split, "--save_tag", save_tag,
    ]
    if args.max_samples:
        command += ["--max_samples", str(args.max_samples)]
    if review_path:
        command += ["--review_overlay", str(review_path)]
    if args.dry_run:
        print(json.dumps({"data_file": str(data_file), "manifest": str(manifest), "split": args.split,
                          "review_overlay": str(review_path) if review_path else None,
                          "model_path": str(model_path), "endpoint": base_url, "command": command}, indent=2))
        return 0
    review = ReviewedTrainingData(review_path, FrozenDataManifest(manifest, data_root)) if review_path else None
    count = validate_input(data_file, manifest, args.split, data_root, review=review)
    print(f"Validated {count} eligible {args.split} questions against the frozen manifest and review", flush=True)
    environment = dict(os.environ, DATA_ROOT=str(data_root), QWEN38_MODEL_PATH=str(model_path),
                       QWEN38_PORT=port, PORT=port, QWEN38_API_KEY=api_key, QWEN_VLLM_API_KEY=api_key,
                       QWEN38_SERVED_MODEL_NAME=model_name, QWEN_VLLM_MODEL=model_name,
                       QWEN_VLLM_BASE_URL=base_url, QWEN38_ENABLE_THINKING="false")
    server = None
    rollout = None
    log_handle = None
    received_signal = signal.SIGINT
    def interrupted(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        raise KeyboardInterrupt
    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not verify_endpoint(base_url, api_key, model_name, model_path):
            if args.no_start_server:
                raise RuntimeError("Teacher is not ready and --no-start-server was requested")
            parsed = urllib.parse.urlparse(base_url)
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or str(parsed.port or 80) != port:
                raise RuntimeError("Automatic serving requires a loopback endpoint matching PORT")
            logs = REPO_ROOT / "vllm_logs"
            logs.mkdir(exist_ok=True)
            logfile = logs / f"qwen38_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
            log_handle = logfile.open("w")
            server = subprocess.Popen(["bash", "scripts/vllm/serve_qwen38.sh"], cwd=REPO_ROOT,
                                      env=environment, stdout=log_handle, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            print(f"Started owned teacher PID {server.pid}; logs: {logfile}", flush=True)
            deadline = time.monotonic() + args.ready_timeout
            while not verify_endpoint(base_url, api_key, model_name, model_path):
                if server.poll() is not None:
                    raise RuntimeError(f"Teacher exited with status {server.returncode}; inspect {logfile}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Teacher readiness timed out; inspect {logfile}")
                time.sleep(2)
        print(f"Verified Qwen3.8 checkpoint at {base_url}; running {save_tag}", flush=True)
        rollout = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, start_new_session=True)
        status = rollout.wait()
        return status if status >= 0 else 128 - status
    except KeyboardInterrupt:
        return 128 + received_signal
    finally:
        stop_owned_process(rollout)
        stop_owned_process(server)
        if log_handle:
            log_handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Qwen3.8 rollout failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
