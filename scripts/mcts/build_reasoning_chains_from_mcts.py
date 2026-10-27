import argparse
import base64
import html
import json
import os
import random
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None
from PIL import Image, ImageDraw
from vlmsearch.data_manifest import FrozenDataManifest
from vlmsearch.data_review import ReviewedTrainingData
from vlmsearch.reward_funcs.answers import parse_final_answer
from vlmsearch.image_geometry import (
    CoordinateError, PIXEL_COORDINATE_INSTRUCTION,
    convert_coordinate_text,
)


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<(think|evidence)>\s*(.*?)\s*</\1>", flags=re.IGNORECASE | re.DOTALL)
COORD_RE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

STUDENT_PROMPT_VERSION = "amber_grounded_full_response_v1"
AMBER_DISCRIMINATIVE_SYSTEM_PROMPT = """You are a careful visual verifier answering a yes/no question about an image.
Determine whether the claim is supported by visible evidence. Give one complete response containing exactly one <think> block followed by exactly one <answer> block, with no text outside those blocks:
<think>Brief observations that support the answer.</think>
<answer>yes</answer>
The answer must be yes or no.

Check the queried object, attribute, count, action, or relation directly in the image. For relations, check both entities before their relation. Describe only visible evidence; do not infer hidden details or invent objects. Keep observations concise and avoid repetition.

When a location helps, include a representative point (x, y) for the visible region being checked. Coordinates are evidence anchors, not proof of a claim. For an absent object, describe the checked regions or the image-wide evidence; coordinates may be omitted. Never invent a point for an absent object.
""" + "\n" + PIXEL_COORDINATE_INSTRUCTION


def normalize_yes_no_answer(text: Any) -> Optional[str]:
    return parse_final_answer(text, {"yes", "no"})


def normalize_system_prompt(system_prompt: Optional[str], original_size=None) -> str:
    # The teacher prompt controls one MCTS generation step. A student target is
    # a complete trace, so copying the teacher's one-block/previous-evidence
    # instructions would contradict every target. Retain source prompts in the
    # original rollout record and use this versioned AMBER student contract.
    prompt = AMBER_DISCRIMINATIVE_SYSTEM_PROMPT
    if original_size is not None:
        width, height = original_size
        prompt += f"\nThe original image is {width} x {height} pixels; valid points satisfy 0 <= x <= {width - 1} and 0 <= y <= {height - 1}."
    return prompt


def strip_image_tags(text: str) -> str:
    return text.replace("<image>", "").replace("</image>", "").strip()


def is_user_prompt_text(text: str, question: str) -> bool:
    stripped_text = strip_image_tags(text)
    stripped_question = strip_image_tags(question)
    if stripped_text == stripped_question:
        return True
    return text.strip().startswith("<image>") and "Question:" in text


def extract_reasoning_step(text: str) -> Optional[str]:
    if not text:
        return None

    think_blocks = [match.group(2).strip() for match in THINK_RE.finditer(text) if match.group(2).strip()]
    if think_blocks:
        return "\n".join(think_blocks).strip()
    if ANSWER_RE.search(text) or normalize_yes_no_answer(text) is not None:
        return None
    if re.search(r"</?(?:think|evidence)\b", text, flags=re.IGNORECASE):
        return None

    cleaned = strip_image_tags(text)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def normalize_rollout_chain(
    rollout: Dict[str, Any],
    question: str,
    require_coordinate: bool = True,
    image_geometry=None,
    original_size=None,
    legacy_coordinate_frame=None,
    coordinate_audit=None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    ephemeral_texts = rollout.get("ephemeral_texts") or []
    if not ephemeral_texts:
        return None, None, "missing_ephemeral_texts"

    raw_answer = rollout.get("raw_final_answer")
    if raw_answer is not None and normalize_yes_no_answer(raw_answer) is None:
        return None, None, "invalid_raw_final_answer"
    answer = normalize_yes_no_answer(raw_answer or rollout.get("final_answer"))
    if answer is None:
        for text in reversed(ephemeral_texts):
            answer = normalize_yes_no_answer(text)
            if answer is not None:
                break
    if answer is None:
        return None, None, "invalid_final_answer"

    reasoning_steps = []
    reasoning_texts = list(ephemeral_texts)
    if raw_answer and raw_answer not in reasoning_texts:
        reasoning_texts.append(raw_answer)
    for text in reasoning_texts:
        if is_user_prompt_text(text, question):
            continue
        step = extract_reasoning_step(text)
        if step:
            reasoning_steps.append(step)

    reasoning = "\n".join(reasoning_steps).strip()
    if not reasoning:
        return None, None, "empty_think"
    try:
        reasoning, mapping = convert_coordinate_text(
            reasoning, geometry=image_geometry, original_size=original_size,
            legacy_coordinate_frame=legacy_coordinate_frame,
        )
    except CoordinateError as exc:
        return None, None, str(exc)
    if coordinate_audit is not None:
        coordinate_audit["mapping"] = mapping
    if require_coordinate and not mapping:
        return None, None, "missing_coordinate"

    chain_text = f"<think>\n{reasoning}\n</think>\n<answer> {answer} </answer>"
    if chain_text.count("<answer>") != 1 or chain_text.count("</answer>") != 1:
        return None, None, "nested_answer"
    if "<image>" in chain_text or "</image>" in chain_text:
        return None, None, "assistant_contains_image"

    return chain_text, answer, None


def iter_tree_rollouts(node_data: Dict[str, Any]):
    for rollout in node_data.get("rollouts", []):
        yield rollout
    for child in node_data.get("children", []):
        yield from iter_tree_rollouts(child)


def build_all_reasoning_for_sample(
    sample_json: Dict[str, Any],
    require_coordinate: bool = True,
    original_size=None,
    legacy_coordinate_frame=None,
) -> Tuple[List[Dict[str, Any]], float, Counter]:
    root_node = sample_json.get("tree")
    if not root_node:
        return [], 0.0, Counter({"missing_tree": 1})

    question = sample_json.get("question", "")
    root_node_value = root_node.get("value", 0.0)
    chains = []
    seen = set()
    stats = Counter()

    for rollout_idx, rollout in enumerate(iter_tree_rollouts(root_node)):
        stats["rollouts_seen"] += 1
        if rollout.get("reward", 0.0) < 1.0:
            stats["skipped_non_positive_reward"] += 1
            continue

        stats["correct_rollouts_seen"] += 1
        coordinate_audit = {}
        chain_text, final_answer, error = normalize_rollout_chain(
            rollout,
            question=question,
            require_coordinate=require_coordinate,
            image_geometry=sample_json.get("image_geometry"),
            original_size=original_size,
            legacy_coordinate_frame=legacy_coordinate_frame,
            coordinate_audit=coordinate_audit,
        )
        if error:
            stats[f"filtered_{error}"] += 1
            continue
        if chain_text in seen:
            stats["filtered_duplicate_chain"] += 1
            continue

        seen.add(chain_text)
        chains.append(
            {
                "chain": chain_text,
                "final_answer": final_answer,
                "rollout_idx": rollout_idx,
                "reward": rollout.get("reward", 0.0),
                "ephemeral_depth": rollout.get("ephemeral_depth"),
                "depth": rollout.get("depth"),
                "coordinate_mapping": coordinate_audit.get("mapping", []),
            }
        )
        stats["chains_kept"] += 1

    return chains, root_node_value, stats


def extract_think_points(chain_text: str):
    all_points = []
    point_idx = 1

    for match in THINK_RE.finditer(chain_text):
        block = match.group(2)
        for x_str, y_str in COORD_RE.findall(block):
            try:
                x = float(x_str)
                y = float(y_str)
                all_points.append((x, y, point_idx))
                point_idx += 1
            except ValueError:
                continue

    return all_points


def parse_args():
    parser = argparse.ArgumentParser(description="Linearize AMBER MCTS trees into ShareGPT SFT data.")
    parser.add_argument(
        "--data-str",
        default="data/mcts/MCTS_AMBER_DISCRIMINATIVE_72b_20260708_223000_full",
        help="Path to a rollout JSONL file or a directory containing rollout JSONL files.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for generated reasoning chain files.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Legacy fallback validation fraction by image.")
    parser.add_argument("--split-manifest", default=os.environ.get("DATA_SPLIT_MANIFEST"), help="Use the frozen train/dev/test assignments; auto-detected in new rollout directories")
    parser.add_argument("--review-overlay", default=None, help="Train-only annotation review; auto-detected as training_review.json in rollout directories")
    parser.add_argument("--max-samples-per-file", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-to-wandb", action="store_true", help="Log a small HTML preview to W&B.")
    parser.add_argument("--draw-points", action="store_true", help="Draw extracted reasoning points in the W&B preview.")
    parser.add_argument(
        "--allow-no-coordinate",
        action="store_true",
        help="Keep chains whose <think> block has no coordinate evidence.",
    )
    parser.add_argument(
        "--legacy-coordinate-frame", choices=("original_pixels",), default=None,
        help="Explicitly declare coordinate-bearing old traces as original-image pixels; validate bounds. Never inferred from values.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "data"),
        help="Data root used to normalize image paths.",
    )
    return parser.parse_args()


def natural_sort_key(path: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def discover_data_files(data_str: str) -> List[str]:
    if os.path.isdir(data_str):
        data_files = [os.path.join(data_str, f) for f in os.listdir(data_str) if f.endswith(".jsonl")]
    else:
        data_files = [data_str]
    return sorted(data_files, key=natural_sort_key)


def iter_jsonl_rows(data_files: List[str]):
    for data_file in data_files:
        with open(data_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield data_file, line_idx, json.loads(line)
                except json.JSONDecodeError as exc:
                    yield data_file, line_idx, {"error": f"json_decode_error: {exc}"}


def normalize_image_path(image_path: str, data_root: Optional[str] = None) -> str:
    if not image_path:
        return image_path

    normalized = image_path.replace("\\", "/")
    if "/mllm_hal/" in normalized:
        return "mllm_hal/" + normalized.split("/mllm_hal/", 1)[1]

    data_root = data_root or os.environ.get("DATA_ROOT") or "data"
    try:
        rel_path = os.path.relpath(normalized, data_root)
        if not rel_path.startswith(".."):
            return rel_path.replace("\\", "/")
    except ValueError:
        pass

    if "/data/" in normalized:
        return normalized.split("/data/", 1)[1]
    return normalized


def resolve_image_path(image_path: str, data_root: str) -> str:
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    candidate = os.path.join(data_root, image_path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(image_path):
        return image_path
    return candidate


def sample_key_for_row(data_file: str, line_idx: int, data_json: Dict[str, Any], data_root: str) -> str:
    if data_json.get("id"):
        return str(data_json["id"])
    question = data_json.get("question", "")
    image = normalize_image_path(data_json.get("image", ""), data_root)
    if question or image:
        return f"{image}\n{question}"
    return f"{data_file}:{line_idx}"


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def maybe_log_to_wandb(entries: List[Dict[str, Any]], data_root: str, draw_points: bool) -> None:
    if wandb is None:
        raise ImportError("wandb is required for --log-to-wandb but is not installed.")
    if not entries:
        print("No SFT entries available for W&B logging.")
        return

    wandb.init(project="vlm-search", name="amber_mcts_reasoning_chains_sft")
    max_samples = 50
    random_indices = random.sample(range(len(entries)), min(max_samples, len(entries)))
    html_content = "<html><body>"

    for idx in random_indices:
        entry = entries[idx]
        question = entry["messages"][1]["content"]
        chain_text = entry["messages"][2]["content"]
        image_path = entry["images"][0]
        true_answer = entry["gt_answer"]
        final_answer = ""
        answer_match = ANSWER_RE.search(chain_text)
        if answer_match:
            final_answer = answer_match.group(1).strip()

        img_html = ""
        resolved_image_path = resolve_image_path(image_path, data_root)
        if image_path and os.path.exists(resolved_image_path):
            with Image.open(resolved_image_path) as pil_img:
                if draw_points:
                    draw = ImageDraw.Draw(pil_img)
                    circle_radius = 7
                    for x, y, point_num in extract_think_points(chain_text):
                        draw.ellipse(
                            (x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius),
                            fill="green",
                        )
                        draw.text((x + 15, y), str(point_num), fill="green")

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                enc = base64.b64encode(buf.getvalue()).decode("utf-8")
                img_html = f'<img src="data:image/png;base64,{enc}" style="max-width:600px;" />'

        row_html = f"""
        <div style="border:1px solid #ddd; padding:10px; margin:10px 0;">
            <p><b>Question:</b> {html.escape(question)}</p>
            {img_html}
            <p><b>Prediction:</b> {html.escape(final_answer)} &nbsp; <b>GT:</b> {html.escape(true_answer)}</p>
            <p><b>Chain:</b> {html.escape(chain_text)}</p>
        </div>
        """
        html_content += row_html

    html_content += "</body></html>"
    wandb.log({"mcts_reasoning_chains": wandb.Html(html_content)})
    wandb.finish()


def main():
    args = parse_args()
    random.seed(args.seed)
    require_coordinate = not args.allow_no_coordinate
    data_files = discover_data_files(args.data_str)
    rows = list(iter_jsonl_rows(data_files))

    manifest_path = args.split_manifest
    if not manifest_path:
        candidates = {os.path.join(os.path.dirname(path), "split_manifest.json") for path in data_files}
        existing = sorted(path for path in candidates if os.path.isfile(path))
        if existing:
            fingerprints = {Path(path).read_bytes() for path in existing}
            if len(fingerprints) != 1:
                raise ValueError("Rollout directories contain different frozen split manifests")
            manifest_path = existing[0]
    manifest = FrozenDataManifest(manifest_path, args.data_root) if manifest_path else None
    if not manifest and any(row.get("split_manifest_sha256") for _, _, row in rows):
        raise ValueError("These rollouts require their frozen --split-manifest")
    review_path = args.review_overlay
    if not review_path:
        existing = sorted({os.path.join(os.path.dirname(path), "training_review.json") for path in data_files
                           if os.path.isfile(os.path.join(os.path.dirname(path), "training_review.json"))})
        if existing:
            if len({Path(path).read_bytes() for path in existing}) != 1:
                raise ValueError("Rollout directories contain different training review overlays")
            review_path = existing[0]
    if review_path and manifest is None:
        raise ValueError("Training review requires a frozen split manifest")
    review = ReviewedTrainingData(review_path, manifest) if review_path else None
    if not review and any(row.get("review_overlay_id") or row.get("review_overlay_sha256") for _, _, row in rows):
        raise ValueError("These rollouts require their --review-overlay / training_review.json")

    sample_keys = sorted(
        {sample_key_for_row(data_file, line_idx, data_json, args.data_root) for data_file, line_idx, data_json in rows}
    )
    image_keys = sorted({normalize_image_path(row.get("image", ""), args.data_root) for _, _, row in rows if not row.get("error")})
    random.shuffle(image_keys)
    val_images = set(image_keys[:int(len(image_keys) * args.val_size)])
    if manifest:
        val_keys = {key for key in sample_keys if key in manifest.samples and manifest.samples[key]["split"] == "dev"}
    else:
        val_keys = {sample_key_for_row(path, line, row, args.data_root) for path, line, row in rows if normalize_image_path(row.get("image", ""), args.data_root) in val_images}
    test_keys = {key for key in sample_keys if manifest and key in manifest.samples and manifest.samples[key]["split"] == "test"}

    all_chains_text = []
    sft_entries_train = []
    sft_entries_val = []
    sft_entries_test = []
    all_images_processed = set()
    chain_global_id = 0
    correct_count = 0
    report = {
        "data_str": args.data_str,
        "data_files": len(data_files),
        "rows_seen": len(rows),
        "sample_keys": len(sample_keys),
        "val_size": args.val_size,
        "train_sample_keys": len(sample_keys) - len(val_keys) - len(test_keys),
        "val_sample_keys": len(val_keys),
        "test_sample_keys": len(test_keys),
        "split_policy": "frozen_manifest" if manifest else "legacy_image_grouped",
        "split_manifest_sha256": manifest.sha256 if manifest else None,
        "review_overlay_id": review.id if review else None,
        "review_overlay_sha256": review.sha256 if review else None,
        "legacy_coordinate_frame": args.legacy_coordinate_frame,
        "student_prompt_version": STUDENT_PROMPT_VERSION,
        "counts": Counter(),
        "filters": Counter(),
        "per_file_rows": defaultdict(int),
    }

    for row_idx, (data_file, line_idx, data_json) in enumerate(rows):
        report["per_file_rows"][data_file] += 1
        error = data_json.get("error", "")
        if error:
            report["counts"]["error_rows"] += 1
            print(f"error processing {data_file}:{line_idx}: {error}")
            continue

        question = data_json.get("question", "")
        image = normalize_image_path(data_json.get("image", ""), args.data_root)
        resolved_image_path = resolve_image_path(image, args.data_root)
        with Image.open(resolved_image_path) as original_image:
            original_size = original_image.size
        system_prompt = normalize_system_prompt(data_json.get("system_prompt"), original_size)
        true_answer = normalize_yes_no_answer(data_json.get("true_answer", ""))
        if true_answer is None:
            report["filters"]["invalid_true_answer"] += 1
            continue

        sample_key = sample_key_for_row(data_file, line_idx, data_json, args.data_root)
        manifest_metadata = manifest.validate_rollout(data_json) if manifest else {}
        if review:
            review_metadata = review.validate_sample(sample_key)
            for key, expected in review_metadata.items():
                if data_json.get(key) != expected:
                    raise ValueError(f"Rollout {sample_key} has incompatible {key}")
            manifest_metadata.update(review_metadata)
        split = manifest_metadata["data_split"] if manifest else ("dev" if sample_key in val_keys else "train")
        chains, root_node_value, sample_stats = build_all_reasoning_for_sample(
            data_json,
            require_coordinate=require_coordinate,
            original_size=original_size,
            legacy_coordinate_frame=args.legacy_coordinate_frame,
        )
        report["counts"].update(sample_stats)
        if root_node_value > 0:
            correct_count += 1

        if len(chains) > args.max_samples_per_file:
            report["counts"]["sampled_down_chains"] += len(chains) - args.max_samples_per_file
            chains = random.sample(chains, args.max_samples_per_file)

        for chain in chains:
            chain_text = chain["chain"]
            final_answer = chain["final_answer"]
            if final_answer != true_answer:
                report["filters"]["final_answer_mismatches_true_answer"] += 1
                continue

            all_chains_text.append(chain_text)
            chain_entry = {
                "id": f"{row_idx}_{chain_global_id}",
                "metadata": {
                    "source_file": data_file,
                    "source_line": line_idx,
                    "sample_key": sample_key,
                    "split": split,
                    **manifest_metadata,
                    "rollout_idx": chain["rollout_idx"],
                    "reward": chain["reward"],
                    "root_node_value": root_node_value,
                    "student_prompt_version": STUDENT_PROMPT_VERSION,
                    "coordinate_frame": "original_pixels",
                    "image_geometry": data_json.get("image_geometry"),
                    "source_image_geometry": data_json.get("image_geometry"),
                    "coordinate_mapping": chain["coordinate_mapping"],
                    "original_image_size": list(original_size),
                    "legacy_coordinate_frame": args.legacy_coordinate_frame if not data_json.get("image_geometry") else None,
                    **{key: data_json[key] for key in ("review_overlay_id", "review_overlay_sha256", "pilot_selection_id") if key in data_json},
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": chain_text},
                ],
                "images": [image],
                "gt_answer": true_answer,
            }
            if split == "train":
                sft_entries_train.append(chain_entry)
            elif split == "dev":
                sft_entries_val.append(chain_entry)
            else:
                sft_entries_test.append(chain_entry)
            all_images_processed.add(image)
            chain_global_id += 1

    total_samples = len(rows) - report["counts"]["error_rows"]
    if total_samples == 0:
        print("No samples found. Exiting.")
        return

    accuracy = correct_count / total_samples
    print(f"Processed {total_samples} input JSONL rows from {len(data_files)} files.")
    print(f"Root node correctness ratio: {accuracy:.3f} ({correct_count}/{total_samples})")

    base_dir = os.path.dirname(data_files[0]) if data_files else os.path.dirname(args.data_str)
    out_dir = args.output_dir or os.path.join(base_dir, "reasoning_chains")
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "reasoning_chains.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        sep = "\n\n----------------------\n\n----------------------\n\n"
        for chain_text in all_chains_text:
            f.write(chain_text + sep)
    print(f"Wrote {len(all_chains_text)} total chains to {txt_path}")

    train_json_path = os.path.join(out_dir, "reasoning_chains_train.json")
    val_json_path = os.path.join(out_dir, "reasoning_chains_val.json")
    write_json(train_json_path, sft_entries_train)
    print(f"Saved train SFT data to: {train_json_path}  (count={len(sft_entries_train)})")
    write_json(val_json_path, sft_entries_val)
    print(f"Saved val SFT data to: {val_json_path}  (count={len(sft_entries_val)})")
    test_json_path = os.path.join(out_dir, "reasoning_chains_test.json")
    write_json(test_json_path, sft_entries_test)

    report["counts"]["train_entries"] = len(sft_entries_train)
    report["counts"]["val_entries"] = len(sft_entries_val)
    report["counts"]["test_entries"] = len(sft_entries_test)
    report["counts"]["unique_images"] = len(all_images_processed)
    report["counts"] = dict(report["counts"])
    report["filters"] = dict(report["filters"])
    report["per_file_rows"] = dict(report["per_file_rows"])
    report_path = os.path.join(out_dir, "linearization_report.json")
    write_json(report_path, report)
    print(f"Saved linearization report to: {report_path}")

    if args.log_to_wandb:
        maybe_log_to_wandb(sft_entries_train + sft_entries_val, args.data_root, args.draw_points)


if __name__ == "__main__":
    main()
