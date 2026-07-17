import argparse
import base64
import html
import json
import os
import random
import re
from collections import Counter, defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None
from PIL import Image, ImageDraw


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)
COORD_RE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

AMBER_DISCRIMINATIVE_SYSTEM_PROMPT = """You are a careful visual verifier answering a yes/no multiple-choice question about an image.
Your goal is to determine whether the claim in the question is supported by visible evidence in the image.

You should reason step by step by checking relevant image regions and verifying evidence for the queried object, attribute, count, action, or relation.

Each reasoning step must be enclosed within think tags. When useful, reference one representative coordinate (x, y) for the region being checked. These coordinates are evidence anchors, not the final answer.

<think>
{Check one region and describe the visible evidence with a grounded point (x, y)}.
</think>

When you are ready to answer, output only:
<answer> yes </answer>
or
<answer> no </answer>

Rules:

- Generate only one reasoning step or the final answer per response.
- Base each step on visible evidence from the image.
- For existence questions, verify whether the queried object is actually visible and distinguish it from similar objects.
- For attribute questions, verify the specific property directly from the image rather than assuming it.
- For relation questions, verify both entities first, then verify the claimed relation.
- Use the checked visual evidence to choose the better-supported yes/no answer.
- Do not use world knowledge to infer details that are not visible.
- If you mention coordinates, avoid repeating the same point unless you are explicitly re-checking that evidence.
"""


def normalize_yes_no_answer(text: Any) -> Optional[str]:
    if text is None:
        return None

    answer = str(text).strip()
    answer_match = ANSWER_RE.search(answer)
    if answer_match:
        answer = answer_match.group(1)

    answer = answer.strip().lower()
    answer = re.sub(r"^[\s\[{(]+|[\s\]})]+$", "", answer)
    answer = re.sub(r"[\s.!,;:]+$", "", answer)
    if answer in {"yes", "no"}:
        return answer
    return None


def normalize_system_prompt(system_prompt: Optional[str]) -> str:
    prompt = system_prompt or AMBER_DISCRIMINATIVE_SYSTEM_PROMPT
    prompt = prompt.replace(
        "Use the checked visual evidence to choose the better-supported option letter.",
        "Use the checked visual evidence to choose the better-supported yes/no answer.",
    )
    prompt = prompt.replace("option letter", "yes/no answer")
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
    if not text or ANSWER_RE.search(text):
        return None

    think_blocks = [block.strip() for block in THINK_RE.findall(text) if block.strip()]
    if think_blocks:
        return "\n".join(think_blocks).strip()

    cleaned = strip_image_tags(text)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def normalize_rollout_chain(
    rollout: Dict[str, Any],
    question: str,
    require_coordinate: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    ephemeral_texts = rollout.get("ephemeral_texts") or []
    if not ephemeral_texts:
        return None, None, "missing_ephemeral_texts"

    answer = normalize_yes_no_answer(rollout.get("final_answer"))
    if answer is None:
        for text in reversed(ephemeral_texts):
            answer = normalize_yes_no_answer(text)
            if answer is not None:
                break
    if answer is None:
        return None, None, "invalid_final_answer"

    reasoning_steps = []
    for text in ephemeral_texts:
        if is_user_prompt_text(text, question):
            continue
        step = extract_reasoning_step(text)
        if step:
            reasoning_steps.append(step)

    reasoning = "\n".join(reasoning_steps).strip()
    if not reasoning:
        return None, None, "empty_think"
    if require_coordinate and not COORD_RE.search(reasoning):
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
        chain_text, final_answer, error = normalize_rollout_chain(
            rollout,
            question=question,
            require_coordinate=require_coordinate,
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
            }
        )
        stats["chains_kept"] += 1

    return chains, root_node_value, stats


def extract_think_points(chain_text: str):
    all_points = []
    point_idx = 1

    for block in THINK_RE.findall(chain_text):
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
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction by sample key.")
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

    sample_keys = sorted(
        {sample_key_for_row(data_file, line_idx, data_json, args.data_root) for data_file, line_idx, data_json in rows}
    )
    random.shuffle(sample_keys)
    val_count = int(len(sample_keys) * args.val_size)
    val_keys = set(sample_keys[:val_count])

    all_chains_text = []
    sft_entries_train = []
    sft_entries_val = []
    all_images_processed = set()
    chain_global_id = 0
    correct_count = 0
    report = {
        "data_str": args.data_str,
        "data_files": len(data_files),
        "rows_seen": len(rows),
        "sample_keys": len(sample_keys),
        "val_size": args.val_size,
        "train_sample_keys": len(sample_keys) - len(val_keys),
        "val_sample_keys": len(val_keys),
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
        system_prompt = normalize_system_prompt(data_json.get("system_prompt"))
        true_answer = normalize_yes_no_answer(data_json.get("true_answer", ""))
        if true_answer is None:
            report["filters"]["invalid_true_answer"] += 1
            continue

        sample_key = sample_key_for_row(data_file, line_idx, data_json, args.data_root)
        split = "val" if sample_key in val_keys else "train"
        chains, root_node_value, sample_stats = build_all_reasoning_for_sample(
            data_json,
            require_coordinate=require_coordinate,
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
                    "rollout_idx": chain["rollout_idx"],
                    "reward": chain["reward"],
                    "root_node_value": root_node_value,
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
            else:
                sft_entries_val.append(chain_entry)
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

    report["counts"]["train_entries"] = len(sft_entries_train)
    report["counts"]["val_entries"] = len(sft_entries_val)
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