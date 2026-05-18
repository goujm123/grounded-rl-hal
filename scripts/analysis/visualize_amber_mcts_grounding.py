#!/usr/bin/env python3
"""Extract and visualize coordinate-bearing thinking from AMBER MCTS rollouts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


COORD_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
WORKER_CKPT_RE = re.compile(r"worker(\d+)_ckpt(\d+)")


@dataclass(frozen=True)
class TextBlock:
    source: str
    node_path: str
    rollout_index: int | None
    text_index: int | None
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract thinking blocks and visualize grounded coordinates from AMBER MCTS rollout files."
    )
    parser.add_argument(
        "--mcts-dir",
        type=Path,
        default=Path("data/mcts/MCTS_AMBER_72b_20260424_103335"),
        help="Directory containing rollouts_*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <mcts-dir>/grounding_review_last10_mtime.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Number of final rollout files to inspect.",
    )
    parser.add_argument(
        "--selection",
        choices=("mtime", "worker_ckpt"),
        default="mtime",
        help="How to define the last rollout files. mtime follows wall-clock completion order.",
    )
    parser.add_argument(
        "--crop-radius",
        type=int,
        default=80,
        help="Half-size of point crops before resizing.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=320,
        help="Final square size of point crops.",
    )
    parser.add_argument(
        "--max-crops-per-sample",
        type=int,
        default=80,
        help="Maximum crops to write per sample.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy original referred images into the output directory for convenience.",
    )
    return parser.parse_args()


def rollout_sort_key(path: Path, selection: str) -> tuple[float, int, int, str] | tuple[int, int, str]:
    match = WORKER_CKPT_RE.search(path.name)
    worker = int(match.group(1)) if match else -1
    ckpt = int(match.group(2)) if match else -1
    if selection == "mtime":
        return (path.stat().st_mtime, worker, ckpt, path.name)
    return (worker, ckpt, path.name)


def select_rollout_files(mcts_dir: Path, last_n: int, selection: str) -> list[Path]:
    files = sorted(mcts_dir.glob("rollouts_*.jsonl"), key=lambda path: rollout_sort_key(path, selection))
    if not files:
        raise FileNotFoundError(f"No rollouts_*.jsonl files found under {mcts_dir}")
    return files[-last_n:]


def load_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSONL records found in {path}")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_question(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    if question:
        match = re.search(r"Question:\s*(.*?)\s*Answer Choices:", question, re.DOTALL)
        if match:
            return clean_text(match.group(1))
        return clean_text(question.replace("<image>", ""))
    root_text = str(sample.get("tree", {}).get("thought_text", ""))
    match = re.search(r"Question:\s*(.*?)\s*Answer Choices:", root_text, re.DOTALL)
    if match:
        return clean_text(match.group(1))
    return clean_text(root_text.replace("<image>", ""))


def iter_node_thought_blocks(node: dict[str, Any], node_path: str = "root") -> Iterable[TextBlock]:
    if node_path != "root":
        yield from split_thinking_blocks(str(node.get("thought_text", "")), source="best_path", node_path=node_path)


def select_best_child(children: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not children:
        return None
    return max(
        children,
        key=lambda child: (
            float(child.get("value", 0.0)),
            int(child.get("visit_count", 0)),
            len(child.get("rollouts", [])),
        ),
    )


def best_path_blocks(root: dict[str, Any]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    node = root
    node_path = "root"
    while True:
        children = node.get("children", [])
        child = select_best_child(children)
        if child is None:
            break
        child_index = children.index(child)
        node_path = f"{node_path}.{child_index}"
        blocks.extend(iter_node_thought_blocks(child, node_path))
        node = child
    return blocks


def iter_tree_blocks(node: dict[str, Any], node_path: str = "root") -> Iterable[TextBlock]:
    thought_text = str(node.get("thought_text", ""))
    if node_path != "root":
        yield from split_thinking_blocks(thought_text, source="node", node_path=node_path)

    for rollout_index, rollout in enumerate(node.get("rollouts", [])):
        for text_index, text in enumerate(rollout.get("ephemeral_texts", [])):
            yield from split_thinking_blocks(
                str(text),
                source="rollout",
                node_path=node_path,
                rollout_index=rollout_index,
                text_index=text_index,
            )

    for child_index, child in enumerate(node.get("children", [])):
        child_path = f"{node_path}.{child_index}"
        yield from iter_tree_blocks(child, child_path)


def split_thinking_blocks(
    text: str,
    source: str,
    node_path: str,
    rollout_index: int | None = None,
    text_index: int | None = None,
) -> Iterable[TextBlock]:
    matches = THINK_RE.findall(text)
    if matches:
        for match in matches:
            yield TextBlock(source, node_path, rollout_index, text_index, clean_text(match))
        return

    answer_only = ANSWER_RE.search(text) and not COORD_RE.search(text)
    if text.strip() and not answer_only:
        yield TextBlock(source, node_path, rollout_index, text_index, clean_text(text))


def ensure_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def unique_coordinate_key(x: float, y: float) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def draw_overlay(image: Image.Image, coords: list[dict[str, Any]], output_path: Path) -> None:
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    width, height = image.size
    seen: set[tuple[int, int]] = set()
    label_index = 0

    for coord in coords:
        x, y = float(coord["x"]), float(coord["y"])
        key = unique_coordinate_key(x, y)
        if key in seen:
            continue
        seen.add(key)
        if not (0 <= x <= width and 0 <= y <= height):
            continue
        label_index += 1
        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="white", width=3)
        label = str(label_index)
        bbox = draw.textbbox((x + 10, y - 10), label, font=font)
        pad = 3
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill="white",
            outline="red",
        )
        draw.text((x + 10, y - 10), label, fill="red", font=font)

    image.save(output_path)


def write_crop(
    image: Image.Image,
    x: float,
    y: float,
    crop_radius: int,
    crop_size: int,
    output_path: Path,
) -> bool:
    width, height = image.size
    if not (0 <= x <= width and 0 <= y <= height):
        return False
    xi, yi = int(round(x)), int(round(y))
    left = max(0, xi - crop_radius)
    top = max(0, yi - crop_radius)
    right = min(width, xi + crop_radius)
    bottom = min(height, yi + crop_radius)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    local_x = xi - left
    local_y = yi - top
    radius = 7
    draw.ellipse(
        (local_x - radius, local_y - radius, local_x + radius, local_y + radius),
        fill="red",
        outline="white",
        width=2,
    )
    crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    crop.save(output_path)
    return True


def write_sample_markdown(sample_records: list[dict[str, Any]], output_path: Path) -> None:
    lines = ["# Extracted Thinking And Coordinates", ""]
    for record in sample_records:
        lines.append(
            f"## Sample {record['sample_rank']}: id={record['sample_id']} true={record['true_answer']}"
        )
        lines.append("")
        lines.append(f"- Rollout file: `{record['rollout_file']}`")
        lines.append(f"- Image: `{record['image']}`")
        lines.append(f"- Question: {record['question']}")
        lines.append(f"- Overlay: `{record['overlay']}`")
        lines.append(f"- Best-path overlay: `{record['best_path_overlay']}`")
        if record.get("original_copy"):
            lines.append(f"- Original copy: `{record['original_copy']}`")
        lines.append("")
        lines.append("### Best Path")
        lines.append("")
        for coord in record["best_path_coordinates"]:
            status = "in bounds" if coord["in_bounds"] else "OUT OF BOUNDS"
            lines.append(
                f"{coord['coord_number']}. ({coord['x']:.1f}, {coord['y']:.1f}) - {status} - "
                f"{coord['node_path']}"
            )
            lines.append(f"   Text: {coord['text']}")
            if coord.get("crop"):
                lines.append(f"   Crop: `{coord['crop']}`")
            lines.append("")
        if not record["best_path_coordinates"]:
            lines.append("No coordinate-bearing best-path thinking blocks were found.")
            lines.append("")
        lines.append("### All Coordinate Mentions")
        lines.append("")
        for coord in record["coordinates"]:
            crop = coord.get("crop") or ""
            status = "in bounds" if coord["in_bounds"] else "OUT OF BOUNDS"
            lines.append(
                f"{coord['coord_number']}. ({coord['x']:.1f}, {coord['y']:.1f}) - {status} - "
                f"{coord['source']} {coord['node_path']}"
            )
            lines.append(f"   Text: {coord['text']}")
            if crop:
                lines.append(f"   Crop: `{crop}`")
            lines.append("")
        if not record["coordinates"]:
            lines.append("No coordinate-bearing thinking blocks were found.")
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    mcts_dir = args.mcts_dir.resolve()
    output_dir = (args.output_dir or (mcts_dir / f"grounding_review_last{args.last_n}_{args.selection}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    best_overlays_dir = output_dir / "best_path_overlays"
    crops_dir = output_dir / "crops"
    best_crops_dir = output_dir / "best_path_crops"
    originals_dir = output_dir / "original_images"
    overlays_dir.mkdir(exist_ok=True)
    best_overlays_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    best_crops_dir.mkdir(exist_ok=True)
    if args.copy_images:
        originals_dir.mkdir(exist_ok=True)

    rollout_files = select_rollout_files(mcts_dir, args.last_n, args.selection)
    all_coord_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []

    for sample_rank, rollout_path in enumerate(rollout_files, start=1):
        sample = load_first_jsonl(rollout_path)
        sample_id = str(sample.get("id", rollout_path.stem))
        image_path = Path(str(sample["image"])).resolve()
        image = ensure_rgb_image(image_path)
        width, height = image.size
        sample_slug = f"{sample_rank:02d}_id_{sample_id}"
        sample_crop_dir = crops_dir / sample_slug
        best_sample_crop_dir = best_crops_dir / sample_slug
        sample_crop_dir.mkdir(exist_ok=True)
        best_sample_crop_dir.mkdir(exist_ok=True)

        coord_records: list[dict[str, Any]] = []
        coord_number = 0
        for block in iter_tree_blocks(sample["tree"]):
            for match in COORD_RE.finditer(block.text):
                coord_number += 1
                x = float(match.group(1))
                y = float(match.group(2))
                in_bounds = 0 <= x <= width and 0 <= y <= height
                crop_rel = None
                if coord_number <= args.max_crops_per_sample:
                    crop_path = sample_crop_dir / f"coord_{coord_number:03d}_{int(round(x))}_{int(round(y))}.png"
                    if write_crop(image, x, y, args.crop_radius, args.crop_size, crop_path):
                        crop_rel = crop_path.relative_to(output_dir).as_posix()
                record = {
                    "sample_rank": sample_rank,
                    "sample_id": sample_id,
                    "rollout_file": rollout_path.name,
                    "image": str(image_path),
                    "question": extract_question(sample),
                    "true_answer": str(sample.get("true_answer", "")),
                    "source": block.source,
                    "node_path": block.node_path,
                    "rollout_index": block.rollout_index,
                    "text_index": block.text_index,
                    "coord_number": coord_number,
                    "x": x,
                    "y": y,
                    "image_width": width,
                    "image_height": height,
                    "in_bounds": in_bounds,
                    "text": block.text,
                    "crop": crop_rel,
                }
                coord_records.append(record)
                all_coord_records.append(record)

        best_coord_records: list[dict[str, Any]] = []
        for best_index, block in enumerate(best_path_blocks(sample["tree"]), start=1):
            for match in COORD_RE.finditer(block.text):
                x = float(match.group(1))
                y = float(match.group(2))
                in_bounds = 0 <= x <= width and 0 <= y <= height
                crop_rel = None
                crop_path = best_sample_crop_dir / f"coord_{best_index:03d}_{int(round(x))}_{int(round(y))}.png"
                if write_crop(image, x, y, args.crop_radius, args.crop_size, crop_path):
                    crop_rel = crop_path.relative_to(output_dir).as_posix()
                best_coord_records.append(
                    {
                        "sample_rank": sample_rank,
                        "sample_id": sample_id,
                        "rollout_file": rollout_path.name,
                        "image": str(image_path),
                        "question": extract_question(sample),
                        "true_answer": str(sample.get("true_answer", "")),
                        "source": block.source,
                        "node_path": block.node_path,
                        "rollout_index": block.rollout_index,
                        "text_index": block.text_index,
                        "coord_number": len(best_coord_records) + 1,
                        "x": x,
                        "y": y,
                        "image_width": width,
                        "image_height": height,
                        "in_bounds": in_bounds,
                        "text": block.text,
                        "crop": crop_rel,
                    }
                )

        overlay_path = overlays_dir / f"{sample_slug}_overlay.png"
        draw_overlay(image.copy(), coord_records, overlay_path)
        best_overlay_path = best_overlays_dir / f"{sample_slug}_best_path_overlay.png"
        draw_overlay(image.copy(), best_coord_records, best_overlay_path)
        original_copy_rel = None
        if args.copy_images:
            original_copy = originals_dir / f"{sample_slug}{image_path.suffix.lower()}"
            shutil.copy2(image_path, original_copy)
            original_copy_rel = original_copy.relative_to(output_dir).as_posix()

        sample_record = {
            "sample_rank": sample_rank,
            "sample_id": sample_id,
            "rollout_file": rollout_path.name,
            "image": str(image_path),
            "original_copy": original_copy_rel,
            "question": extract_question(sample),
            "true_answer": str(sample.get("true_answer", "")),
            "image_width": width,
            "image_height": height,
            "num_coordinate_mentions": len(coord_records),
            "num_out_of_bounds": sum(1 for record in coord_records if not record["in_bounds"]),
            "overlay": overlay_path.relative_to(output_dir).as_posix(),
            "best_path_overlay": best_overlay_path.relative_to(output_dir).as_posix(),
            "best_path_coordinates": best_coord_records,
            "coordinates": coord_records,
        }
        sample_records.append(sample_record)

    coord_jsonl = output_dir / "extracted_thinking_coordinates.jsonl"
    with coord_jsonl.open("w", encoding="utf-8") as handle:
        for record in all_coord_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    best_coord_jsonl = output_dir / "best_path_thinking_coordinates.jsonl"
    with best_coord_jsonl.open("w", encoding="utf-8") as handle:
        for sample_record in sample_records:
            for record in sample_record["best_path_coordinates"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "sample_summary.json"
    summary = {
        "mcts_dir": str(mcts_dir),
        "selection": args.selection,
        "last_n": args.last_n,
        "num_samples": len(sample_records),
        "num_coordinate_mentions": len(all_coord_records),
        "num_out_of_bounds": sum(1 for record in all_coord_records if not record["in_bounds"]),
        "num_best_path_coordinate_mentions": sum(
            len(sample_record["best_path_coordinates"]) for sample_record in sample_records
        ),
        "rollout_files": [path.name for path in rollout_files],
        "samples": sample_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sample_markdown(sample_records, output_dir / "extracted_thinking_coordinates.md")

    print(f"Wrote {len(sample_records)} samples to {output_dir}")
    print(f"Coordinate mentions: {len(all_coord_records)}")
    print(f"Out-of-bounds mentions: {summary['num_out_of_bounds']}")
    print(f"JSONL: {coord_jsonl}")
    print(f"Best-path JSONL: {best_coord_jsonl}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()