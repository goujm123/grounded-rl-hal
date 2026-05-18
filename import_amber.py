#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert AMBER discriminative annotations into ViGoRL MCTS jsonl format."
	)
	parser.add_argument(
		"--amber-root",
		type=Path,
		default=Path("data/mllm_hal"),
		help="Root directory containing AMBER images, query files, and annotations.",
	)
	parser.add_argument(
		"--query-file",
		type=Path,
		default=None,
		help=(
			"Optional query json file. Defaults to query/query_discriminative.json under --amber-root."
		),
	)
	parser.add_argument(
		"--annotations-file",
		type=Path,
		default=None,
		help="Optional annotations json file. Defaults to annotations.json under --amber-root.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help=(
			"Output jsonl path. Defaults to amber_discriminative_MCTS.jsonl under --amber-root."
		),
	)
	parser.add_argument(
		"--image-prefix",
		default=None,
		help=(
			"Image path prefix to store in each record. Defaults to '<amber-root-name>/images', "
			"which matches the repo's DATA_ROOT-relative convention."
		),
	)
	parser.add_argument(
		"--question-prefix",
		default="Question:",
		help="Prefix placed before each AMBER query in the serialized user prompt.",
	)
	parser.add_argument(
		"--strict",
		action="store_true",
		help="Fail instead of skipping rows with missing annotations or images.",
	)
	return parser.parse_args()


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def build_prompt(query: str, question_prefix: str) -> str:
	return (
		f"<image>\n{question_prefix} {query}\n"
		"Answer Choices:\n"
		"A. yes\n"
		"B. no\n"
		"Answer with the option letter only."
	)


def truth_to_option(truth: str) -> str:
	if truth == "yes":
		return "A"
	if truth == "no":
		return "B"
	raise ValueError(f"Unsupported truth label: {truth!r}")


def normalize_image_path(image_name: str, image_prefix: str) -> str:
	return str(Path(image_prefix) / image_name)


def convert_records(
	queries: list[dict[str, Any]],
	annotations_by_id: dict[int, dict[str, Any]],
	image_dir: Path,
	image_prefix: str,
	question_prefix: str,
	strict: bool,
) -> tuple[list[dict[str, Any]], list[str], Counter[str], Counter[str]]:
	converted: list[dict[str, Any]] = []
	skipped: list[str] = []
	truth_counts: Counter[str] = Counter()
	type_counts: Counter[str] = Counter()

	for query_row in queries:
		query_id = int(query_row["id"])
		annotation = annotations_by_id.get(query_id)
		if annotation is None:
			message = f"Missing annotation for query id={query_id}"
			if strict:
				raise KeyError(message)
			skipped.append(message)
			continue

		truth = str(annotation.get("truth", "")).strip().lower()
		if truth not in {"yes", "no"}:
			message = f"Unsupported truth label for query id={query_id}: {truth!r}"
			if strict:
				raise ValueError(message)
			skipped.append(message)
			continue

		image_name = str(query_row["image"])
		image_path = image_dir / image_name
		if not image_path.exists():
			message = f"Missing image for query id={query_id}: {image_path}"
			if strict:
				raise FileNotFoundError(message)
			skipped.append(message)
			continue

		annotation_type = str(annotation.get("type", "unknown"))
		truth_counts[truth] += 1
		type_counts[annotation_type] += 1

		converted.append(
			{
				"id": str(query_id),
				"image": normalize_image_path(image_name, image_prefix),
				"conversations": [
					{
						"from": "human",
						"value": build_prompt(str(query_row["query"]), question_prefix),
					},
					{"from": "gpt", "value": truth_to_option(truth)},
				],
				"source": "AMBER",
				"amber_type": annotation_type,
			}
		)

	return converted, skipped, truth_counts, type_counts


def main() -> None:
	args = parse_args()

	amber_root = args.amber_root.resolve()
	query_file = args.query_file or (amber_root / "query" / "query_discriminative.json")
	annotations_file = args.annotations_file or (amber_root / "annotations.json")
	output_path = args.output or (amber_root / "amber_discriminative_MCTS.jsonl")
	image_dir = amber_root / "images"
	image_prefix = args.image_prefix or f"{amber_root.name}/images"

	queries = load_json(query_file)
	annotations = load_json(annotations_file)
	annotations_by_id = {int(row["id"]): row for row in annotations}

	converted, skipped, truth_counts, type_counts = convert_records(
		queries=queries,
		annotations_by_id=annotations_by_id,
		image_dir=image_dir,
		image_prefix=image_prefix,
		question_prefix=args.question_prefix,
		strict=args.strict,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for row in converted:
			handle.write(json.dumps(row, ensure_ascii=True) + "\n")

	print(f"Wrote {len(converted)} records to {output_path}")
	if truth_counts:
		print("Label distribution:", dict(sorted(truth_counts.items())))
	if type_counts:
		print("Annotation types:", dict(sorted(type_counts.items())))
	if skipped:
		print(f"Skipped {len(skipped)} records")
		for message in skipped[:10]:
			print(f"  - {message}")
		if len(skipped) > 10:
			print(f"  ... and {len(skipped) - 10} more")


if __name__ == "__main__":
	main()
