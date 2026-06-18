#!/usr/bin/env python3
"""Build a frozen AMBER validation file for src.vlmsearch inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the AMBER RL validation split into src.vlmsearch JSONL format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mllm_hal/amber_discriminative_val_RL.jsonl"),
        help="Frozen AMBER validation JSONL in RL schema.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mllm_hal/amber_discriminative_val_1000_vlmsearch.jsonl"),
        help="Output JSONL path in src.vlmsearch schema.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data"),
        help="Root used to verify relative image paths.",
    )
    parser.add_argument(
        "--ids-output",
        type=Path,
        default=None,
        help="Optional path to write the frozen sample ids, one per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke-test files.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Do not verify that referenced images exist under --image-root.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_record(record: dict[str, Any], image_root: Path, skip_image_check: bool) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"Expected exactly one image for id={record.get('id')!r}; got {images!r}")

    image_path = str(images[0])
    if not skip_image_check and not (image_root / image_path).exists():
        raise FileNotFoundError(f"Missing image for id={record.get('id')!r}: {image_root / image_path}")

    answer = str(record.get("answer", "")).strip().upper()
    if answer not in {"A", "B"}:
        raise ValueError(f"Expected A/B answer for id={record.get('id')!r}; got {answer!r}")

    return {
        "id": str(record["id"]),
        "image": image_path,
        "conversations": [
            {"from": "human", "value": str(record["problem"])},
            {"from": "gpt", "value": answer},
        ],
        "source": record.get("source", "AMBER"),
        "amber_type": record.get("amber_type", "unknown"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    converted: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for index, record in enumerate(read_jsonl(args.input)):
        if args.limit is not None and index >= args.limit:
            break
        converted_record = convert_record(record, args.image_root, args.skip_image_check)
        converted.append(converted_record)
        label_counts[converted_record["conversations"][1]["value"]] += 1
        type_counts[str(converted_record.get("amber_type", "unknown"))] += 1

    if not converted:
        raise ValueError(f"No records converted from {args.input}")

    write_jsonl(args.output, converted)

    if args.ids_output is not None:
        args.ids_output.parent.mkdir(parents=True, exist_ok=True)
        with args.ids_output.open("w", encoding="utf-8") as handle:
            for record in converted:
                handle.write(f"{record['id']}\n")

    print(f"Wrote {len(converted)} records to {args.output}")
    print("Label distribution:", dict(sorted(label_counts.items())))
    print("AMBER types:", dict(sorted(type_counts.items())))
    if args.ids_output is not None:
        print(f"Wrote frozen ids to {args.ids_output}")


if __name__ == "__main__":
    main()