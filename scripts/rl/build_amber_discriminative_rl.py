#!/usr/bin/env python3
"""Build image-grouped verl JSONL files from the AMBER discriminative data."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TYPE_TO_CATEGORY = {
    "discriminative-hallucination": "existence",
    "discriminative-attribute-state": "attribute-state",
    "discriminative-attribute-number": "attribute-number",
    "discriminative-attribute-action": "attribute-action",
    "discriminative-relation": "relation",
    "relation": "relation",
}


def parse_args() -> argparse.Namespace:
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=data_root / "mllm_hal/query/query_discriminative.json",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=data_root / "mllm_hal/annotations.json",
    )
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--output-dir", type=Path, default=data_root / "mllm_hal/rl")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array of objects in {path}.")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_annotations(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    annotations: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        annotation_id = row.get("id")
        if not isinstance(annotation_id, int):
            raise ValueError(f"Annotation has a non-integer id: {annotation_id!r}.")
        if annotation_id in annotations:
            raise ValueError(f"Duplicate annotation id: {annotation_id}.")
        annotations[annotation_id] = row
    return annotations


def normalize_image_path(image: Any) -> str:
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f"Invalid AMBER image path: {image!r}.")
    image_name = Path(image.replace("\\", "/")).name
    if not image_name:
        raise ValueError(f"Invalid AMBER image path: {image!r}.")
    return (Path("mllm_hal") / "images" / image_name).as_posix()


def normalize_answer(answer: Any, sample_id: int) -> str:
    if not isinstance(answer, str) or answer.strip().lower() not in {"yes", "no"}:
        raise ValueError(f"AMBER id {sample_id} has a non-binary truth value: {answer!r}.")
    return answer.strip().lower()


def select_validation_images(images: Iterable[str], val_fraction: float, seed: int) -> set[str]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1).")

    shuffled_images = sorted(set(images))
    random.Random(seed).shuffle(shuffled_images)
    val_count = int(len(shuffled_images) * val_fraction)
    if val_fraction > 0 and len(shuffled_images) > 1:
        val_count = max(1, min(val_count, len(shuffled_images) - 1))
    return set(shuffled_images[:val_count])


def build_rows(
    query_rows: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    annotations = index_annotations(annotation_rows)
    converted_rows: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_image_queries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    deduplicated_rows = []

    for query_row in query_rows:
        sample_id = query_row.get("id")
        if not isinstance(sample_id, int):
            raise ValueError(f"Query has a non-integer id: {sample_id!r}.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate query id: {sample_id}.")
        seen_ids.add(sample_id)

        annotation = annotations.get(sample_id)
        if annotation is None:
            raise ValueError(f"No annotation found for AMBER id {sample_id}.")
        if annotation.get("id") != sample_id:
            raise ValueError(f"Annotation id mismatch for AMBER id {sample_id}.")

        amber_type = annotation.get("type")
        category = TYPE_TO_CATEGORY.get(amber_type)
        if category is None:
            raise ValueError(f"AMBER id {sample_id} has an unknown discriminative type: {amber_type!r}.")

        query = query_row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"AMBER id {sample_id} has an empty query.")
        query = query.strip()
        image_path = normalize_image_path(query_row.get("image"))
        if not (data_root / image_path).is_file():
            raise FileNotFoundError(f"AMBER id {sample_id} image does not exist: {data_root / image_path}")

        image_query_key = (image_path, query.casefold())
        answer = normalize_answer(annotation.get("truth"), sample_id)
        existing_row = seen_image_queries.get(image_query_key)
        if existing_row is not None:
            if existing_row["answer"] != answer or existing_row["amber_type"] != amber_type:
                raise ValueError(
                    f"Conflicting duplicate image/query pair for AMBER ids {existing_row['id']} and {sample_id}."
                )
            deduplicated_rows.append(
                {
                    "kept_id": existing_row["id"],
                    "dropped_id": sample_id,
                    "image_key": image_path,
                    "query": query,
                }
            )
            continue

        converted_row = {
            "id": sample_id,
            "prompt": f"<image>\n{query}",
            "answer": answer,
            "images": [image_path],
            "image_key": image_path,
            "category": category,
            "amber_type": amber_type,
        }
        converted_rows.append(converted_row)
        seen_image_queries[image_query_key] = converted_row

    converted_rows.sort(key=lambda row: row["id"])
    validation_images = select_validation_images(
        (row["image_key"] for row in converted_rows), val_fraction=val_fraction, seed=seed
    )
    train_rows = [row for row in converted_rows if row["image_key"] not in validation_images]
    val_rows = [row for row in converted_rows if row["image_key"] in validation_images]

    train_images = {row["image_key"] for row in train_rows}
    val_images = {row["image_key"] for row in val_rows}
    report = build_report(
        train_rows,
        val_rows,
        train_images,
        val_images,
        seed,
        val_fraction,
        source_query_rows=len(query_rows),
        deduplicated_rows=deduplicated_rows,
    )
    return train_rows, val_rows, report


def split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    category_counts = Counter(str(row["category"]) for row in rows)
    type_counts = Counter(str(row["amber_type"]) for row in rows)
    answer_counts = Counter(str(row["answer"]) for row in rows)
    return {
        "rows": len(rows),
        "images": len({str(row["image_key"]) for row in rows}),
        "answers": dict(sorted(answer_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "amber_types": dict(sorted(type_counts.items())),
    }


def build_report(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    train_images: set[str],
    val_images: set[str],
    seed: int,
    val_fraction: float,
    source_query_rows: int,
    deduplicated_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    all_rows = [*train_rows, *val_rows]
    ids_by_image: Dict[str, List[int]] = defaultdict(list)
    for row in all_rows:
        ids_by_image[str(row["image_key"])].append(int(row["id"]))

    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "source_query_rows": source_query_rows,
        "deduplicated_row_count": len(deduplicated_rows),
        "deduplicated_rows": list(deduplicated_rows),
        "total": split_counts(all_rows),
        "train": split_counts(train_rows),
        "validation": split_counts(val_rows),
        "train_validation_image_overlap": len(train_images & val_images),
        "ids_by_image": {key: sorted(value) for key, value in sorted(ids_by_image.items())},
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_outputs(
    output_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    report: Dict[str, Any],
    query_file: Path,
    annotation_file: Path,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "amber_discriminative_train.jsonl"
    val_path = output_dir / "amber_discriminative_val.jsonl"
    report_path = output_dir / "amber_discriminative_split_report.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    report = {
        **report,
        "sources": {
            "query_file": str(query_file.resolve()),
            "query_sha256": sha256_file(query_file),
            "annotation_file": str(annotation_file.resolve()),
            "annotation_sha256": sha256_file(annotation_file),
        },
        "outputs": {
            "train_file": str(train_path.resolve()),
            "validation_file": str(val_path.resolve()),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return train_path, val_path, report_path


def main() -> None:
    args = parse_args()
    query_file = args.query_file.resolve()
    annotation_file = args.annotation_file.resolve()
    data_root = args.data_root.resolve()

    train_rows, val_rows, report = build_rows(
        load_json_array(query_file),
        load_json_array(annotation_file),
        data_root=data_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_path, val_path, report_path = write_outputs(
        args.output_dir.resolve(), train_rows, val_rows, report, query_file, annotation_file
    )
    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(val_rows)} validation rows to {val_path}")
    print(f"Wrote split report to {report_path}")


if __name__ == "__main__":
    main()