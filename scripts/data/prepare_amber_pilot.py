#!/usr/bin/env python3
"""Prepare a reviewed, image-disjoint-within-pilot AMBER training pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random

from vlmsearch.data_manifest import FrozenDataManifest, file_sha256


CATEGORIES = {
    "discriminative-hallucination": "existence",
    "discriminative-attribute-state": "state",
    "discriminative-attribute-number": "count",
    "discriminative-attribute-action": "action",
    "discriminative-relation": "relation",
    "relation": "relation",
}
BASE_QUOTAS = {
    "existence:no": 18, "state:no": 14, "count:no": 14, "action:no": 14, "relation:no": 15,
    "state:yes": 19, "count:yes": 19, "action:yes": 19, "relation:yes": 18,
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sample_cell(sample):
    return CATEGORIES[sample["category"]] + ":" + sample["answer"]


def allocate_quotas(count):
    """Keep equal class totals and proportional category cells;150 yields BASE_QUOTAS."""
    if count < 18:
        raise ValueError("At least 18 questions are required to cover all available cells")
    quotas = {}
    for answer, target in [("yes", count // 2), ("no", count - count // 2)]:
        cells = [cell for cell in BASE_QUOTAS if cell.endswith(":" + answer)]
        exact = {cell: BASE_QUOTAS[cell] * target / 75 for cell in cells}
        quotas.update({cell: int(exact[cell]) for cell in cells})
        left = target - sum(quotas[cell] for cell in cells)
        for cell in sorted(cells, key=lambda cell: (-(exact[cell] - quotas[cell]), cell))[:left]:
            quotas[cell] += 1
    return {cell: quotas[cell] for cell in BASE_QUOTAS}


def size_bucket(sample, images, max_pixels):
    image = images[sample["image"]]
    return "large" if image["width"] * image["height"] > max_pixels else "standard"


def choose_samples(samples, images, quotas, seed=42, max_pixels=1048576, large_fraction=0.2):
    """Use max flow to assign disjoint image groups to category/answer/size quotas."""
    if not 0 <= large_fraction <= 1:
        raise ValueError("large_fraction must lie between zero and one")
    rng = random.Random(seed)
    candidates = sorted(samples, key=lambda sample: sample["id"])
    rng.shuffle(candidates)
    pools = defaultdict(dict)
    for sample in candidates:
        key = (sample_cell(sample), size_bucket(sample, images, max_pixels))
        pools[key].setdefault(sample["cluster_id"], sample)
    targets = {}
    for cell, quota in quotas.items():
        available_large = len(pools[cell, "large"])
        large_target = min(available_large, round(quota * large_fraction))
        targets[cell, "large"] = large_target
        targets[cell, "standard"] = quota - large_target

    # Include the largest eligible original image explicitly so that transport
    # costs are exercised even if random stratification misses the extreme.
    largest = min(candidates, key=lambda sample: (
        -images[sample["image"]]["width"] * images[sample["image"]]["height"],
        0 if sample_cell(sample) == "existence:no" else 1,
        sample["id"],
    ))
    largest_key = (sample_cell(largest), size_bucket(largest, images, max_pixels))
    selected = []
    reserved_group = None
    if targets.get(largest_key, 0):
        selected.append(largest)
        reserved_group = largest["cluster_id"]
        targets[largest_key] -= 1

    graph = defaultdict(dict)

    def edge(start, end, capacity):
        graph[start][end] = capacity
        graph[end][start] = 0

    source, sink = ("source",), ("sink",)
    stratum_nodes = {}
    candidate_for_edge = {}
    for key in sorted(targets):
        target = targets[key]
        if not target:
            continue
        stratum = ("stratum", *key)
        stratum_nodes[key] = stratum
        edge(source, stratum, target)
        groups = list(pools[key])
        rng.shuffle(groups)
        for group in groups:
            if group == reserved_group:
                continue
            node = ("image", group)
            edge(stratum, node, 1)
            if sink not in graph[node]:
                edge(node, sink, 1)
            candidate_for_edge[stratum, node] = pools[key][group]

    flow = 0
    while True:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbor, capacity in graph[node].items():
                if capacity > 0 and neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if sink not in parents:
            break
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            graph[parent][node] -= 1
            graph[node][parent] += 1
            node = parent
        flow += 1

    for (stratum, node), sample in candidate_for_edge.items():
        if graph[stratum][node] == 0:
            selected.append(sample)
    target_count = sum(quotas.values())
    if len(selected) != target_count:
        raise ValueError(f"Cannot fill disjoint image/category/size quotas: selected {len(selected)}/{target_count}")
    if len({sample["cluster_id"] for sample in selected}) != target_count:
        raise AssertionError("Pilot selection reused an image group")
    if Counter(sample_cell(sample) for sample in selected) != Counter(quotas):
        raise AssertionError("Pilot selection did not preserve cell quotas")
    rng.shuffle(selected)
    return selected


def prepare_pilot(manifest_path, review_path, train_file, data_root, output_dir, count=150, seed=42):
    from vlmsearch.data_review import ReviewedTrainingData

    manifest = FrozenDataManifest(manifest_path, data_root)
    review = ReviewedTrainingData(review_path, manifest)
    input_rows = [json.loads(line) for line in Path(train_file).read_text().splitlines() if line.strip()]
    seen = set()
    eligible = []
    source_by_id = {}
    for row in input_rows:
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate input ID: {sample_id}")
        seen.add(sample_id)
        conversation = row.get("conversations", [])
        if len(conversation) != 2 or [turn.get("from") for turn in conversation] != ["human", "gpt"]:
            raise ValueError(f"Malformed MCTS conversation for {sample_id}")
        manifest.validate_sample(sample_id, row["image"], conversation[0]["value"], conversation[1]["value"], "train")
        sample = manifest.samples[sample_id]
        if sample["dataset"] != "amber":
            raise ValueError("Pilot source must contain only AMBER training rows")
        source_by_id[sample_id] = row
        if sample_id in review.excluded_ids:
            continue
        review.validate_sample(sample_id)
        eligible.append(dict(sample, id=sample_id))
    expected = {sample_id for sample_id, sample in manifest.samples.items()
                if sample["dataset"] == "amber" and sample["split"] == "train"}
    if seen != expected:
        raise ValueError("Pilot source must be the complete frozen AMBER training export")
    quotas = allocate_quotas(count)
    selected = choose_samples(eligible, manifest.images, quotas, seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "pilot.mcts.jsonl"
    payload = "".join(json.dumps(source_by_id[sample["id"]], ensure_ascii=False, separators=(",", ":")) + "\n" for sample in selected)
    selection = {
        "schema_version": 1,
        "purpose": "Reviewed AMBER-train diagnostic pilot, not a held-out accuracy estimate",
        "seed": seed,
        "policy": {"questions": count, "yes_no_balanced": True, "image_groups_unique": True,
                   "canonical_cells": quotas, "target_large_image_fraction_per_cell": 0.2,
                   "large_image_threshold_pixels": 1048576, "largest_eligible_image_included": True,
                   "sampling": "Seeded max-flow assignment across category/label/image-size strata"},
        "frozen_manifest": {"path": str(Path(manifest_path).resolve()), "sha256": manifest.sha256,
                            "manifest_id": manifest.data.get("manifest_id")},
        "review_overlay": {"path": str(Path(review_path).resolve()), "sha256": review.sha256, "overlay_id": review.id},
        "source_training_export": {"path": str(Path(train_file).resolve()), "sha256": file_sha256(train_file)},
        "pilot_file": {"path": str(rows_path), "sha256": hashlib.sha256(payload.encode()).hexdigest()},
        "counts": {"eligible_questions": len(eligible), "excluded_train_ids": len(review.excluded_ids),
                   "selected_questions": len(selected), "image_groups": len({sample["cluster_id"] for sample in selected}),
                   "labels": dict(Counter(sample["answer"] for sample in selected)),
                   "cells": dict(Counter(sample_cell(sample) for sample in selected)),
                   "image_size_buckets": dict(Counter(size_bucket(sample, manifest.images, 1048576) for sample in selected))},
        "samples": [{"id": sample["id"], "cell": sample_cell(sample), "category": CATEGORIES[sample["category"]],
                     "source_category": sample["category"], "answer": sample["answer"], "image": sample["image"],
                     "question": sample["question"], "split": "train", "image_group": sample["cluster_id"],
                     "original_size": [manifest.images[sample["image"]]["width"], manifest.images[sample["image"]]["height"]],
                     "image_sha256": manifest.images[sample["image"]]["file_sha256"],
                     "image_size_bucket": size_bucket(sample, manifest.images, 1048576)} for sample in selected],
    }
    selection["selection_id"] = canonical_hash(selection)
    selection_path = output_dir / "selection.json"
    encoded_selection = json.dumps(selection, indent=2, ensure_ascii=False) + "\n"
    for path, content in [(rows_path, payload), (selection_path, encoded_selection)]:
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite a different frozen pilot: {path}; use a new output directory")
    rows_path.write_text(payload)
    selection_path.write_text(encoded_selection)
    return selection


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--review-overlay", type=Path, default=repo / "configs/data/amber_train_review_v1.json")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 100 <= args.count <= 200:
        parser.error("--count must lie between 100 and 200")
    prepared = args.data_root / "hallucination/prepared"
    selection = prepare_pilot(args.split_manifest or prepared / "manifest.json", args.review_overlay,
                              args.train_file or prepared / "amber/train.mcts.jsonl", args.data_root,
                              args.output_dir or args.data_root / f"hallucination/pilots/amber_{args.count}_seed{args.seed}_v1",
                              args.count, args.seed)
    print(json.dumps({"selection_id": selection["selection_id"], "pilot_file": selection["pilot_file"],
                      "counts": selection["counts"]}, indent=2))


if __name__ == "__main__":
    main()
