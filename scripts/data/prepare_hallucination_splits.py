#!/usr/bin/env python3
"""Freeze image-disjoint AMBER training splits and external benchmark tests.

Inputs are the provenance-preserving JSONL files produced by
download_hallucination_data.py. POPE and all LongHalQA variants remain test
data. All records of a matched image cluster share one train/dev/test split.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image
from scipy.fft import dctn


DATASETS = ("amber", "pope", "longhalqa")
SPLITS = ("train", "dev", "test")
SCHEMA_VERSION = 1


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(canonical_json(row) + "\n" for row in rows))


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(row)
    return rows


def load_inputs(normalized_dir, data_root):
    records = []
    inputs = {}
    seen_ids = set()
    required = {"id", "dataset", "task", "image", "question", "answer", "answer_type", "source_revision"}
    for dataset in DATASETS:
        path = Path(normalized_dir) / f"{dataset}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"Empty required dataset: {path}")
        try:
            relative_input = path.resolve().relative_to(Path(data_root).resolve()).as_posix()
        except ValueError:
            relative_input = str(path.resolve())
        inputs[dataset] = {"path": relative_input, "sha256": sha256_file(path), "rows": len(rows)}
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in {path}: {row.get('id')}")
            if row["dataset"] != dataset:
                raise ValueError(f"Wrong dataset in {path}: {row['id']}")
            sample_id = row["id"]
            if not isinstance(sample_id, str) or not sample_id.startswith(dataset + ":"):
                raise ValueError(f"Sample ID must be namespaced with {dataset}: {sample_id!r}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"Empty question: {sample_id}")
            allowed = {"yes", "no"} if row["answer_type"] == "binary" else {"A", "B", "C", "D"}
            if row["answer_type"] not in {"binary", "choice"} or row["answer"] not in allowed:
                raise ValueError(f"Invalid answer type/value: {sample_id}: {row['answer_type']}/{row['answer']}")
            if dataset == "amber" and row["answer_type"] != "binary":
                raise ValueError(f"AMBER training input must be discriminative binary: {sample_id}")
            image = Path(row["image"])
            if image.is_absolute() or ".." in image.parts or not image.parts:
                raise ValueError(f"Image must be relative to data root: {sample_id}: {image}")
            image_path = (Path(data_root) / image).resolve()
            try:
                image_path.relative_to(Path(data_root).resolve())
            except ValueError as exc:
                raise ValueError(f"Image escapes data root: {sample_id}") from exc
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image for {sample_id}: {image_path}")
            row = dict(row, image=image.as_posix())
            records.append(row)
    return sorted(records, key=lambda row: row["id"]), inputs


def source_key(row):
    value = str(row.get("source_image_id") or "").strip()
    if not value:
        return None
    # Only explicit image-corpus namespaces are shared across datasets. A bare
    # number such as 123 from AMBER and LongHalQA is not an image identity.
    if ":" in value and value.split(":", 1)[0].lower() in {
        "amber", "coco", "coco2014", "coco2017", "objects365", "visualgenome", "vg", "unsplash"
    }:
        prefix, identifier = value.split(":", 1)
        if prefix.lower() == "vg":
            prefix = "visualgenome"
        return prefix.lower() + ":" + identifier
    return row["dataset"] + ":" + value


def fingerprint_image(path):
    # verify() checks structural integrity; loading a second time ensures the
    # complete pixel payload decodes before the dataset is admitted.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {path}")
        pixels = hashlib.sha256(f"RGB:{width}:{height}:".encode())
        pixels.update(rgb.tobytes())
        gray = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        spectrum = dctn(gray, type=2, norm="ortho")[:8, :8].reshape(-1)
        median = np.median(spectrum[1:])
        bits = spectrum > median
        phash = 0
        for bit in bits:
            phash = (phash << 1) | int(bit)
        thumbnail = np.asarray(rgb.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    return {
        "file_sha256": sha256_file(path),
        "rgb_sha256": pixels.hexdigest(),
        "width": width,
        "height": height,
        "phash": f"{phash:016x}",
        "luminance_std": round(float(gray.std()), 6),
    }, thumbnail


class UnionFind:
    def __init__(self, keys):
        self.parents = {key: key for key in keys}

    def find(self, key):
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def near_duplicate(left, right, thumbnails, policy):
    distance = (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count()
    if distance > policy["phash_distance"]:
        return None
    left_ratio = left["width"] / left["height"]
    right_ratio = right["width"] / right["height"]
    ratio_error = abs(math.log(left_ratio / right_ratio))
    if ratio_error > policy["aspect_ratio_log_tolerance"]:
        return None
    rmse = float(np.sqrt(np.mean((thumbnails[0] - thumbnails[1]) ** 2)))
    # Uniform images have uninformative pHashes. Red and blue blank images must
    # not collapse into one cluster solely because both lack spatial structure.
    low_information = min(left["luminance_std"], right["luminance_std"]) < policy["min_luminance_std"]
    accepted = not low_information or rmse <= policy["flat_image_rgb_rmse"]
    return {
        "phash_distance": distance,
        "aspect_ratio_log_error": round(ratio_error, 8),
        "rgb_thumbnail_rmse": round(rmse, 8),
        "low_information_hash": low_information,
        "merged": accepted,
        "reason": "perceptual_match" if accepted else "uninformative_flat_hash_with_different_pixels",
    }


def group_images(records, data_root, policy):
    by_image = defaultdict(list)
    for row in records:
        by_image[row["image"]].append(row)
    images = {}
    thumbnails = {}
    def fingerprint_name(name):
        try:
            info, thumbnail = fingerprint_image(Path(data_root) / name)
        except Exception as exc:
            raise ValueError(f"Invalid image {name}: {exc}") from exc
        return name, info, thumbnail

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        for index, (name, info, thumbnail) in enumerate(executor.map(fingerprint_name, sorted(by_image)), 1):
            thumbnails[name] = thumbnail
            info["source_keys"] = sorted({key for row in by_image[name] if (key := source_key(row)) is not None})
            images[name] = info
            if index % 250 == 0:
                print(f"Fingerprinted {index}/{len(by_image)} images", flush=True)
    union = UnionFind(images)
    exact_indexes = {"file_sha256": {}, "rgb_sha256": {}, "source_key": {}}
    exact_edges = []
    for name, info in images.items():
        for kind in ("file_sha256", "rgb_sha256", "source_key"):
            values = info["source_keys"] if kind == "source_key" else [info[kind]]
            for value in values:
                previous = exact_indexes[kind].setdefault(value, name)
                if previous != name:
                    union.union(previous, name)
                    exact_edges.append({"left": previous, "right": name, "reason": kind, "value": value})
    # Compare one representative per exact group. Multiple source encodings can
    # have different pHashes, so retain each distinct pHash/aspect combination.
    representatives = {}
    for name, info in images.items():
        key = (union.find(name), info["phash"], info["width"], info["height"])
        representatives.setdefault(key, name)
    names = sorted(representatives.values())
    candidates = []
    for index, left_name in enumerate(names):
        left = images[left_name]
        for right_name in names[index + 1:]:
            if union.find(left_name) == union.find(right_name):
                continue
            match = near_duplicate(left, images[right_name], (thumbnails[left_name], thumbnails[right_name]), policy)
            if match is not None:
                match.update(left=left_name, right=right_name)
                candidates.append(match)
                if match["merged"]:
                    union.union(left_name, right_name)
    grouped = defaultdict(list)
    for name in images:
        grouped[union.find(name)].append(name)
    clusters = {}
    for members in grouped.values():
        members.sort()
        pixel_hashes = sorted({images[name]["rgb_sha256"] for name in members})
        cluster_id = "image:" + hashlib.sha256("\n".join(pixel_hashes).encode()).hexdigest()[:24]
        datasets = sorted({row["dataset"] for name in members for row in by_image[name]})
        clusters[cluster_id] = {
            "images": members,
            "datasets": datasets,
            "source_keys": sorted({key for name in members for key in images[name]["source_keys"]}),
            "reserved_for_external_test": bool(set(datasets) & {"pope", "longhalqa"}),
        }
        for name in members:
            images[name]["cluster_id"] = cluster_id
    return images, dict(sorted(clusters.items())), exact_edges, candidates


def assign_splits(clusters, policy):
    eligible = sorted(key for key, value in clusters.items() if not value["reserved_for_external_test"])
    random.Random(policy["seed"]).shuffle(eligible)
    count = len(eligible)
    train_count = int(count * policy["amber_fractions"]["train"])
    dev_count = int(count * policy["amber_fractions"]["dev"])
    if count >= 3:
        dev_count = max(1, dev_count)
        train_count = min(max(1, train_count), count - dev_count - 1)
    elif count == 1:
        train_count = 1
    for index, cluster_id in enumerate(eligible):
        clusters[cluster_id]["split"] = "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
    for cluster in clusters.values():
        if cluster["reserved_for_external_test"]:
            cluster["split"] = "test"


def make_manifest(records, inputs, images, clusters, policy):
    assign_splits(clusters, policy)
    for info in images.values():
        info["split"] = clusters[info["cluster_id"]]["split"]
    samples = {}
    for row in records:
        image = images[row["image"]]
        samples[row["id"]] = {
            "dataset": row["dataset"], "task": row["task"], "image": row["image"],
            "cluster_id": image["cluster_id"], "split": image["split"],
            "answer_type": row["answer_type"], "answer": row["answer"],
            "question": row["question"],
            "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
            "source_revision": row["source_revision"],
            "source_id": row.get("source_id"), "source_image_id": row.get("source_image_id"),
            "category": row.get("category"),
        }
    manifest = {"schema_version": SCHEMA_VERSION, "policy": policy, "source_inputs": inputs,
                "images": images, "clusters": clusters, "samples": samples}
    manifest["manifest_id"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return manifest


def freeze_manifest(path, manifest):
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            raise ValueError(f"Frozen manifest differs from this input/policy: {path}. Use a new output directory for a new version; do not replace the existing split.")
    else:
        write_json(path, manifest)


def export_rows(records, manifest, output_dir):
    output_dir = Path(output_dir)
    exports = {}
    for split in SPLITS:
        selected = [row for row in records if row["dataset"] == "amber" and manifest["samples"][row["id"]]["split"] == split]
        mcts = [{"id": row["id"], "image": row["image"], "conversations": [
            {"from": "human", "value": image_prompt(row["question"])}, {"from": "gpt", "value": row["answer"]}
        ]} for row in selected]
        rl = [{"id": row["id"], "prompt": image_prompt(row["question"]), "answer": row["answer"],
               "images": [row["image"]], "image_key": manifest["samples"][row["id"]]["cluster_id"], "category": row.get("category", "unknown"),
               "cluster_id": manifest["samples"][row["id"]]["cluster_id"], "split": split} for row in selected]
        evaluation = [dict(row, split=split, cluster_id=manifest["samples"][row["id"]]["cluster_id"]) for row in selected]
        for suffix, rows in (("mcts", mcts), ("rl", rl), ("evaluation", evaluation)):
            relative = f"amber/{split}.{suffix}.jsonl"
            write_jsonl(output_dir / relative, rows)
            exports[relative] = {"rows": len(rows), "sha256": sha256_file(output_dir / relative)}
    for dataset in ("pope", "longhalqa"):
        selected = [dict(row, split="test", cluster_id=manifest["samples"][row["id"]]["cluster_id"]) for row in records if row["dataset"] == dataset]
        relative = f"evaluation/{dataset}.jsonl"
        write_jsonl(output_dir / relative, selected)
        exports[relative] = {"rows": len(selected), "sha256": sha256_file(output_dir / relative)}
        for task in sorted({row["task"] for row in selected}):
            if not all(character.isalnum() or character in "_-" for character in task):
                raise ValueError(f"Unsafe task name: {task}")
            relative = f"evaluation/{dataset}/{task}.jsonl"
            rows = [row for row in selected if row["task"] == task]
            write_jsonl(output_dir / relative, rows)
            exports[relative] = {"rows": len(rows), "sha256": sha256_file(output_dir / relative)}
    conflicts = ambiguous_question_groups(records, manifest)
    ambiguous_ids = {sample_id for group in conflicts for ids in group["answers"].values() for sample_id in ids}
    unambiguous = [dict(row, split="test", cluster_id=manifest["samples"][row["id"]]["cluster_id"])
                   for row in records if row["dataset"] == "longhalqa" and row["id"] not in ambiguous_ids]
    relative = "evaluation/longhalqa.unambiguous.jsonl"
    write_jsonl(output_dir / relative, unambiguous)
    exports[relative] = {"rows": len(unambiguous), "sha256": sha256_file(output_dir / relative),
                         "purpose": "Sensitivity analysis excluding all members of exact image/question groups with conflicting labels; official primary export is unchanged."}
    write_json(output_dir / "label-conflicts.json", {
        "manifest_id": manifest["manifest_id"], "official_labels_preserved": True,
        "ambiguous_samples": len(ambiguous_ids), "conflicting_groups": len(conflicts), "groups": conflicts,
        "policy": "Preserve original benchmark labels. Only the separate longhalqa.unambiguous.jsonl sensitivity export excludes conflicting groups; no label is corrected or inferred.",
    })
    return exports


def image_prompt(question):
    return question if "<image>" in question else "<image>\n" + question


def write_contact_sheets(candidates, data_root, output_dir):
    """Render matched candidates for a human review of perceptual grouping."""
    from PIL import ImageDraw, ImageOps

    output_dir = Path(output_dir)
    for page, offset in enumerate(range(0, len(candidates), 8), 1):
        rows = candidates[offset:offset + 8]
        sheet = Image.new("RGB", (900, 300 * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            top = index * 300
            draw.text((10, top + 4), f"Pair {offset + index + 1}: pHash={row['phash_distance']} RGB RMSE={row['rgb_thumbnail_rmse']:.4f} merged={row['merged']}", fill="black")
            for column, key in enumerate(("left", "right")):
                name = row[key]
                with Image.open(Path(data_root) / name) as original:
                    tile = ImageOps.contain(original.convert("RGB"), (430, 245))
                x = column * 450 + 10
                sheet.paste(tile, (x, top + 28))
                draw.text((x, top + 278), Path(name).name[:60], fill="black")
        output_dir.mkdir(parents=True, exist_ok=True)
        sheet.save(output_dir / f"pairs-{page:03d}.jpg", quality=90)


def ambiguous_question_groups(records, manifest):
    groups = defaultdict(lambda: defaultdict(list))
    by_id = {row["id"]: row for row in records}
    for row in records:
        sample = manifest["samples"][row["id"]]
        key = (sample["cluster_id"], row["question"].strip().casefold())
        groups[key][row["answer"]].append(row["id"])
    conflicts = []
    for (cluster_id, question_key), answers in sorted(groups.items()):
        if len(answers) <= 1:
            continue
        sample_ids = sorted(sample_id for ids in answers.values() for sample_id in ids)
        conflicts.append({
            "group_id": hashlib.sha256((cluster_id + "\n" + question_key).encode()).hexdigest()[:24],
            "cluster_id": cluster_id, "question": by_id[sample_ids[0]]["question"],
            "answers": {answer: sorted(ids) for answer, ids in sorted(answers.items())},
            "samples": [{key: by_id[sample_id].get(key) for key in ("id", "dataset", "task", "image", "answer", "source_id", "source_image_id", "source_revision")} for sample_id in sample_ids],
            "reason": "Same image cluster and whitespace-trimmed/casefolded question have conflicting official answer labels.",
        })
    return conflicts


def make_report(records, manifest, exact_edges, candidates, exports):
    counts = {}
    for dataset in DATASETS:
        counts[dataset] = {}
        for split in SPLITS:
            selected = [row for row in records if row["dataset"] == dataset and manifest["samples"][row["id"]]["split"] == split]
            counts[dataset][split] = {
                "rows": len(selected), "images": len({row["image"] for row in selected}),
                "clusters": len({manifest["samples"][row["id"]]["cluster_id"] for row in selected}),
                "answers": dict(sorted(Counter(row["answer"] for row in selected).items())),
                "categories": dict(sorted(Counter(str(row.get("category", "unknown")) for row in selected).items())),
                "tasks": dict(sorted(Counter(row["task"] for row in selected).items())),
            }
    duplicates = defaultdict(list)
    question_answers = defaultdict(lambda: defaultdict(list))
    for row in records:
        key = (manifest["samples"][row["id"]]["cluster_id"], row["question"].strip().casefold(), row["answer"])
        duplicates[key].append(row["id"])
        question_answers[key[:2]][row["answer"]].append(row["id"])
    return {
        "manifest_id": manifest["manifest_id"], "counts": counts,
        "total_rows": len(records), "total_image_files": len(manifest["images"]), "total_image_clusters": len(manifest["clusters"]),
        "exact_matching_edges": len(exact_edges),
        "perceptual_candidates": len(candidates), "perceptual_matches_merged": sum(row["merged"] for row in candidates),
        "rejected_flat_hash_candidates": sum(not row["merged"] for row in candidates),
        "amber_rows_reserved_for_external_test": sum(row["dataset"] == "amber" and manifest["clusters"][manifest["samples"][row["id"]]["cluster_id"]]["reserved_for_external_test"] for row in records),
        "duplicate_question_groups_retained": [ids for ids in duplicates.values() if len(ids) > 1],
        "conflicting_question_groups_for_review": [dict(values) for values in question_answers.values() if len(values) > 1],
        "label_conflict_details": "label-conflicts.json",
        "exports": exports,
        "scope": "Project train/dev/test isolation. Public benchmark images are not guaranteed unseen to pretrained base models.",
        "near_duplicate_limit": "Canonical source IDs, exact file/RGB hashes, and 64-bit pHash <= configured threshold with aspect-ratio guard. Unknown severe crops or large edits can evade perceptual matching; candidates are retained for review.",
    }


def audit_dataset(records, inputs, data_root, output_dir, recompute_images=True):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.pop("manifest_id")
    expected_identity = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    manifest["manifest_id"] = identity
    if identity != expected_identity:
        raise ValueError("Manifest content hash does not match manifest_id")
    if manifest["source_inputs"] != inputs:
        raise ValueError("Normalized source files changed after the split was frozen")
    if {row["id"] for row in records} != set(manifest["samples"]):
        raise ValueError("Manifest and normalized sample IDs differ")
    if {row["image"] for row in records} != set(manifest["images"]):
        raise ValueError("Manifest and normalized image paths differ")
    expected_source_keys = defaultdict(set)
    for row in records:
        key = source_key(row)
        if key is not None:
            expected_source_keys[row["image"]].add(key)
    exact_splits = defaultdict(set)
    fingerprints = {}
    thumbnails = {}
    if recompute_images:
        names = sorted(manifest["images"])
        def verify_image(name):
            current, thumbnail = fingerprint_image(Path(data_root) / name)
            return name, current, thumbnail
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            for index, (name, current, thumbnail) in enumerate(executor.map(verify_image, names), 1):
                fingerprints[name] = current
                thumbnails[name] = thumbnail
                if index % 250 == 0:
                    print(f"Audited {index}/{len(names)} image payloads", flush=True)
    for name, image in manifest["images"].items():
        if image["source_keys"] != sorted(expected_source_keys[name]):
            raise ValueError(f"Manifest source identities differ from normalized data: {name}")
        if recompute_images:
            current = fingerprints[name]
            for key in ("file_sha256", "rgb_sha256", "width", "height", "phash", "luminance_std"):
                if current[key] != image[key]:
                    raise ValueError(f"Image changed after freeze: {name}: {key}")
        split = image["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {name}: {split}")
        cluster = manifest["clusters"][image["cluster_id"]]
        if cluster["split"] != split or name not in cluster["images"]:
            raise ValueError(f"Image/cluster split mismatch: {name}")
        for key in ("file_sha256", "rgb_sha256"):
            exact_splits[(key, image[key])].add(split)
        for key in image["source_keys"]:
            exact_splits[("source", key)].add(split)
    if any(len(splits) > 1 for splits in exact_splits.values()):
        raise ValueError("Exact image/source identity crosses train/dev/test")
    for row in records:
        sample = manifest["samples"][row["id"]]
        image = manifest["images"][row["image"]]
        for key in ("dataset", "task", "image", "question", "answer", "answer_type", "source_revision", "source_id", "source_image_id", "category"):
            if sample[key] != row.get(key):
                raise ValueError(f"Sample metadata changed: {row['id']}: {key}")
        if sample["split"] != image["split"] or sample["cluster_id"] != image["cluster_id"]:
            raise ValueError(f"Sample/image split mismatch: {row['id']}")
        if sample["answer"] != row["answer"] or sample["question_sha256"] != hashlib.sha256(row["question"].encode()).hexdigest():
            raise ValueError(f"Question or answer changed: {row['id']}")
        if row["dataset"] in {"pope", "longhalqa"} and sample["split"] != "test":
            raise ValueError(f"External benchmark leaked into training/development: {row['id']}")
    near_pairs_checked = 0
    if recompute_images:
        names = sorted(fingerprints)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1:]:
                if manifest["images"][left_name]["split"] == manifest["images"][right_name]["split"]:
                    continue
                match = near_duplicate(fingerprints[left_name], fingerprints[right_name], (thumbnails[left_name], thumbnails[right_name]), manifest["policy"])
                near_pairs_checked += 1
                if match is not None and match["merged"]:
                    raise ValueError(f"Near-duplicate image crosses partitions: {left_name} / {right_name}")
    report = json.loads((output_dir / "report.json").read_text())
    if report["manifest_id"] != identity:
        raise ValueError("Report belongs to a different manifest")
    conflicts = ambiguous_question_groups(records, manifest)
    ambiguous_ids = {sample_id for group in conflicts for ids in group["answers"].values() for sample_id in ids}
    conflict_file = json.loads((output_dir / "label-conflicts.json").read_text())
    if conflict_file["groups"] != conflicts or conflict_file["manifest_id"] != identity or not conflict_file["official_labels_preserved"]:
        raise ValueError("Label-conflict audit does not match frozen labels")
    for relative, info in report["exports"].items():
        path = output_dir / relative
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Export changed after preparation: {path}")
        exported = load_jsonl(path)
        if len(exported) != info["rows"] or len({row["id"] for row in exported}) != len(exported):
            raise ValueError(f"Export sample count/identity error: {path}")
        if relative.startswith("amber/"):
            split = Path(relative).name.split(".")[0]
            expected_ids = {row["id"] for row in records if row["dataset"] == "amber" and manifest["samples"][row["id"]]["split"] == split}
            if {row["id"] for row in exported} != expected_ids:
                raise ValueError(f"Export does not match frozen AMBER split: {path}")
        elif relative.startswith("evaluation/"):
            parts = Path(relative).parts
            dataset = Path(parts[1]).stem.split(".")[0]
            task = Path(parts[2]).stem if len(parts) > 2 else None
            sensitivity = relative == "evaluation/longhalqa.unambiguous.jsonl"
            expected_ids = {row["id"] for row in records if row["dataset"] == dataset and (task is None or row["task"] == task)
                            and (not sensitivity or row["id"] not in ambiguous_ids)}
            if {row["id"] for row in exported} != expected_ids:
                raise ValueError(f"Evaluation export omits or adds benchmark samples: {path}")
        for row in exported:
            if row["id"] not in manifest["samples"]:
                raise ValueError(f"Unknown sample in export: {path}: {row['id']}")
            if relative.startswith("evaluation/") and manifest["samples"][row["id"]]["split"] != "test":
                raise ValueError(f"Evaluation export includes non-test sample: {path}")
            sample = manifest["samples"][row["id"]]
            if relative.endswith(".mcts.jsonl"):
                expected_conversations = [
                    {"from": "human", "value": image_prompt(sample["question"])},
                    {"from": "gpt", "value": sample["answer"]},
                ]
                if row.get("image") != sample["image"] or row.get("conversations") != expected_conversations:
                    raise ValueError(f"MCTS export prompt/image/answer differs from manifest: {path}: {row['id']}")
            elif relative.endswith(".rl.jsonl"):
                expected = {
                    "prompt": image_prompt(sample["question"]), "answer": sample["answer"],
                    "images": [sample["image"]], "image_key": sample["cluster_id"],
                    "category": sample["category"], "cluster_id": sample["cluster_id"], "split": sample["split"],
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"RL export prompt/image/answer differs from manifest: {path}: {row['id']}")
            else:
                keys = ("dataset", "task", "image", "question", "answer", "answer_type", "source_revision", "source_id", "source_image_id", "category", "cluster_id", "split")
                if any(row.get(key) != sample[key] for key in keys):
                    raise ValueError(f"Evaluation export fields differ from manifest: {path}: {row['id']}")
    return {"passed": True, "manifest_id": identity, "samples": len(records), "images": len(manifest["images"]),
            "clusters": len(manifest["clusters"]), "cross_partition_near_pairs_checked": near_pairs_checked,
            "image_bytes_reverified": recompute_images, "exports_verified": len(report["exports"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--normalized-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--audit", action="store_true", help="Reverify existing frozen outputs and every image; do not change splits")
    parser.add_argument("--inspect-only", action="store_true", help="Write matching candidates and provisional counts for review before freezing")
    args = parser.parse_args()
    normalized_dir = args.normalized_dir or args.data_root / "hallucination/normalized"
    output_dir = args.output_dir or args.data_root / "hallucination/prepared"
    records, inputs = load_inputs(normalized_dir, args.data_root)
    if args.audit:
        audit = audit_dataset(records, inputs, args.data_root, output_dir)
        write_json(output_dir / "audit.json", audit)
        print(json.dumps(audit, indent=2))
        return
    if not 0 <= args.phash_distance <= 16:
        parser.error("--phash-distance must be in [0, 16]")
    policy = {
        "algorithm": "source-file-rgb-phash-connected-components-v1", "seed": args.seed,
        "amber_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
        "heldout_datasets": ["pope", "longhalqa"], "phash_distance": args.phash_distance,
        "aspect_ratio_log_tolerance": 0.02, "min_luminance_std": 2.0, "flat_image_rgb_rmse": 0.02,
    }
    images, clusters, exact_edges, candidates = group_images(records, args.data_root, policy)
    manifest = make_manifest(records, inputs, images, clusters, policy)
    if args.inspect_only:
        report = make_report(records, manifest, exact_edges, candidates, {})
        write_json(output_dir / "inspection/report.json", report)
        write_jsonl(output_dir / "inspection/image-matching-exact.jsonl", exact_edges)
        write_jsonl(output_dir / "inspection/image-matching-perceptual.jsonl", candidates)
        write_contact_sheets(candidates, args.data_root, output_dir / "inspection")
        print(json.dumps({"inspection": str(output_dir / "inspection/report.json"), "counts": report["counts"],
                          "perceptual_candidates": len(candidates), "perceptual_matches_merged": report["perceptual_matches_merged"]}, indent=2))
        return
    freeze_manifest(output_dir / "manifest.json", manifest)
    exports = export_rows(records, manifest, output_dir)
    report = make_report(records, manifest, exact_edges, candidates, exports)
    write_json(output_dir / "report.json", report)
    write_jsonl(output_dir / "image-matching-exact.jsonl", exact_edges)
    write_jsonl(output_dir / "image-matching-perceptual.jsonl", candidates)
    write_contact_sheets(candidates, args.data_root, output_dir / "matching-review")
    # Fast invariant verification here; --audit independently rereads/re-hashes
    # the actual images and scans all cross-partition perceptual candidates.
    audit = audit_dataset(records, inputs, args.data_root, output_dir, recompute_images=False)
    write_json(output_dir / "audit.json", audit)
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "manifest_id": manifest["manifest_id"],
                      "counts": report["counts"], "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
