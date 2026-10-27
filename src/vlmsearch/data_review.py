"""Apply a hashed, train-only annotation review without rewriting gold labels."""

import hashlib
import json
from pathlib import Path
import re

from vlmsearch.data_manifest import file_sha256


def content_id(value, field="overlay_id"):
    content = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def amber_cross_task_candidates(data_root):
    raw = Path(data_root) / "hallucination/raw/amber"
    annotation_path = raw / "annotations.json"
    query_path = raw / "query/query_discriminative.json"
    if not annotation_path.is_file() or not query_path.is_file():
        return set()
    annotations = json.loads(annotation_path.read_text())
    by_id = {str(row["id"]): row for row in annotations}
    present = {f"AMBER_{row['id']}.jpg": {word.casefold() for word in row["truth"]}
               for row in annotations if row["type"] == "generative"}
    candidates = set()
    for query in json.loads(query_path.read_text()):
        label = by_id[str(query["id"])]
        match = re.fullmatch(r"Is there a (.+) in this image\?", query["query"])
        if (match and label["type"] == "discriminative-hallucination" and label["truth"] == "no"
                and match.group(1).casefold() in present[Path(query["image"]).name]):
            candidates.add(f"amber:{query['id']}")
    return candidates


class ReviewedTrainingData:
    def __init__(self, path, frozen_manifest, required_candidate_ids=None):
        self.path = Path(path).resolve()
        self.manifest = frozen_manifest
        self.sha256 = file_sha256(self.path)
        self.data = json.loads(self.path.read_text())
        self.id = self.data.get("overlay_id")
        if self.data.get("schema_version") != 1 or self.id != content_id(self.data):
            raise ValueError("Training review overlay content hash/schema is invalid")
        binding = self.data.get("frozen_manifest", {})
        if (binding.get("sha256") != frozen_manifest.sha256
                or binding.get("manifest_id") != frozen_manifest.data.get("manifest_id")):
            raise ValueError("Training review belongs to a different frozen manifest")
        entries = self.data.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Training review entries must be a list")
        self.entries = {}
        for entry in entries:
            sample_id = str(entry["id"])
            if sample_id in self.entries:
                raise ValueError(f"Duplicate training review ID: {sample_id}")
            if entry.get("split") != "train" or entry.get("decision") not in {"retain", "exclude"}:
                raise ValueError(f"Invalid train-only review decision: {sample_id}")
            metadata = frozen_manifest.validate_sample(sample_id, entry["image"], entry["question"],
                                                       entry["original_label"], expected_split="train")
            source = frozen_manifest.samples[sample_id]
            image = frozen_manifest.images[source["image"]]
            if (metadata["dataset"] != "amber" or entry.get("category") != source.get("category")
                    or entry.get("image_sha256") != image["file_sha256"]):
                raise ValueError(f"Training review provenance changed: {sample_id}")
            if not entry.get("rationale") or not entry.get("confidence"):
                raise ValueError(f"Training review lacks a rationale/confidence: {sample_id}")
            self.entries[sample_id] = entry
        self.excluded_ids = {key for key, entry in self.entries.items() if entry["decision"] == "exclude"}
        retained = {key for key, entry in self.entries.items() if entry["decision"] == "retain"}
        for field, expected in [("excluded_ids", self.excluded_ids), ("retained_reviewed_ids", retained)]:
            declared = self.data.get(field, [])
            if not isinstance(declared, list) or len(declared) != len(set(declared)) or set(declared) != expected:
                raise ValueError(f"Training review {field} does not match its decisions")
        required = amber_cross_task_candidates(frozen_manifest.data_root) if required_candidate_ids is None else set(required_candidate_ids)
        required = {key for key in required if frozen_manifest.samples.get(key, {}).get("split") == "train"}
        missing = required - self.entries.keys()
        if missing:
            raise ValueError(f"Training review is incomplete for {len(missing)} flagged training records")
        for record in self.data.get("source_files", {}).values():
            if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                raise ValueError("Training review source_files must contain path/sha256 records")
            source_path = Path(record["path"])
            if not source_path.is_absolute():
                choices = [frozen_manifest.data_root / source_path,
                           frozen_manifest.data_root.parent / source_path, self.path.parent / source_path]
                source_path = next((choice for choice in choices if choice.is_file()), choices[0])
            if not source_path.is_file() or file_sha256(source_path) != record["sha256"]:
                raise ValueError(f"Training review source checksum changed: {record['path']}")

    def validate_sample(self, sample_id):
        sample_id = str(sample_id)
        entry = self.manifest.samples.get(sample_id, {})
        if entry.get("split") != "train" or entry.get("dataset") != "amber":
            raise ValueError(f"Training review cannot admit held-out/non-AMBER sample: {sample_id}")
        if sample_id in self.excluded_ids:
            raise ValueError(f"Training review excluded sample {sample_id}")
        return {"review_overlay_id": self.id, "review_overlay_sha256": self.sha256}


def attach_review(samples, review):
    admitted = []
    for sample in samples:
        if str(sample["id"]) in review.excluded_ids:
            continue
        sample.update(review.validate_sample(sample["id"]))
        admitted.append(sample)
    return admitted
