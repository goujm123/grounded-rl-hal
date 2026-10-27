"""Validate rollout inputs and propagate a frozen image-grouped data split."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_text(value):
    return str(value).replace("<image>", "").strip()


class FrozenDataManifest:
    def __init__(self, path, data_root):
        self.path = Path(path).resolve()
        self.data_root = Path(data_root).resolve()
        self.sha256 = file_sha256(self.path)
        self.data = json.loads(self.path.read_text())
        if "manifest_id" in self.data:
            content = {key: value for key, value in self.data.items() if key != "manifest_id"}
            identity = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if self.data["manifest_id"] != identity:
                raise ValueError("Frozen manifest content hash does not match manifest_id")
        self.samples = self.data["samples"]
        self.images = self.data["images"]
        self._checked_images = set()
        if not isinstance(self.samples, dict) or not isinstance(self.images, dict):
            raise ValueError("Frozen manifest must contain samples/images mappings")

    def image_key(self, image):
        path = Path(image)
        if not path.is_absolute():
            path = self.data_root / path
        return path.resolve().relative_to(self.data_root).as_posix()

    def validate_sample(self, sample_id, image, question=None, answer=None, expected_split=None):
        sample_id = str(sample_id)
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id!r} is absent from frozen manifest")
        entry = self.samples[sample_id]
        image_key = self.image_key(image)
        if entry["image"] != image_key:
            raise ValueError(f"Image changed for {sample_id}: {image_key}")
        split = entry["split"]
        if expected_split and split != expected_split:
            raise ValueError(f"Sample {sample_id} belongs to {split}, expected {expected_split}")
        if question is not None and "question" in entry:
            if question_text(question) != question_text(entry["question"]):
                raise ValueError(f"Question changed for {sample_id}")
        if answer is not None and "answer" in entry:
            if str(answer).strip().casefold() != str(entry["answer"]).strip().casefold():
                raise ValueError(f"Ground-truth answer changed for {sample_id}")
        image_entry = self.images[image_key]
        if image_entry["split"] != split or image_entry["cluster_id"] != entry["cluster_id"]:
            raise ValueError(f"Inconsistent image-group assignment for {sample_id}")
        if image_key not in self._checked_images:
            if file_sha256(self.data_root / image_key) != image_entry["file_sha256"]:
                raise ValueError(f"Image checksum changed: {image_key}")
            self._checked_images.add(image_key)
        return {
            "data_split": split,
            "image_group": entry["cluster_id"],
            "split_manifest_sha256": self.sha256,
            "dataset": entry["dataset"],
            "task": entry["task"],
        }

    def validate_rollout(self, row):
        metadata = self.validate_sample(
            row.get("id"), row.get("image", ""), row.get("question"), row.get("true_answer")
        )
        for key in ("data_split", "image_group", "split_manifest_sha256"):
            if row.get(key) and row[key] != metadata[key]:
                raise ValueError(f"Rollout {row.get('id')} has incompatible {key}")
        return metadata


def attach_manifest(samples, manifest, expected_split=None):
    """Validate every input before sampling or submitting any inference request."""
    seen = set()
    splits = set()
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate input sample ID: {sample_id}")
        seen.add(sample_id)
        sample.update(manifest.validate_sample(
            sample_id, sample["image"], sample.get("input_query"), sample.get("true_answer"), expected_split
        ))
        splits.add(sample["data_split"])
    if len(splits) > 1:
        raise ValueError(f"A rollout run cannot mix data partitions: {sorted(splits)}")
    return samples


def tag_rollout(output, sample):
    output["id"] = sample["id"]
    for key in ("data_split", "image_group", "split_manifest_sha256", "dataset", "task",
                "review_overlay_id", "review_overlay_sha256"):
        if key in sample:
            output[key] = sample[key]
    return output
