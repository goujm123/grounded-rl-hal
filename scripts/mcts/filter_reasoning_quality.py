#!/usr/bin/env python3
"""Apply image-grounded, gold-blind semantic and point checks to reviewed SFT.

Outputs are new files. Source SFT, source rollouts, labels, and splits are never
rewritten. Cached model critiques are auditable automated decisions, not human gold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import threading

from PIL import Image

from vlmsearch.data_manifest import FrozenDataManifest, file_sha256
from vlmsearch.data_review import ReviewedTrainingData
from vlmsearch.image_geometry import validate_image_geometry
from vlmsearch.quality_verifier import (
    QualityVerifier, VerifierConfig, VERSION, atomic_json, content_hash, observations_and_points,
)
from vlmsearch.reward_funcs.answers import parse_final_answer


ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = {"discriminative-attribute-state": "state", "discriminative-attribute-number": "count",
              "discriminative-attribute-action": "action", "discriminative-hallucination": "existence",
              "discriminative-relation": "relation", "relation": "relation"}


def load_selection(path, frozen, review):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    if value.get("selection_id") != content_hash({key: item for key, item in value.items() if key != "selection_id"}):
        raise ValueError("Pilot selection checksum changed")
    if value["frozen_manifest"]["sha256"] != frozen.sha256 or value["review_overlay"]["sha256"] != review.sha256:
        raise ValueError("Pilot selection source/review binding differs")
    for sample in value["samples"]:
        frozen.validate_sample(sample["id"], sample["image"], sample["question"], sample["answer"], "train")
        review.validate_sample(sample["id"])
    return value


def get_assistant(entry):
    assistants = [m["content"] for m in entry["messages"] if m["role"] == "assistant"]
    questions = [m["content"] for m in entry["messages"] if m["role"] == "user"]
    if len(assistants) != 1 or len(questions) != 1:
        raise ValueError("Quality filter expects one complete question/assistant SFT pair")
    return questions[0], assistants[0]


def parallel_verdicts(jobs, verifier, workers, invoke):
    """Each thread owns a verifier; yield decisions in original input order."""
    if workers < 1:
        raise ValueError("Verifier workers must be positive")
    if workers == 1:
        for job in jobs:
            yield invoke(verifier, job)
        return
    local = threading.local()
    instances = []
    lock = threading.Lock()
    def run(job):
        if not hasattr(local, "verifier"):
            local.verifier = verifier.clone()
            with lock:
                instances.append(local.verifier)
        return invoke(local.verifier, job)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            yield from executor.map(run, jobs)
        finally:
            metrics = Counter()
            for instance in instances:
                metrics.update(instance.metrics)
            verifier.metrics = dict(metrics)


def _iter_rollouts(node):
    yield from node.get("rollouts", [])
    for child in node.get("children", []):
        yield from _iter_rollouts(child)


class SourceAdmission:
    def __init__(self, data_root, rollout_dir, manifest_path=None, review_path=None, selection_path=None):
        self.data_root = Path(data_root).resolve()
        self.rollout_dir = Path(rollout_dir).resolve()
        self.manifest = FrozenDataManifest(manifest_path or self.rollout_dir / "split_manifest.json", self.data_root)
        self.review = ReviewedTrainingData(review_path or self.rollout_dir / "training_review.json", self.manifest)
        self.selection = load_selection(selection_path, self.manifest, self.review)
        self.selected = {str(s["id"]) for s in self.selection["samples"]} if self.selection else None
        self.source_files = {}
        self.image_files = {}
        self._rows = {}
        spec = importlib.util.spec_from_file_location("quality_source_linearizer", ROOT / "scripts/mcts/build_reasoning_chains_from_mcts.py")
        self.linearizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.linearizer)
        self._rebuilt = {}

    def validate(self, entry):
        metadata = entry["metadata"]
        sample_id = str(metadata["sample_key"])
        if self.selected is not None and sample_id not in self.selected:
            raise ValueError("SFT candidate is outside the frozen pilot selection")
        expected_review = self.review.validate_sample(sample_id)
        for key, expected in expected_review.items():
            if metadata.get(key) != expected:
                raise ValueError("SFT review fingerprint mismatch")
        if metadata.get("coordinate_frame") != "original_pixels" or metadata.get("data_split") != "train":
            raise ValueError("Quality filter requires original-pixel reviewed TRAIN candidates")
        source = Path(metadata["source_file"]).resolve()
        if not source.is_relative_to(self.rollout_dir):
            raise ValueError("SFT source rollout is outside the declared run directory")
        if source not in self._rows:
            self.source_files[str(source)] = file_sha256(source)
            self._rows[source] = {index: json.loads(line) for index, line in enumerate(source.read_text().splitlines(), 1) if line.strip()}
        line = metadata["source_line"]
        if not isinstance(line, int) or isinstance(line, bool) or line not in self._rows[source]:
            raise ValueError("Invalid SFT source line")
        row = self._rows[source][line]
        if str(row["id"]) != sample_id:
            raise ValueError("SFT source sample ID mismatch")
        bound = self.manifest.validate_rollout(row)
        for key in ("split_manifest_sha256", "data_split", "image_group", "dataset", "task"):
            if metadata.get(key) != bound.get(key):
                raise ValueError(f"SFT source {key} mismatch")
        for key, expected in expected_review.items():
            if row.get(key) != expected:
                raise ValueError("Source rollout review fingerprint mismatch")
        if not isinstance(entry["images"], list) or len(entry["images"]) != 1:
            raise ValueError("Quality filter expects exactly one original image")
        question, chain = get_assistant(entry)
        sample = self.manifest.samples[sample_id]
        self.manifest.validate_sample(sample_id, entry["images"][0], question, entry.get("gt_answer"), "train")
        if parse_final_answer(chain, {"yes", "no"}) != sample["answer"]:
            raise ValueError("SFT candidate label differs from its original supervised source")
        geometry = row.get("image_geometry")
        if metadata.get("source_image_geometry") != geometry or metadata.get("image_geometry") != geometry:
            raise ValueError("SFT source image geometry mismatch")
        image_path = self.data_root / sample["image"]
        self.image_files[str(image_path)] = self.manifest.images[sample["image"]]["file_sha256"]
        with Image.open(image_path) as source_image:
            size = source_image.size
        validate_image_geometry(geometry, size)
        if metadata.get("original_image_size") != list(size):
            raise ValueError("SFT original image size mismatch")
        rollouts = list(_iter_rollouts(row["tree"]))
        index = metadata["rollout_idx"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(rollouts):
            raise ValueError("Invalid source rollout index")
        if rollouts[index].get("reward", 0) < 1:
            raise ValueError("SFT candidate does not originate from a positive source rollout")
        rebuild_key = (source, line)
        if rebuild_key not in self._rebuilt:
            self._rebuilt[rebuild_key] = self.linearizer.build_all_reasoning_for_sample(
                row, require_coordinate=False, original_size=size)[0]
        exact = [item for item in self._rebuilt[rebuild_key] if item["rollout_idx"] == index and item["chain"] == chain]
        if len(exact) != 1:
            raise ValueError("SFT candidate does not reproduce from its exact source rollout/normalization")
        if metadata.get("evidence_normalization") != exact[0].get("evidence_normalization"):
            raise ValueError("SFT evidence normalization provenance mismatch")
        return {"sample_id": sample_id, "question": sample["question"], "candidate_chain": chain,
                "image_path": image_path, "image_sha256": self.manifest.images[sample["image"]]["file_sha256"],
                "original_size": list(size), "image_geometry": geometry,
                "cell": CATEGORIES[sample["category"]] + ":" + sample["answer"],
                "source_fingerprints": {"split_manifest_sha256": self.manifest.sha256,
                                        "review_overlay_sha256": self.review.sha256,
                                        "source_rollout_sha256": self.source_files[str(source)],
                                        "source_row_sha256": content_hash(row), "sft_record_sha256": content_hash(entry)}}


def filter_sft(input_dir, output_dir, admission, verifier, workers=1):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or output_dir.exists():
        raise ValueError("Quality output directory must be new and different from source SFT")
    inputs = {}
    entries = []
    for split in ("train", "val", "test"):
        path = input_dir / f"reasoning_chains_{split}.json"
        value = json.loads(path.read_text())
        if not isinstance(value, list) or (split != "train" and value):
            raise ValueError("Reviewed training quality filter cannot admit held-out SFT entries")
        inputs[str(path)] = file_sha256(path)
        entries.extend(value)
    # Validate the whole input before any model work or publishing output files.
    candidates = [admission.validate(entry) for entry in entries]
    output_dir.mkdir(parents=True, exist_ok=False)
    kept = []
    cells = defaultdict(Counter)
    ledger = []
    seen_chains = set()
    ledger_file = output_dir / "quality_decisions.jsonl"
    with ledger_file.open("w") as handle:
        def invoke(worker, candidate):
            return worker.verify_candidate(
                image_path=candidate["image_path"], question=candidate["question"],
                candidate_chain=candidate["candidate_chain"], expected_image_sha256=candidate["image_sha256"],
                original_size=candidate["original_size"], image_geometry=candidate["image_geometry"],
                source_fingerprints=candidate["source_fingerprints"])
        verdicts = parallel_verdicts(candidates, verifier, workers, invoke)
        for index, result in enumerate(verdicts):
            entry, candidate = entries[index], candidates[index]
            cells[candidate["cell"]]["input_chains"] += 1
            identity = (candidate["sample_id"], result["binding"]["candidate_sha256"])
            accepted = result["accepted"] and identity not in seen_chains
            decision = {"entry_index": index, "sft_entry_id": entry.get("id"), "sample_id": candidate["sample_id"],
                        "cell": candidate["cell"], "accepted": accepted,
                        "duplicate_exact_chain": identity in seen_chains, "verdict": result}
            if accepted:
                seen_chains.add(identity)
                filtered = deepcopy(entry)
                filtered["metadata"]["quality_status"] = "accepted_by_automated_visual_verifier"
                filtered["metadata"]["quality_filter"] = {"version": VERSION, "verifier_fingerprint": verifier.fingerprint,
                                                          "decision_sha256": result["decision_sha256"],
                                                          "ledger_entry_index": index,
                                                          "candidate_sha256": result["binding"]["candidate_sha256"]}
                kept.append(filtered)
                cells[candidate["cell"]]["accepted_chains"] += 1
            else:
                cells[candidate["cell"]]["rejected_chains"] += 1
            if not result.get("verification_complete"):
                cells[candidate["cell"]]["verifier_errors"] += 1
            elif result["semantic_accept"] is False:
                cells[candidate["cell"]]["semantic_rejected_chains"] += 1
            ledger.append(decision)
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"processed": index + 1, "total": len(entries), "accepted": len(kept),
                              "sample_id": candidate["sample_id"], "reasons": result["reasons"]}), flush=True)
    bindings = {**inputs, **admission.source_files, **admission.image_files,
                str(admission.manifest.path): admission.manifest.sha256,
                str(admission.review.path): admission.review.sha256}
    for path, expected in bindings.items():
        if file_sha256(path) != expected:
            raise ValueError("A source artifact changed during verification; no completion manifest published")
    for split, value in (("train", kept), ("val", []), ("test", [])):
        atomic_json(output_dir / f"reasoning_chains_{split}.json", value)
    reasons = Counter(reason for decision in ledger for reason in decision["verdict"]["reasons"])
    report = {"schema_version": 1, "filter_version": VERSION, "verifier_fingerprint": verifier.fingerprint,
              "verifier_config": asdict(verifier.config),
              "model_identity": verifier.model_identity, "source_sft_sha256": inputs,
              "source_rollout_sha256": admission.source_files,
              "original_image_sha256": admission.image_files,
              "split_manifest_sha256": admission.manifest.sha256, "review_overlay_id": admission.review.id,
              "review_overlay_sha256": admission.review.sha256,
              "selection_id": admission.selection["selection_id"] if admission.selection else None,
              "input_chains": len(entries), "accepted_chains": len(kept), "rejected_chains": len(entries) - len(kept),
              "verifier_errors": sum(not d["verdict"].get("verification_complete") for d in ledger),
              "semantic_rejected_chains": sum(d["verdict"]["semantic_accept"] is False for d in ledger),
              "per_cell": {cell: dict(counts) for cell, counts in cells.items()},
              "reasons": dict(reasons), "model_metrics": verifier.metrics,
              "limitation": "Automated same-model critique can share generator errors; acceptance is not guaranteed visual correctness.",
              "outputs": {p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()}}
    report["manifest_id"] = content_hash(report)
    atomic_json(output_dir / "quality_manifest.json", report)
    return report


def verify_inputs(inputs_path, output_dir, data_root, verifier, reference_path=None, workers=1):
    """Calibrate via reference-free inputs; references are joined only after calls."""
    inputs_path, output_dir = Path(inputs_path), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Calibration output directory must be new")
    records = [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    input_sha = file_sha256(inputs_path)
    example_ids = set()
    for record in records:
        ci = record["critic_input"]
        if record["example_id"] in example_ids:
            raise ValueError("Calibration inputs contain duplicate example IDs")
        example_ids.add(record["example_id"])
        chain_hash = hashlib.sha256(ci["candidate_chain"].encode()).hexdigest()
        if record.get("chain_sha256") != chain_hash or ci.get("coordinate_frame") != "original_pixels":
            raise ValueError("Calibration candidate checksum/frame changed")
        path = Path(data_root) / ci["image"]
        if file_sha256(path) != ci["image_sha256"]:
            raise ValueError("Calibration original image checksum changed")
        with Image.open(path) as image:
            if list(image.size) != ci["original_size"]:
                raise ValueError("Calibration original image dimensions changed")
        _, points, _ = observations_and_points(ci["candidate_chain"])
        for point in points:
            if any(not 0 <= coordinate <= bound - 1 for coordinate, bound in zip(point["xy"], ci["original_size"])):
                raise ValueError("Calibration candidate point is out of original-image bounds")
    output_dir.mkdir(parents=True, exist_ok=False)
    decisions = []
    with (output_dir / "quality_decisions.jsonl").open("w") as handle:
        def invoke(worker, record):
            ci = record["critic_input"]
            return worker.verify_candidate(
                Path(data_root) / ci["image"], ci["question"], ci["candidate_chain"], ci["image_sha256"],
                original_size=ci["original_size"],
                source_fingerprints={"critic_inputs_sha256": input_sha})
        verdicts = parallel_verdicts(records, verifier, workers, invoke)
        for index, result in enumerate(verdicts):
            record = records[index]
            decision = {"example_id": record["example_id"], "chain_sha256": record["chain_sha256"], "verdict": result}
            decisions.append(decision)
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"example_id": record["example_id"], "accepted": result["accepted"],
                              "semantic_accept": result["semantic_accept"], "reasons": result["reasons"]}), flush=True)
    summary = {"schema_version": 1, "verifier_fingerprint": verifier.fingerprint,
               "verifier_config": asdict(verifier.config),
               "critic_inputs_sha256": input_sha, "examples": len(decisions),
               "accepted": sum(d["verdict"]["accepted"] for d in decisions),
               "semantic_accept": sum(d["verdict"]["semantic_accept"] is True for d in decisions),
               "verifier_errors": sum(not d["verdict"].get("verification_complete") for d in decisions),
               "model_metrics": verifier.metrics, "model_identity": verifier.model_identity}
    if reference_path:
        fixture = json.loads(Path(reference_path).read_text())
        reference = {e["example_id"]: e for e in fixture["examples"]}
        confusion = Counter()
        for decision in decisions:
            known = reference[decision["example_id"]]
            if known["chain_sha256"] != decision["chain_sha256"]:
                raise ValueError("Calibration reference chain differs from verified candidate")
            expected = known["automated_reference"]["expected_decision"]
            actual = ("error" if not decision["verdict"].get("verification_complete") else
                      "pass" if decision["verdict"]["semantic_accept"] else "reject")
            confusion[expected + "/" + actual] += 1
        summary["reference_fixture_sha256"] = file_sha256(reference_path)
        summary["semantic_confusion_expected_actual"] = dict(confusion)
        summary["reference_limitation"] = "Previously automated visual review, not independent human gold; no reference fields were passed to the critic."
    summary["summary_id"] = content_hash(summary)
    atomic_json(output_dir / "calibration_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--input-dir", help="Source complete SFT exports to filter")
    modes.add_argument("--critic-inputs", help="Reference-free calibration JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rollout-dir")
    parser.add_argument("--split-manifest")
    parser.add_argument("--review-overlay")
    parser.add_argument("--selection")
    parser.add_argument("--reference-fixture", help="Calibration references joined only after verification")
    parser.add_argument("--base-url", default="http://127.0.0.1:9011/v1")
    parser.add_argument("--api-key", default="qwen")
    parser.add_argument("--model", default="qwen_vllm")
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.8-27B")
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--retry-max-tokens", type=int, default=4608,
                        help="Explicit output cap for the single protocol-error retry; failures remain rejected")
    parser.add_argument("--workers", type=int, default=4, help="Independent verifier clients sharing the locked cache")
    parser.add_argument("--allow-repetition", action="store_true", help="Calibration-only separation of semantic quality from repetition; still reported")
    args = parser.parse_args()
    config = VerifierConfig(base_url=args.base_url, model=args.model, model_path=args.model_path,
                            max_tokens=args.max_tokens, retry_max_tokens=args.retry_max_tokens,
                            reject_repetition=not args.allow_repetition)
    verifier = QualityVerifier(config, args.cache_dir, api_key=args.api_key)
    if args.critic_inputs:
        summary = verify_inputs(args.critic_inputs, args.output_dir, args.data_root, verifier, args.reference_fixture, args.workers)
    else:
        if not args.rollout_dir:
            parser.error("--input-dir requires --rollout-dir")
        if args.allow_repetition:
            parser.error("Production SFT filtering always rejects repetition")
        admission = SourceAdmission(args.data_root, args.rollout_dir, args.split_manifest, args.review_overlay, args.selection)
        summary = filter_sft(args.input_dir, args.output_dir, admission, verifier, args.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
