#!/usr/bin/env python3
"""Validate and merge completed visual-quality shards without model calls.

The parent SFT candidates, raw rollouts, image bytes, training review and pilot
selection remain authoritative. A completion manifest is required for every
shard; interrupted or mismatched shards never produce a merged completion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import (
    atomic_json, content_hash, observations_and_points, strict_json, validate_inventory, validate_schema,
    fine_attribute_targets, mask_coordinate_observations, point_attachment,
    observability_risk_targets,
)


ROOT = Path(__file__).resolve().parents[2]
MERGE_VERSION = "quality_shards_merge_v1"
SPLITS = ("train", "val", "test")
LEGACY_STAGES = ("blind_answer", "scene_claims", "blind_centers", "point_referents")
STAGES = ("blind_answer", "scene_claims", "fine_attributes", "blind_centers", "point_referents")


def policy_stages(version):
    if version == "image_claim_point_critic_v1":
        return LEGACY_STAGES
    if version in {"image_claim_point_critic_v2", "image_claim_point_critic_v3"}:
        return STAGES
    raise ValueError("Unsupported critic policy version")


def checked_json(path, checksum_field=None, expected=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing completed artifact: {path}")
    if expected is not None and file_sha256(path) != expected:
        raise ValueError(f"Artifact checksum changed: {path}")
    value = json.loads(path.read_text())
    if checksum_field and value.get(checksum_field) != content_hash({key: item for key, item in value.items() if key != checksum_field}):
        raise ValueError(f"Invalid {checksum_field}: {path}")
    return value


def checked_bindings(values):
    if not isinstance(values, dict):
        raise ValueError("Artifact bindings must be a path/SHA256 mapping")
    for path, expected in values.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Bound source checksum changed: {path}")


def source_filter_module():
    spec = importlib.util.spec_from_file_location("merge_quality_source_filter", ROOT / "scripts/mcts/filter_reasoning_quality.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sft_inputs(directory):
    values, checksums = {}, {}
    for split in SPLITS:
        path = Path(directory).resolve() / f"reasoning_chains_{split}.json"
        value = checked_json(path)
        if not isinstance(value, list) or (split != "train" and value):
            raise ValueError("Quality merge admits TRAIN candidates only; val/test must be empty")
        values[split] = value
        checksums[str(path)] = file_sha256(path)
    return values["train"], checksums


def chain_identity(entry, get_assistant):
    _, chain = get_assistant(entry)
    return str(entry["metadata"]["sample_key"]), hashlib.sha256(chain.encode()).hexdigest()


def assert_equal(actual, expected, reason):
    if actual != expected:
        raise ValueError(reason)


class CriticRecordAudit:
    """Check cached pass/attempt records using their self-contained schemas."""
    def __init__(self):
        self.files = {}
        self.prompts = {}
        self.retry_instruction = None

    def _binding(self, binding, manifest):
        stage = binding["stage"]
        if stage not in STAGES:
            raise ValueError("Unknown critic stage in cache binding")
        expected_fingerprint = content_hash({"identity": binding["stage_identity"], "stage": stage,
                                             "system": binding["system"], "schema": binding["schema"]})
        assert_equal(binding["fingerprint"], expected_fingerprint, "Critic stage fingerprint changed")
        identity = binding["stage_identity"]
        assert_equal(identity["version"], manifest["filter_version"], "Critic stage version mismatch")
        inference = {key: value for key, value in manifest["verifier_config"].items() if key != "reject_repetition"}
        assert_equal(identity["inference_config"], inference, "Critic stage inference settings mismatch")
        assert_equal(identity["checkpoint"], manifest["model_identity"], "Critic stage model identity mismatch")
        if stage in self.prompts and self.prompts[stage] != binding["system"]:
            raise ValueError("Quality shards used different critic prompts")
        self.prompts[stage] = binding["system"]
        if self.retry_instruction is not None and self.retry_instruction != identity["retry_instruction"]:
            raise ValueError("Quality shards used different retry instructions")
        self.retry_instruction = identity["retry_instruction"]
        return stage

    def attempt(self, reference, manifest):
        path = Path(reference["file"])
        record = checked_json(path, "record_sha256")
        assert_equal(record["record_sha256"], reference["record_sha256"], "Critic attempt reference checksum mismatch")
        self._binding(record["binding"], manifest)
        self.files[str(path.resolve())] = file_sha256(path)
        if record.get("error") is None:
            if record.get("finish_reason") != "stop":
                raise ValueError("Successful critic attempt did not finish normally")
            value = strict_json(record["raw_response"])
            validate_schema(value, record["binding"]["schema"])
            validate_inventory(record["binding"]["stage"], value, record["binding"]["payload"])

    def record(self, reference, manifest):
        path = Path(reference["cache_file"])
        record = checked_json(path, "record_sha256")
        assert_equal(record["record_sha256"], reference["record_sha256"], "Critic cache reference checksum mismatch")
        binding = record["binding"]
        stage = self._binding(binding, manifest)
        assert_equal(content_hash(binding), reference["cache_key"], "Critic cache key mismatch")
        assert_equal(binding["fingerprint"], reference["stage_fingerprint"], "Critic cached stage fingerprint mismatch")
        value = strict_json(record["raw_response"])
        assert_equal(value, record["parsed_response"], "Critic parsed/raw response mismatch")
        validate_schema(value, binding["schema"])
        validate_inventory(stage, value, binding["payload"])
        for attempt in record.get("attempt_records", []):
            self.attempt(attempt, manifest)
        self.files[str(path.resolve())] = file_sha256(path)
        return stage, value

    def verify_fingerprint(self, manifest):
        stages = policy_stages(manifest["filter_version"])
        if all(stage in self.prompts for stage in stages):
            expected = content_hash({"version": manifest["filter_version"], "config": manifest["verifier_config"],
                                     "checkpoint": manifest["model_identity"],
                                     "prompts": [self.prompts[stage] for stage in stages] + [self.retry_instruction]})
            assert_equal(expected, manifest["verifier_fingerprint"], "Critic manifest fingerprint does not match recorded prompts/settings")


def validate_verdict(decision, candidate, manifest, audit):
    verdict = decision["verdict"]
    expected_decision = content_hash({key: value for key, value in verdict.items() if key != "decision_sha256"})
    assert_equal(verdict.get("decision_sha256"), expected_decision, "Candidate decision SHA256 mismatch")
    expected_binding = {"candidate_sha256": hashlib.sha256(candidate["candidate_chain"].encode()).hexdigest(),
                        "image_sha256": candidate["image_sha256"],
                        "question_sha256": hashlib.sha256(candidate["question"].encode()).hexdigest(),
                        "geometry_sha256": content_hash(candidate["image_geometry"]) if candidate["image_geometry"] is not None else None,
                        "source_fingerprints": candidate["source_fingerprints"],
                        "verifier_fingerprint": manifest["verifier_fingerprint"]}
    assert_equal(verdict.get("binding"), expected_binding, "Candidate/source/critic binding mismatch")
    assert_equal(verdict.get("version"), manifest["filter_version"], "Candidate filter version mismatch")
    if not isinstance(verdict.get("accepted"), bool) or not isinstance(verdict.get("verification_complete"), bool):
        raise ValueError("Invalid candidate decision booleans")
    if not isinstance(verdict.get("reasons"), list) or any(not isinstance(reason, str) for reason in verdict["reasons"]):
        raise ValueError("Invalid candidate decision reasons")
    complete = verdict["verification_complete"]
    if complete:
        if verdict.get("status") != "verified" or not isinstance(verdict.get("semantic_accept"), bool):
            raise ValueError("Completed critic decision lacks semantic status")
        if verdict["accepted"] != (not verdict["reasons"]):
            raise ValueError("Accepted decision contradicts its rejection reasons")
        if verdict["accepted"] and verdict["semantic_accept"] is not True:
            raise ValueError("Accepted decision is not semantically accepted")
    elif verdict.get("status") != "verifier_error" or verdict.get("semantic_accept") is not None or verdict["accepted"]:
        raise ValueError("Incomplete critic decision was not kept separate from semantic decisions")
    if decision.get("accepted") is not verdict["accepted"] or decision.get("duplicate_exact_chain") is not False:
        raise ValueError("Shard decision duplicates or acceptance do not match its unique candidate")
    observations, points, _ = observations_and_points(candidate["candidate_chain"])
    stages = policy_stages(manifest["filter_version"])
    masked, _ = mask_coordinate_observations(observations)
    fine_targets = fine_attribute_targets(masked, include_color=manifest["filter_version"] == "image_claim_point_critic_v3") if "fine_attributes" in stages else []
    if "fine_attributes" in stages:
        assert_equal(verdict.get("fine_attribute_targets"), fine_targets, "Fine-attribute source inventory changed")
    if manifest["filter_version"] == "image_claim_point_critic_v3":
        assert_equal(verdict.get("observability_risk_targets"), observability_risk_targets(masked),
                     "Observability source inventory changed")
    assert_equal(verdict.get("point_count"), len(points), "Candidate point inventory mismatch")
    records = [audit.record(reference, manifest) for reference in verdict.get("cache_records", [])]
    for reference in verdict.get("failed_attempt_records", []):
        audit.attempt(reference, manifest)
    passes = verdict.get("passes", {})
    for stage in stages[:2]:
        if stage in passes and (stage, passes[stage]) not in records:
            raise ValueError("Critic semantic pass does not match a bound cache record")
    if complete:
        if [stage for stage, _ in records[:2]] != list(stages[:2]):
            raise ValueError("Completed critic decision lacks blind/scene records")
        if any(stage not in passes for stage in stages[:2]):
            raise ValueError("Completed critic decision lacks semantic passes")
        batches = passes.get("localization_batches", [])
        point_ids = [point_id for batch in batches for point_id in batch["point_ids"]]
        assert_equal(sorted(point_ids), list(range(1, len(points) + 1)), "Completed critic decision did not cover every point exactly once")
        expected_records = list(zip(stages[:2], [passes[stage] for stage in stages[:2]]))
        if manifest["filter_version"] == "image_claim_point_critic_v3" and verdict["accepted"]:
            risks = {item["risk_target_id"]: item for item in verdict["observability_risk_targets"]}
            for observation in passes["scene_claims"]["observations"]:
                for check in observation["observability_checks"]:
                    selected = [risks[key] for key in check["risk_target_ids"]]
                    if (any(item.get("requires_visible_scale_reference") for item in selected)
                            and check.get("scale_reference_kind") not in {"visible_calibrated_ruler_or_grid", "readable_explicit_dimension_label"}):
                        raise ValueError("Accepted physical measurement lacks its required visible scale")
                    if (any(item.get("requires_independent_motion_evidence") for item in selected)
                            and check.get("motion_evidence_kind") != "readable_temporal_measurement_or_motion_annotation"):
                        raise ValueError("Accepted motion inference relies on blur or pose alone")
        if "fine_attributes" in stages:
            if fine_targets:
                if "fine_attributes" not in passes:
                    raise ValueError("Completed critic decision lacks fine-attribute pass")
                expected_records.append(("fine_attributes", passes["fine_attributes"]))
            else:
                assert_equal(passes.get("fine_attributes"), {"attributes": []}, "Empty fine-attribute pass changed")
            if verdict["accepted"]:
                targets = {item["target_id"]: item for item in fine_targets}
                for attribute in passes["fine_attributes"]["attributes"]:
                    target = targets[attribute["target_id"]]
                    if (target.get("polarity_uncertain") or attribute["clarity"] != "clear" or attribute["observed_value"] in {"unclear", "lying_unspecified"}
                            or ((attribute["observed_value"] == target["expected_value"]) == target["negated"])):
                        raise ValueError("Accepted decision contradicts its independent fine-attribute check")
        for batch in batches:
            expected_records.extend([("blind_centers", batch["blind_centers"]), ("point_referents", batch["referents"])])
            if "fine_attributes" in stages and verdict["accepted"]:
                by_point = {point["point_id"]: point for point in points}
                for referent in batch["referents"]["points"]:
                    source = by_point[referent["point_id"]]
                    attachment = point_attachment(masked[source["observation_id"] - 1], source["point_id"])
                    if (referent.get("checked_attachment_quote") != attachment["attachment_quote"]
                            or referent.get("attachment_scope_preserved") is not True
                            or any(not re.search(r"\b" + re.escape(term) + r"\b", referent["referent_quote"], re.I)
                                   for term in attachment["required_feature_terms"])):
                        raise ValueError("Accepted point changed its source attachment scope")
        assert_equal(records, expected_records, "Critic point/scene passes differ from their cache records")
    for key, value in verdict.get("metrics", {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid critic metric: {key}")


def merge_quality_shards(parent_sft, selection_path, shard_dirs, output_dir, data_root, rollout_dir=None):
    parent_sft, output_dir = Path(parent_sft).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Merged quality output must be a new directory")
    if len(shard_dirs) < 2 or len({str(Path(path).resolve()) for path in shard_dirs}) != len(shard_dirs):
        raise ValueError("At least two distinct completed quality shards are required")
    filter_module = source_filter_module()
    admission = filter_module.SourceAdmission(data_root, rollout_dir or parent_sft.parent, selection_path=selection_path)
    parent_entries, parent_inputs = sft_inputs(parent_sft)
    parent = {}
    for index, entry in enumerate(parent_entries):
        identity = chain_identity(entry, filter_module.get_assistant)
        if identity in parent:
            raise ValueError("Parent candidates contain a duplicate sample/chain identity")
        parent[identity] = {"index": index, "entry": entry, "candidate": admission.validate(entry)}
    if not parent:
        raise ValueError("Parent SFT contains no candidates")
    seen, merged_by_parent, kept_by_parent = set(), {}, {}
    shard_reports = []
    shard_input_bindings = {}
    model_metrics = Counter()
    audit = CriticRecordAudit()
    policy_base = None
    source_bindings = {**parent_inputs, **admission.source_files, **admission.image_files,
                       str(admission.manifest.path): admission.manifest.sha256,
                       str(admission.review.path): admission.review.sha256,
                       str(Path(selection_path).resolve()): file_sha256(selection_path)}
    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir).resolve()
        manifest_path = shard_dir / "quality_manifest.json"
        manifest = checked_json(manifest_path, "manifest_id")
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported quality completion manifest schema")
        assert_equal(manifest.get("split_manifest_sha256"), admission.manifest.sha256, "Shard split manifest mismatch")
        assert_equal(manifest.get("review_overlay_id"), admission.review.id, "Shard review identity mismatch")
        assert_equal(manifest.get("review_overlay_sha256"), admission.review.sha256, "Shard review hash mismatch")
        assert_equal(manifest.get("selection_id"), admission.selection["selection_id"], "Shard pilot selection mismatch")
        policy = {"filter_version": manifest["filter_version"], "model_identity": manifest["model_identity"],
                  "verifier_config": {key: value for key, value in manifest["verifier_config"].items() if key != "base_url"}}
        if policy_base is not None and policy != policy_base:
            raise ValueError("Quality shards used different critic policy/model settings")
        policy_base = policy
        outputs = manifest.get("outputs", {})
        required_outputs = {"quality_decisions.jsonl", *(f"reasoning_chains_{split}.json" for split in SPLITS)}
        if not required_outputs <= outputs.keys():
            raise ValueError("Quality completion manifest lacks required outputs")
        for name, expected in outputs.items():
            if Path(name).name != name:
                raise ValueError("Quality output binding must be a simple filename")
            path = shard_dir / name
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Quality shard output checksum changed: {path}")
            source_bindings[str(path)] = expected
        source_bindings[str(manifest_path)] = file_sha256(manifest_path)
        checked_bindings(manifest["source_sft_sha256"])
        checked_bindings(manifest["source_rollout_sha256"])
        checked_bindings(manifest["original_image_sha256"])
        input_paths = {Path(path).name: Path(path) for path in manifest["source_sft_sha256"]}
        if set(input_paths) != {f"reasoning_chains_{split}.json" for split in SPLITS} or len(input_paths) != len(manifest["source_sft_sha256"]):
            raise ValueError("Quality shard must bind exactly one TRAIN/val/test input set")
        input_entries = checked_json(input_paths["reasoning_chains_train.json"])
        for split in ("val", "test"):
            assert_equal(checked_json(input_paths[f"reasoning_chains_{split}.json"]), [], "Shard input contains held-out data")
            assert_equal(checked_json(shard_dir / f"reasoning_chains_{split}.json"), [], "Shard output contains held-out data")
        ledger = [json.loads(line) for line in (shard_dir / "quality_decisions.jsonl").read_text().splitlines() if line.strip()]
        if len(ledger) != len(input_entries) or len(ledger) != manifest["input_chains"]:
            raise ValueError("Quality shard decision coverage is incomplete")
        accepted_entries = checked_json(shard_dir / "reasoning_chains_train.json")
        expected_accepted = []
        local_cells = defaultdict(Counter)
        expected_rollouts, expected_images = {}, {}
        metrics = Counter()
        for index, (entry, decision) in enumerate(zip(input_entries, ledger)):
            identity = chain_identity(entry, filter_module.get_assistant)
            if identity not in parent or identity in seen:
                raise ValueError("Quality shards contain an unknown or duplicate sample/chain identity")
            original = parent[identity]
            assert_equal(entry, original["entry"], "Shard input candidate differs from its parent SFT record")
            candidate = original["candidate"]
            assert_equal(decision.get("entry_index"), index, "Quality ledger input order/index mismatch")
            assert_equal(decision.get("sft_entry_id"), entry.get("id"), "Quality ledger SFT ID mismatch")
            assert_equal(decision.get("sample_id"), identity[0], "Quality ledger sample ID mismatch")
            assert_equal(decision.get("cell"), candidate["cell"], "Quality ledger category/label mismatch")
            validate_verdict(decision, candidate, manifest, audit)
            expected_rollouts[str(Path(entry["metadata"]["source_file"]).resolve())] = candidate["source_fingerprints"]["source_rollout_sha256"]
            expected_images[str(candidate["image_path"].resolve())] = candidate["image_sha256"]
            metrics.update(decision["verdict"].get("metrics", {}))
            counts = local_cells[candidate["cell"]]
            counts["input_chains"] += 1
            counts["accepted_chains" if decision["accepted"] else "rejected_chains"] += 1
            if not decision["verdict"]["verification_complete"]:
                counts["verifier_errors"] += 1
            elif decision["verdict"]["semantic_accept"] is False:
                counts["semantic_rejected_chains"] += 1
            provenance = {"manifest_path": str(manifest_path), "manifest_id": manifest["manifest_id"],
                          "manifest_sha256": file_sha256(manifest_path), "ledger_entry_index": index,
                          "input_train_path": str(input_paths["reasoning_chains_train.json"]),
                          "output_train_sha256": outputs["reasoning_chains_train.json"]}
            combined = deepcopy(decision)
            combined["entry_index"] = original["index"]
            combined["shard_provenance"] = provenance
            merged_by_parent[original["index"]] = combined
            if decision["accepted"]:
                filtered = deepcopy(entry)
                filtered["metadata"]["quality_status"] = "accepted_by_automated_visual_verifier"
                filtered["metadata"]["quality_filter"] = {
                    "version": manifest["filter_version"], "verifier_fingerprint": manifest["verifier_fingerprint"],
                    "decision_sha256": decision["verdict"]["decision_sha256"], "ledger_entry_index": index,
                    "candidate_sha256": identity[1]}
                expected_accepted.append(deepcopy(filtered))
                filtered["metadata"]["quality_filter"]["ledger_entry_index"] = original["index"]
                filtered["metadata"]["quality_filter"]["shard_provenance"] = provenance
                kept_by_parent[original["index"]] = filtered
            seen.add(identity)
        assert_equal(accepted_entries, expected_accepted, "Accepted shard chains/quality references differ from authoritative source decisions")
        assert_equal(manifest["source_rollout_sha256"], expected_rollouts, "Shard rollout binding inventory mismatch")
        assert_equal(manifest["original_image_sha256"], expected_images, "Shard image binding inventory mismatch")
        assert_equal(manifest["per_cell"], {cell: dict(counts) for cell, counts in local_cells.items()}, "Shard per-cell counts mismatch")
        assert_equal(manifest["accepted_chains"], len(expected_accepted), "Shard accepted count mismatch")
        assert_equal(manifest["rejected_chains"], len(ledger) - len(expected_accepted), "Shard rejected count mismatch")
        assert_equal(manifest["verifier_errors"], sum(not d["verdict"]["verification_complete"] for d in ledger), "Shard verifier-error count mismatch")
        assert_equal(manifest["semantic_rejected_chains"], sum(d["verdict"]["semantic_accept"] is False for d in ledger), "Shard semantic-rejection count mismatch")
        reasons = Counter(reason for decision in ledger for reason in decision["verdict"]["reasons"])
        assert_equal(manifest["reasons"], dict(reasons), "Shard rejection-reason counts mismatch")
        for key in set(metrics) | set(manifest["model_metrics"]):
            if not math.isclose(metrics[key], manifest["model_metrics"].get(key, 0), rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError("Shard total critic metrics do not match candidate metrics")
        model_metrics.update(manifest["model_metrics"])
        shard_input_bindings.update(manifest["source_sft_sha256"])
        source_bindings.update(manifest["source_sft_sha256"])
        shard_reports.append({"path": str(manifest_path), "sha256": file_sha256(manifest_path),
                              "manifest_id": manifest["manifest_id"], "verifier_fingerprint": manifest["verifier_fingerprint"],
                              "verifier_config": manifest["verifier_config"], "input_chains": len(ledger),
                              "accepted_chains": len(expected_accepted)})
    if seen != parent.keys():
        raise ValueError(f"Quality shard coverage does not equal parent candidates: missing {len(parent.keys() - seen)}")
    for report in shard_reports:
        audit.verify_fingerprint(checked_json(report["path"]))
    source_bindings.update(audit.files)
    checked_bindings(source_bindings)
    merged = [merged_by_parent[index] for index in sorted(merged_by_parent)]
    kept = [kept_by_parent[index] for index in sorted(kept_by_parent)]
    per_cell = defaultdict(Counter)
    question_sets = defaultdict(lambda: defaultdict(set))
    reasons, protocols = Counter(), []
    for decision in merged:
        cell, sample_id = decision["cell"], decision["sample_id"]
        counts = per_cell[cell]
        counts["input_chains"] += 1
        question_sets[cell]["input"].add(sample_id)
        status = "accepted" if decision["accepted"] else "rejected"
        counts[status + "_chains"] += 1
        question_sets[cell][status].add(sample_id)
        verdict = decision["verdict"]
        reasons.update(verdict["reasons"])
        if not verdict["verification_complete"]:
            counts["verifier_errors"] += 1
            question_sets[cell]["verifier_error"].add(sample_id)
            protocols.append({"entry_index": decision["entry_index"], "sample_id": sample_id, "cell": cell,
                              "failed_stage": verdict.get("failed_stage"), "reasons": verdict["reasons"]})
        elif verdict["semantic_accept"] is False:
            counts["semantic_rejected_chains"] += 1
            question_sets[cell]["semantic_rejected"].add(sample_id)
        elif not decision["accepted"]:
            counts["policy_rejected_chains"] += 1
    for cell, counts in per_cell.items():
        for status, values in question_sets[cell].items():
            counts[status + "_unique_questions"] = len(values)
        counts["selected_questions"] = admission.selection["counts"]["cells"].get(cell, 0)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, value in (("train", kept), ("val", []), ("test", [])):
        atomic_json(output_dir / f"reasoning_chains_{split}.json", value)
    ledger_path = output_dir / "quality_decisions.jsonl"
    ledger_path.write_text("".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in merged))
    manifest = {"schema_version": 1, "merge_version": MERGE_VERSION,
                "filter_version": policy_base["filter_version"], "quality_shards": shard_reports,
                "verifier_policy_id": content_hash({**policy_base, "stage_prompts": audit.prompts,
                                                     "retry_instruction": audit.retry_instruction}),
                "model_identity": policy_base["model_identity"], "model_metrics": dict(model_metrics),
                "source_sft_sha256": parent_inputs, "source_shard_sft_sha256": shard_input_bindings,
                "source_rollout_sha256": admission.source_files, "original_image_sha256": admission.image_files,
                "split_manifest_sha256": admission.manifest.sha256, "review_overlay_id": admission.review.id,
                "review_overlay_sha256": admission.review.sha256, "selection_id": admission.selection["selection_id"],
                "audited_source_artifact_sha256": source_bindings,
                "input_chains": len(merged), "accepted_chains": len(kept), "rejected_chains": len(merged) - len(kept),
                "input_unique_questions": len({d["sample_id"] for d in merged}),
                "accepted_unique_questions": len({d["sample_id"] for d in merged if d["accepted"]}),
                "semantic_rejected_chains": sum(d["verdict"]["semantic_accept"] is False for d in merged),
                "policy_rejected_chains": sum(d["verdict"]["semantic_accept"] is True and not d["accepted"] for d in merged),
                "verifier_errors": len(protocols), "protocol_errors": protocols,
                "per_cell": {cell: dict(counts) for cell, counts in sorted(per_cell.items())}, "reasons": dict(reasons),
                "all_parent_candidates_covered_once": True, "all_candidates_semantically_evaluated": not protocols,
                "visual_grounding_certified": False,
                "limitation": "Automated same-model critique can share generator errors; acceptance is not guaranteed visual correctness.",
                "outputs": {path.name: file_sha256(path) for path in output_dir.iterdir() if path.is_file()}}
    manifest["manifest_id"] = content_hash(manifest)
    atomic_json(output_dir / "quality_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sft", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--rollout-dir", type=Path)
    args = parser.parse_args()
    report = merge_quality_shards(args.parent_sft, args.selection, args.shard, args.output_dir, args.data_root, args.rollout_dir)
    print(json.dumps({key: report[key] for key in ("manifest_id", "input_chains", "accepted_chains", "accepted_unique_questions", "semantic_rejected_chains", "verifier_errors", "model_metrics")}, indent=2))


if __name__ == "__main__":
    main()
