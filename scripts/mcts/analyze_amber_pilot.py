#!/usr/bin/env python3
"""Audit a reviewed AMBER pilot's coverage, answer yield, grounding and SFT retention."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

from vlmsearch.data_manifest import FrozenDataManifest, file_sha256
from vlmsearch.image_geometry import CoordinateError, parse_coordinate_points, validate_image_geometry
from vlmsearch.reward_funcs.answers import parse_final_answer


BLOCK_RE = re.compile(r"<(think|evidence)>\s*(.*?)\s*</\1>", re.IGNORECASE | re.DOTALL)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "median": values[len(values) // 2],
            "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
            "max": values[-1], "mean": sum(values) / len(values), "sum": sum(values)}


def iter_rollouts(node):
    yield from node.get("rollouts", [])
    for child in node.get("children", []):
        yield from iter_rollouts(child)


def coordinate_status(text, frame, size):
    """Independent numeric/bounds check of every coordinate-like parenthesis."""
    points = []
    errors = []
    if frame not in {"normalized_1000", "original_pixels"}:
        return {"points": [], "errors": ["unknown_coordinate_frame"]}
    bounds = [1000, 1000] if frame == "normalized_1000" else [size[0] - 1, size[1] - 1]
    try:
        parsed = parse_coordinate_points(text)
    except CoordinateError as exc:
        return {"points": [], "errors": [{"reason": str(exc)}]}
    for x, y in parsed:
        if not (0 <= x <= bounds[0] and 0 <= y <= bounds[1]):
            errors.append({"coordinate": [x, y], "reason": "out_of_bounds", "upper_bounds": bounds})
        else:
            points.append([x, y])
    return {"points": points, "errors": errors}


def evidence_text(rollout):
    # The input question contains no evidence; inspect generated blocks only.
    texts = list(rollout.get("ephemeral_texts", []))
    if rollout.get("raw_final_answer"):
        texts.append(rollout["raw_final_answer"])
    blocks = []
    for text in texts:
        blocks.extend(match.group(2) for match in BLOCK_RE.finditer(text))
    return "\n".join(blocks)


def load_selection(path, data_root):
    selection = json.loads(Path(path).read_text())
    expected_id = selection.get("selection_id")
    if expected_id != canonical_hash({key: value for key, value in selection.items() if key != "selection_id"}):
        raise ValueError("Pilot selection content hash changed")
    frozen = selection["frozen_manifest"]
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("Frozen split manifest changed since pilot selection")
    manifest = FrozenDataManifest(frozen["path"], data_root)
    if manifest.data.get("manifest_id") != frozen["manifest_id"]:
        raise ValueError("Pilot split manifest ID mismatch")
    if file_sha256(selection["pilot_file"]["path"]) != selection["pilot_file"]["sha256"]:
        raise ValueError("Selected pilot inputs changed")
    from vlmsearch.data_review import ReviewedTrainingData
    review = ReviewedTrainingData(selection["review_overlay"]["path"], manifest)
    if review.id != selection["review_overlay"]["overlay_id"] or review.sha256 != selection["review_overlay"]["sha256"]:
        raise ValueError("Training review changed since pilot selection")
    selected = {}
    for sample in selection["samples"]:
        sample_id = sample["id"]
        if sample_id in selected:
            raise ValueError(f"Duplicate pilot ID: {sample_id}")
        manifest.validate_sample(sample_id, sample["image"], sample["question"], sample["answer"], "train")
        review.validate_sample(sample_id)
        if sample["image_group"] != manifest.samples[sample_id]["cluster_id"]:
            raise ValueError(f"Pilot image group changed for {sample_id}")
        selected[sample_id] = sample
    pilot_rows = [json.loads(line) for line in Path(selection["pilot_file"]["path"]).read_text().splitlines() if line.strip()]
    if [str(row["id"]) for row in pilot_rows] != list(selected):
        raise ValueError("Pilot file and selection IDs/order disagree")
    for row in pilot_rows:
        conversation = row.get("conversations", [])
        if len(conversation) != 2:
            raise ValueError("Malformed pilot conversation")
        manifest.validate_sample(row["id"], row["image"], conversation[0]["value"], conversation[1]["value"], "train")
    return selection, manifest, review, selected


def analyze_pilot(selection_path, run_dir, data_root, sft_dir=None, log_file=None, wall_seconds=None, allow_no_coordinate=False):
    selection, manifest, review, selected = load_selection(selection_path, data_root)
    run_dir = Path(run_dir)
    per_cell = {cell: Counter(selected_questions=count) for cell, count in selection["counts"]["cells"].items()}
    sample_reports = {}
    row_errors = []
    run_errors = []
    geometry_errors = []
    all_trajectory_texts = Counter()
    latencies = []
    model_totals = Counter()
    for path in sorted(run_dir.glob("errors*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                run_errors.append({"file": str(path), "line": line_number, **json.loads(line)})
    for path in sorted(run_dir.glob("rollouts*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id"))
            if sample_id not in selected:
                raise ValueError(f"Run contains a non-pilot sample: {sample_id}")
            if sample_id in sample_reports:
                raise ValueError(f"Run contains duplicate output records for {sample_id}")
            sample = selected[sample_id]
            metadata = manifest.validate_rollout(row)
            if metadata["data_split"] != "train":
                raise ValueError("Pilot run contains held-out data")
            if row.get("review_overlay_id") != review.id or row.get("review_overlay_sha256") != review.sha256:
                raise ValueError(f"Missing or incompatible review provenance for {sample_id}")
            cell = sample["cell"]
            geometry = row.get("image_geometry", {})
            try:
                validate_image_geometry(geometry, sample["original_size"])
                geometry_ok = True
            except CoordinateError:
                geometry_ok = False
            transport = geometry.get("transport_size", [])
            if len(transport) != 2 or any(not isinstance(value, int) or value <= 0 for value in transport):
                geometry_ok = False
            elif transport[0] * transport[1] > 1048576:
                geometry_ok = False
            if not geometry_ok:
                geometry_errors.append(sample_id)
            counts = Counter(output_records=1)
            traces = list(iter_rollouts(row.get("tree", {})))
            details = []
            for rollout_index, rollout in enumerate(traces):
                counts["trajectories"] += 1
                answer = parse_final_answer(rollout.get("raw_final_answer") or rollout.get("final_answer"), {"yes", "no"})
                if answer is not None:
                    counts["valid_final_answers"] += 1
                if answer == sample["answer"]:
                    counts["correct_final_answers"] += 1
                if rollout.get("reward", 0) > 0:
                    counts["positive_reward_trajectories"] += 1
                    if answer != sample["answer"]:
                        row_errors.append({"id": sample_id, "rollout_index": rollout_index, "error": "positive_reward_for_wrong_or_invalid_answer"})
                text = evidence_text(rollout)
                coordinates = coordinate_status(text, "normalized_1000", sample["original_size"])
                if coordinates["points"]:
                    counts["coordinate_bearing_trajectories"] += 1
                    counts["coordinate_points"] += len(coordinates["points"])
                else:
                    counts["trajectories_without_coordinates"] += 1
                if coordinates["errors"]:
                    counts["invalid_coordinate_trajectories"] += 1
                if not text.strip():
                    counts["trajectories_without_evidence"] += 1
                normalized_text = re.sub(r"\s+", " ", text).strip()
                if normalized_text:
                    all_trajectory_texts[normalized_text] += 1
                details.append({"rollout_index": rollout_index, "answer": answer,
                                "correct": answer == sample["answer"], "reward": rollout.get("reward"),
                                "coordinate_points": len(coordinates["points"]), "coordinate_errors": coordinates["errors"]})
            counts["questions_with_any_correct_final"] = int(counts["correct_final_answers"] > 0)
            counts["questions_with_any_positive_reward"] = int(counts["positive_reward_trajectories"] > 0)
            per_cell[cell].update(counts)
            latency = row.get("search_elapsed_seconds", row.get("global_search_time"))
            if isinstance(latency, (int, float)) and math.isfinite(latency):
                latencies.append(latency)
            for key, value in row.get("model_diagnostics", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    model_totals[key] += value
            sample_reports[sample_id] = {"cell": cell, "counts": dict(counts), "geometry_valid": geometry_ok,
                                          "image_geometry": geometry, "global_search_time": latency,
                                          "source_file": str(path), "source_line": line_number, "trajectories": details}

    sft_errors = []
    sft_chains = Counter()
    linearization = None
    if sft_dir:
        sft_dir = Path(sft_dir)
        for split in ["train", "val", "test"]:
            path = sft_dir / f"reasoning_chains_{split}.json"
            if not path.is_file():
                sft_errors.append({"error": "missing_sft_export", "split": split})
                continue
            entries = json.loads(path.read_text())
            if split != "train" and entries:
                sft_errors.append({"error": "heldout_sft_entries", "split": split, "count": len(entries)})
            for entry in entries:
                metadata = entry.get("metadata", {})
                sample_id = metadata.get("sample_key")
                if sample_id not in selected:
                    raise ValueError(f"SFT contains non-pilot sample: {sample_id}")
                sample = selected[sample_id]
                per_cell[sample["cell"]]["sft_entries"] += 1
                sft_chains[sample_id] += 1
                content = next((message["content"] for message in entry["messages"] if message["role"] == "assistant"), "")
                coordinates = coordinate_status(content, "original_pixels", sample["original_size"])
                source_geometry = metadata.get("source_image_geometry", {})
                if metadata.get("coordinate_frame") != "original_pixels" or source_geometry.get("original_size") != sample["original_size"]:
                    sft_errors.append({"id": sample_id, "error": "missing_coordinate_conversion_provenance"})
                if coordinates["errors"]:
                    sft_errors.append({"id": sample_id, "error": "invalid_student_coordinates", "details": coordinates["errors"]})
                if parse_final_answer(content, {"yes", "no"}) != sample["answer"]:
                    sft_errors.append({"id": sample_id, "error": "wrong_sft_final_answer"})
                if metadata.get("review_overlay_id") != review.id:
                    sft_errors.append({"id": sample_id, "error": "missing_review_provenance"})
        for sample_id in sft_chains:
            per_cell[selected[sample_id]["cell"]]["questions_retained_in_sft"] += 1
        report_path = sft_dir / "linearization_report.json"
        if report_path.is_file():
            linearization = json.loads(report_path.read_text())
        if not sft_chains:
            sft_errors.append({"error": "no_retained_sft_questions"})

    # Recompute converter filters by cell so a global retention number cannot
    # hide systematic loss of a question/label category.
    if sft_dir:
        converter_path = Path(__file__).resolve().parent / "build_reasoning_chains_from_mcts.py"
        spec = importlib.util.spec_from_file_location("pilot_linearizer", converter_path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        for sample_id, info in sample_reports.items():
            row = json.loads(Path(info["source_file"]).read_text().splitlines()[info["source_line"] - 1])
            _, _, filters = converter.build_all_reasoning_for_sample(
                row, require_coordinate=not allow_no_coordinate,
                original_size=selected[sample_id]["original_size"],
            )
            per_cell[selected[sample_id]["cell"]].update({"converter_" + key: value for key, value in filters.items()})

    totals = Counter()
    for values in per_cell.values():
        totals.update(values)
    cell_rates = {}
    for cell, values in per_cell.items():
        trajectories = values["trajectories"]
        questions = values["selected_questions"]
        cell_rates[cell] = {
            "valid_final_fraction": values["valid_final_answers"] / trajectories if trajectories else None,
            "correct_final_fraction": values["correct_final_answers"] / trajectories if trajectories else None,
            "questions_with_any_correct_fraction": values["questions_with_any_correct_final"] / questions if questions else None,
            "questions_retained_in_sft_fraction": values["questions_retained_in_sft"] / questions if sft_dir and questions else None,
        }
    missing_ids = sorted(set(selected) - set(sample_reports))
    selected_without_sft = sorted(set(selected) - set(sft_chains)) if sft_dir else None
    run_report_path = run_dir / "run_report.json"
    logged_requests = None
    if log_file:
        log = Path(log_file).read_text(errors="replace")
        logged_requests = len(re.findall(r"HTTP Request: POST .*?/chat/completions", log))
    return {
        "schema_version": 1, "scope": "Training-only diagnostic; search uses gold labels, so these are search-yield measurements, not benchmark accuracy.",
        "selection_id": selection["selection_id"], "review_overlay_id": review.id,
        "split_manifest_sha256": manifest.sha256, "run_dir": str(run_dir.resolve()),
        "selected": len(selected), "processed_unique_questions": len(sample_reports), "missing_ids": missing_ids,
        "run_errors": run_errors, "scoring_errors": row_errors, "invalid_geometry_ids": geometry_errors,
        "totals": dict(totals), "per_cell": {key: dict(value) for key, value in per_cell.items()},
        "per_cell_rates": cell_rates,
        "search_seconds": distribution(latencies), "wall_seconds": wall_seconds, "logged_api_requests": logged_requests,
        "model_diagnostics_totals": dict(model_totals),
        "repeated_evidence_texts": {"duplicate_excess": sum(count - 1 for count in all_trajectory_texts.values()),
                                    "groups": sum(count > 1 for count in all_trajectory_texts.values())},
        "sft": {"directory": str(sft_dir.resolve()) if sft_dir else None, "allow_no_coordinate": allow_no_coordinate,
                "retained_unique_questions": len(sft_chains), "without_sft_ids": selected_without_sft,
                "validation_errors": sft_errors, "linearization_report": linearization},
        "run_report": json.loads(run_report_path.read_text()) if run_report_path.is_file() else None,
        "sample_reports": sample_reports,
        "structural_checks_passed": not (missing_ids or run_errors or row_errors or geometry_errors or sft_errors),
        "visual_grounding_certified": False,
        "limitations": ["Correct gold-label answers and in-bounds coordinates do not certify visual support.",
                        "Pilot labels/categories are deliberately balanced and do not reflect full AMBER prevalence.",
                        "No held-out evaluation was used to tune pilot examples or generation settings."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rollouts-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--sft-dir", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--allow-no-coordinate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_pilot(args.selection, args.rollouts_dir, args.data_root, args.sft_dir,
                           args.log_file, args.wall_seconds, args.allow_no_coordinate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ["selected", "processed_unique_questions", "totals", "search_seconds", "structural_checks_passed"]}, indent=2))
    return 0 if report["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
