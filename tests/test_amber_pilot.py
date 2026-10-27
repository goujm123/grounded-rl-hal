"""Pilot regression checks: balanced/disjoint sampling and audited output admission."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vlmsearch.data_manifest import file_sha256
from vlmsearch.data_review import content_id
from vlmsearch.image_geometry import ImagePayloadCache


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_pilot", "scripts/data/prepare_amber_pilot.py")
analyze = module("analyze_pilot", "scripts/mcts/analyze_amber_pilot.py")


class PilotSelectionTests(unittest.TestCase):
    def test_quota_allocation_balances_labels_and_preserves_150_design(self):
        self.assertEqual(prepare.allocate_quotas(150), prepare.BASE_QUOTAS)
        for count in [100, 101, 150, 200]:
            quotas = prepare.allocate_quotas(count)
            self.assertEqual(sum(quotas.values()), count)
            self.assertEqual(sum(n for c, n in quotas.items() if c.endswith(":yes")), count // 2)
            self.assertEqual(len(quotas), 9)

    def test_flow_is_deterministic_distinct_and_covers_large_images(self):
        samples, images = [], {}
        reverse = {value: key for key, value in prepare.CATEGORIES.items() if key != "relation"}
        for index in range(24):
            name = f"image_{index}"
            images[name] = {"width": 8000 if index < 8 else 640, "height": 6000 if index < 8 else 480}
            for cell in ["state:yes", "state:no", "existence:no"]:
                category, answer = cell.split(":")
                samples.append({"id": f"{name}:{cell}", "category": reverse[category], "answer": answer,
                                "image": name, "cluster_id": name})
        quotas = {"state:yes": 6, "state:no": 6, "existence:no": 6}
        first = prepare.choose_samples(samples, images, quotas, seed=42)
        second = prepare.choose_samples(samples, images, quotas, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len({sample["cluster_id"] for sample in first}), 18)
        self.assertEqual(sum(prepare.size_bucket(sample, images, 1048576) == "large" for sample in first), 3)
        self.assertTrue(any(sample["image"] == "image_0" for sample in first))

    def test_insufficient_disjoint_images_fail(self):
        samples = [{"id": "one", "image": "one", "cluster_id": "one", "answer": "yes", "category": "discriminative-attribute-state"}]
        with self.assertRaisesRegex(ValueError, "Cannot fill"):
            prepare.choose_samples(samples, {"one": {"width": 640, "height": 480}}, {"state:yes": 2})


class PilotIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest_path = self.root / "manifest.json"
        self.review_path = self.root / "review.json"
        self.train_file = self.root / "train.jsonl"
        self.output = self.root / "pilot"
        self.run = self.root / "run"
        self.run.mkdir()
        samples, images, rows = {}, {}, []
        categories = {value: key for key, value in prepare.CATEGORIES.items() if key != "relation"}
        for cell in prepare.BASE_QUOTAS:
            category, answer = cell.split(":")
            for number in range(3):
                sample_id = f"amber:{cell}:{number}"
                image = f"{category}_{answer}_{number}.png"
                Image.new("RGB", (16, 16), (number * 80, 10, 20)).save(self.root / image)
                images[image] = {"width": 16, "height": 16, "split": "train", "cluster_id": sample_id,
                                 "file_sha256": file_sha256(self.root / image)}
                samples[sample_id] = {"image": image, "split": "train", "cluster_id": sample_id,
                                      "dataset": "amber", "task": "amber_discriminative", "answer": answer,
                                      "category": categories[category], "question": f"Question {sample_id}?"}
                rows.append({"id": sample_id, "image": image,
                             "conversations": [{"from": "human", "value": "<image>\n" + samples[sample_id]["question"]},
                                               {"from": "gpt", "value": answer}]})
        manifest = {"schema_version": 1, "samples": samples, "images": images}
        manifest["manifest_id"] = prepare.canonical_hash(manifest)
        self.manifest_path.write_text(json.dumps(manifest))
        excluded_id = next(iter(samples))
        sample = samples[excluded_id]
        review = {"schema_version": 1, "frozen_manifest": {"sha256": file_sha256(self.manifest_path),
                                                              "manifest_id": manifest["manifest_id"]},
                  "source_files": {}, "entries": [{"id": excluded_id, "split": "train", "image": sample["image"],
                  "question": sample["question"], "category": sample["category"], "original_label": sample["answer"],
                  "image_sha256": images[sample["image"]]["file_sha256"], "decision": "exclude",
                  "rationale": "Fixture exclusion for review admission", "confidence": "high"}],
                  "excluded_ids": [excluded_id], "retained_reviewed_ids": []}
        review["overlay_id"] = content_id(review)
        self.review_path.write_text(json.dumps(review))
        self.train_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.excluded_id = excluded_id

    def prepare(self):
        return prepare.prepare_pilot(self.manifest_path, self.review_path, self.train_file,
                                     self.root, self.output, count=18)

    def write_run(self, selection, bad_coordinate=False):
        rows = []
        for index, sample in enumerate(selection["samples"]):
            image = Image.open(self.root / sample["image"])
            _, geometry = ImagePayloadCache().encode(image)
            image.close()
            coordinate = "(1001, 500)" if bad_coordinate and index == 0 else "(500, 500)"
            final = "<answer>" + sample["answer"] + "</answer>"
            rollout = {"final_answer": sample["answer"], "raw_final_answer": final, "reward": 1,
                       "ephemeral_texts": ["<evidence>Visible object at " + coordinate + ".</evidence>", final]}
            rows.append({"id": sample["id"], "image": sample["image"], "question": sample["question"],
                         "true_answer": sample["answer"], "data_split": "train", "image_group": sample["image_group"],
                         "split_manifest_sha256": selection["frozen_manifest"]["sha256"],
                         "review_overlay_id": selection["review_overlay"]["overlay_id"],
                         "review_overlay_sha256": selection["review_overlay"]["sha256"],
                         "image_geometry": geometry, "global_search_time": index + 1,
                         "tree": {"rollouts": [rollout], "children": []}})
        (self.run / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_prepare_excludes_reviewed_rows_and_freezes_reproducibly(self):
        selection = self.prepare()
        self.assertNotIn(self.excluded_id, {sample["id"] for sample in selection["samples"]})
        self.assertEqual(selection["counts"]["labels"], {"yes": 9, "no": 9})
        self.assertEqual(selection["counts"]["image_groups"], 18)
        self.assertEqual(selection, self.prepare())
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            prepare.prepare_pilot(self.manifest_path, self.review_path, self.train_file,
                                  self.root, self.output, count=18, seed=43)

    def test_changed_gold_fails_before_selection(self):
        rows = [json.loads(line) for line in self.train_file.read_text().splitlines()]
        rows[0]["conversations"][1]["value"] = "yes" if rows[0]["conversations"][1]["value"] == "no" else "no"
        self.train_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "answer changed"):
            self.prepare()

    def test_analyzer_reports_coverage_answers_and_coordinate_failures(self):
        selection = self.prepare()
        self.write_run(selection, bad_coordinate=True)
        report = analyze.analyze_pilot(self.output / "selection.json", self.run, self.root)
        self.assertTrue(report["structural_checks_passed"])
        self.assertEqual(report["processed_unique_questions"], 18)
        self.assertEqual(report["totals"]["correct_final_answers"], 18)
        self.assertEqual(report["totals"]["invalid_coordinate_trajectories"], 1)
        self.assertEqual(report["search_seconds"]["sum"], sum(range(1, 19)))
        self.assertFalse(report["visual_grounding_certified"])

    def test_analyzer_rejects_duplicate_outputs_and_missing_review_provenance(self):
        selection = self.prepare()
        self.write_run(selection)
        path = self.run / "rollouts.jsonl"
        original = path.read_text()
        path.write_text(original + original.splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate output"):
            analyze.analyze_pilot(self.output / "selection.json", self.run, self.root)
        rows = [json.loads(line) for line in original.splitlines()]
        del rows[0]["review_overlay_id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "review provenance"):
            analyze.analyze_pilot(self.output / "selection.json", self.run, self.root)

    def test_analyzer_distinguishes_nonfinite_points_from_parenthetical_prose(self):
        self.assertEqual(analyze.coordinate_status("(sky, cloud)", "original_pixels", [16, 16]), {"points": [], "errors": []})
        for text in ["(nan, 2)", "(-1, 2)", "(16, 2)"]:
            self.assertTrue(analyze.coordinate_status(text, "original_pixels", [16, 16])["errors"])

    def test_converter_and_analyzer_preserve_review_and_coordinate_conversion(self):
        selection = self.prepare()
        self.write_run(selection)
        shutil.copy2(self.manifest_path, self.run / "split_manifest.json")
        shutil.copy2(self.review_path, self.run / "training_review.json")
        converter = module("pilot_integration_converter", "scripts/mcts/build_reasoning_chains_from_mcts.py")
        output = self.root / "sft"
        argv = ["converter", "--data-str", str(self.run), "--data-root", str(self.root),
                "--output-dir", str(output), "--allow-no-coordinate"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            converter.main()
        report = analyze.analyze_pilot(self.output / "selection.json", self.run, self.root,
                                      sft_dir=output, allow_no_coordinate=True)
        self.assertTrue(report["structural_checks_passed"])
        self.assertEqual(report["sft"]["retained_unique_questions"], 18)
        self.assertEqual(report["totals"]["sft_entries"], 18)
        self.assertEqual(report["sft"]["validation_errors"], [])
        for rates in report["per_cell_rates"].values():
            self.assertEqual(rates["questions_retained_in_sft_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
