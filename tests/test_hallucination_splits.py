"""Regression checks for image identity, frozen partitions, and export integrity."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/prepare_hallucination_splits.py"
SPEC = importlib.util.spec_from_file_location("prepare_hallucination_splits", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparedSplitsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.normalized = self.root / "normalized"
        self.output = self.root / "prepared"
        self.policy = {
            "algorithm": "source-file-rgb-phash-connected-components-v1", "seed": 42,
            "amber_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
            "heldout_datasets": ["pope", "longhalqa"], "phash_distance": 4,
            "aspect_ratio_log_tolerance": 0.02, "min_luminance_std": 2.0, "flat_image_rgb_rmse": 0.02,
        }

    def image(self, name, seed):
        path = self.root / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(seed).integers(0, 256, size=(72, 96, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        return path.relative_to(self.root).as_posix()

    def fixture(self, external_duplicate="reencoded"):
        rows = {dataset: [] for dataset in prepare.DATASETS}

        def record(dataset, index, image, source=None, answer="yes", answer_type="binary"):
            row = {
                "id": f"{dataset}:{index}", "dataset": dataset, "task": f"{dataset}_task", "image": image,
                "question": f"Question {index}?", "answer": answer, "answer_type": answer_type,
                "category": "existence", "source_revision": "test-pinned-revision", "source_id": str(index),
                "source_image_id": source or f"{dataset}:{index}",
            }
            rows[dataset].append(row)
            return row

        for index in range(20):
            record("amber", index, self.image(f"amber_{index}.png", index))
        external_image = self.image("pope.png", 100)
        record("pope", 0, external_image, "coco2014:100")
        record("pope", 1, external_image, "coco2014:100", answer="no")
        duplicate_path = self.root / "images/amber_external.jpg"
        with Image.open(self.root / external_image) as image:
            if external_duplicate == "reencoded":
                image.resize((192, 144)).save(duplicate_path, quality=98)
            else:
                duplicate_path = self.root / "images/amber_external.png"
                duplicate_path.write_bytes((self.root / external_image).read_bytes())
        record("amber", "external", duplicate_path.relative_to(self.root).as_posix())
        record("longhalqa", 0, self.image("longhal.png", 101), "visualgenome:101", "C", "choice")
        for dataset, values in rows.items():
            prepare.write_jsonl(self.normalized / f"{dataset}.jsonl", values)
        return rows

    def build(self):
        records, inputs = prepare.load_inputs(self.normalized, self.root)
        images, clusters, exact, candidates = prepare.group_images(records, self.root, self.policy)
        manifest = prepare.make_manifest(records, inputs, images, clusters, self.policy)
        prepare.freeze_manifest(self.output / "manifest.json", manifest)
        exports = prepare.export_rows(records, manifest, self.output)
        report = prepare.make_report(records, manifest, exact, candidates, exports)
        prepare.write_json(self.output / "report.json", report)
        return records, inputs, manifest, report

    def test_reencoded_cross_dataset_match_is_reserved_and_variants_retained(self):
        self.fixture()
        records, inputs, manifest, report = self.build()
        self.assertEqual(manifest["samples"]["amber:external"]["split"], "test")
        self.assertEqual(manifest["samples"]["amber:external"]["cluster_id"], manifest["samples"]["pope:0"]["cluster_id"])
        self.assertEqual(manifest["samples"]["pope:0"]["split"], "test")
        self.assertEqual(manifest["samples"]["pope:1"]["split"], "test")
        self.assertGreaterEqual(report["perceptual_matches_merged"], 1)
        self.assertEqual(sum(report["counts"]["amber"][split]["rows"] for split in prepare.SPLITS), 21)
        self.assertTrue(prepare.audit_dataset(records, inputs, self.root, self.output)["passed"])
        train = prepare.load_jsonl(self.output / "amber/train.mcts.jsonl")
        source = {row["id"]: row for row in records}
        for row in train:
            self.assertEqual(row["conversations"][1]["value"], source[row["id"]]["answer"])
            self.assertEqual(row["conversations"][0]["value"], "<image>\n" + source[row["id"]]["question"])
        evaluation = prepare.load_jsonl(self.output / "evaluation/longhalqa.jsonl")
        self.assertEqual(evaluation[0]["answer"], "C")

    def test_manifest_is_deterministic_and_refuses_resplitting(self):
        self.fixture()
        _, _, first, _ = self.build()
        before = (self.output / "manifest.json").read_bytes()
        _, _, second, _ = self.build()
        self.assertEqual(first, second)
        self.assertEqual(before, (self.output / "manifest.json").read_bytes())
        self.policy["seed"] = 43
        with self.assertRaisesRegex(ValueError, "Frozen manifest differs"):
            self.build()
        self.assertEqual(before, (self.output / "manifest.json").read_bytes())

    def test_source_identity_is_namespaced_and_same_source_groups_variants(self):
        rows = self.fixture()
        # These IDs are local to their own corpus and must not collide.
        self.assertNotEqual(prepare.source_key(dict(rows["amber"][0], source_image_id="1")), prepare.source_key(dict(rows["pope"][0], source_image_id="1")))
        # An explicit canonical image-corpus identity is shared across datasets.
        rows["amber"][0]["source_image_id"] = "visualgenome:101"
        prepare.write_jsonl(self.normalized / "amber.jsonl", rows["amber"])
        _, _, manifest, _ = self.build()
        self.assertEqual(manifest["samples"]["amber:0"]["split"], "test")
        self.assertEqual(manifest["samples"]["amber:0"]["cluster_id"], manifest["samples"]["longhalqa:0"]["cluster_id"])

    def test_audit_rejects_image_or_export_changes(self):
        self.fixture()
        records, inputs, _, _ = self.build()
        with (self.output / "amber/train.mcts.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Export changed"):
            prepare.audit_dataset(records, inputs, self.root, self.output)
        self.build()
        path = self.root / records[0]["image"]
        with Image.open(path) as image:
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(path)
        with self.assertRaisesRegex(ValueError, "Image changed"):
            prepare.audit_dataset(records, inputs, self.root, self.output)

    def test_flat_color_hash_collision_does_not_merge_different_images(self):
        red_path, blue_path = self.root / "red.png", self.root / "blue.png"
        Image.new("RGB", (64, 64), "red").save(red_path)
        Image.new("RGB", (64, 64), "blue").save(blue_path)
        red, red_thumb = prepare.fingerprint_image(red_path)
        blue, blue_thumb = prepare.fingerprint_image(blue_path)
        # If platform DCT rounding gives these the same candidate hash, the
        # low-information guard must reject it; otherwise they are not candidates.
        match = prepare.near_duplicate(red, blue, (red_thumb, blue_thumb), self.policy)
        self.assertTrue(match is None or not match["merged"])

    def test_audit_checks_semantics_even_if_export_checksum_is_updated(self):
        self.fixture()
        records, inputs, _, _ = self.build()
        relative = "amber/train.mcts.jsonl"
        exported = prepare.load_jsonl(self.output / relative)
        exported[0]["conversations"][1]["value"] = "no"
        prepare.write_jsonl(self.output / relative, exported)
        report = json.loads((self.output / "report.json").read_text())
        report["exports"][relative]["sha256"] = prepare.sha256_file(self.output / relative)
        prepare.write_json(self.output / "report.json", report)
        with self.assertRaisesRegex(ValueError, "MCTS export prompt/image/answer differs"):
            prepare.audit_dataset(records, inputs, self.root, self.output)

    def test_conflicting_labels_are_retained_and_only_sensitivity_export_excludes_them(self):
        rows = self.fixture()
        rows["longhalqa"].append(dict(rows["longhalqa"][0], id="longhalqa:conflict", answer="D"))
        prepare.write_jsonl(self.normalized / "longhalqa.jsonl", rows["longhalqa"])
        records, inputs, _, _ = self.build()
        primary = prepare.load_jsonl(self.output / "evaluation/longhalqa.jsonl")
        sensitivity = prepare.load_jsonl(self.output / "evaluation/longhalqa.unambiguous.jsonl")
        conflicts = json.loads((self.output / "label-conflicts.json").read_text())
        self.assertEqual({row["answer"] for row in primary}, {"C", "D"})
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(sensitivity), 0)
        self.assertEqual(conflicts["ambiguous_samples"], 2)
        self.assertEqual(conflicts["conflicting_groups"], 1)
        self.assertTrue(conflicts["official_labels_preserved"])
        self.assertTrue(prepare.audit_dataset(records, inputs, self.root, self.output)["passed"])


if __name__ == "__main__":
    unittest.main()
