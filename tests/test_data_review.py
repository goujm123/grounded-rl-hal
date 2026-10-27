import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from vlmsearch.data_manifest import FrozenDataManifest, file_sha256, tag_rollout
from vlmsearch.data_review import ReviewedTrainingData, attach_review, content_id


class TrainingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        samples, images, entries = {}, {}, []
        for index, (split, decision) in enumerate([("train", "retain"), ("train", "exclude"), ("test", None)]):
            key = f"image{index}.png"
            Image.new("RGB", (16, 8), (index * 60, 20, 30)).save(self.root / key)
            images[key] = {"split": split, "cluster_id": key, "file_sha256": file_sha256(self.root / key)}
            sid = f"amber:{index}"
            samples[sid] = {"image": key, "split": split, "cluster_id": key, "dataset": "amber",
                            "task": "amber_discriminative", "question": "Is there a car?", "answer": "no",
                            "category": "discriminative-hallucination"}
            if decision:
                entries.append({"id": sid, "split": split, "decision": decision, "image": key,
                                "image_sha256": images[key]["file_sha256"], "question": "Is there a car?",
                                "original_label": "no", "category": "discriminative-hallucination",
                                "rationale": "Visible scene reviewed.", "confidence": "high"})
        manifest = {"samples": samples, "images": images}
        manifest["manifest_id"] = content_id(manifest, "manifest_id")
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.manifest = FrozenDataManifest(path, self.root)
        self.review = {"schema_version": 1, "frozen_manifest": {"sha256": self.manifest.sha256,
                       "manifest_id": manifest["manifest_id"]}, "entries": entries,
                       "excluded_ids": ["amber:1"], "retained_reviewed_ids": ["amber:0"], "source_files": {}}
        self.path = self.root / "review.json"
        self.write()

    def write(self):
        self.review["overlay_id"] = content_id(self.review)
        self.path.write_text(json.dumps(self.review))

    def load(self, **kwargs):
        return ReviewedTrainingData(self.path, self.manifest, **kwargs)

    def test_exclusion_applies_before_sampling_and_tags_survive(self):
        review = self.load(required_candidate_ids={"amber:0", "amber:1"})
        samples = attach_review([{"id": "amber:0"}, {"id": "amber:1"}], review)
        self.assertEqual([r["id"] for r in samples], ["amber:0"])
        output = tag_rollout({}, samples[0])
        self.assertEqual(output["review_overlay_id"], review.id)
        self.assertEqual(output["review_overlay_sha256"], file_sha256(self.path))
        with self.assertRaisesRegex(ValueError, "excluded"):
            review.validate_sample("amber:1")

    def test_held_out_samples_are_never_admitted(self):
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.load().validate_sample("amber:2")
        entry = copy.deepcopy(self.review["entries"][0])
        entry.update(id="amber:2", image="image2.png", image_sha256=self.manifest.images["image2.png"]["file_sha256"])
        self.review["entries"].append(entry)
        self.write()
        with self.assertRaisesRegex(ValueError, "belongs to test"):
            self.load()

    def test_tampered_content_and_wrong_frozen_manifest_fail(self):
        data = json.loads(self.path.read_text())
        data["entries"][0]["decision"] = "exclude"
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.load()
        self.review["frozen_manifest"]["sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "different frozen manifest"):
            self.load()

    def test_changed_gold_or_duplicate_review_ids_fail(self):
        self.review["entries"][0]["original_label"] = "yes"
        self.write()
        with self.assertRaisesRegex(ValueError, "answer changed"):
            self.load()
        self.review["entries"][0]["original_label"] = "no"
        self.review["entries"].append(copy.deepcopy(self.review["entries"][0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_incomplete_review_and_forged_exclusion_list_fail(self):
        self.review["entries"] = self.review["entries"][:1]
        self.review["excluded_ids"] = []
        self.write()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load(required_candidate_ids={"amber:0", "amber:1"})
        self.review["excluded_ids"] = ["amber:1"]
        self.write()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_review_source_hashes_are_rechecked(self):
        path = self.root / "source.json"
        path.write_text("official source")
        self.review["source_files"] = {"annotations": {"path": "source.json", "sha256": file_sha256(path)}}
        self.write()
        self.load()
        path.write_text("modified source")
        with self.assertRaisesRegex(ValueError, "source checksum changed"):
            self.load()


if __name__ == "__main__":
    unittest.main()
