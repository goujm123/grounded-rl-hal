"""Completed critic shards must preserve exact candidate coverage and provenance."""

from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import QualityVerifier, VerifierConfig, atomic_json, content_hash


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module("merge_quality_fixtures", "tests/test_quality_verifier.py")
merge = module("merge_quality_shards", "scripts/mcts/merge_quality_shards.py")


class QualityShardMergeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceFilterTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.run = self.fixture.run
        self.parent = self.root / "parent"
        self.parent.mkdir()
        row = deepcopy(self.fixture.row)
        second = deepcopy(row["tree"]["rollouts"][0])
        second["ephemeral_texts"] = ["<evidence>A red square is visible at (750, 750).</evidence>"]
        row["tree"]["rollouts"].append(second)
        self.fixture.source.write_text(json.dumps(row) + "\n")
        admission = fixtures.quality_filter.SourceAdmission(self.root, self.run)
        rebuilt = admission.linearizer.build_all_reasoning_for_sample(row, original_size=(64, 64))[0]
        self.entries = []
        for index, chain in enumerate(rebuilt):
            entry = deepcopy(self.fixture.entry)
            entry["id"] = "entry_" + str(index)
            entry["messages"][1]["content"] = chain["chain"]
            entry["metadata"]["rollout_idx"] = chain["rollout_idx"]
            entry["metadata"]["evidence_normalization"] = chain["evidence_normalization"]
            self.entries.append(entry)
        self.assertEqual(len(self.entries), 2)
        self.write_sft(self.parent, self.entries)
        self.selection = self.root / "selection.json"
        value = {"schema_version": 1, "frozen_manifest": {"sha256": admission.manifest.sha256},
                 "review_overlay": {"sha256": admission.review.sha256},
                 "samples": [{"id": "amber:1", "image": "image.png", "question": "Is there a red square?", "answer": "yes"}],
                 "counts": {"cells": {"state:yes": 1}}}
        value["selection_id"] = content_hash(value)
        self.selection.write_text(json.dumps(value))
        self.shards = []

    @staticmethod
    def write_sft(directory, entries):
        directory.mkdir(exist_ok=True)
        for split, value in [("train", entries), ("val", []), ("test", [])]:
            (directory / f"reasoning_chains_{split}.json").write_text(json.dumps(value))

    def filter_shards(self, second_changes=None):
        for index, entry in enumerate(self.entries):
            inputs = self.root / f"inputs_{index}"
            self.write_sft(inputs, [entry])
            output = self.root / f"filtered_{index}"
            admission = fixtures.quality_filter.SourceAdmission(self.root, self.run, selection_path=self.selection)
            verifier = QualityVerifier(VerifierConfig(base_url=f"http://127.0.0.1:{9011 + index}/v1"),
                                       self.root / f"cache_{index}", client=fixtures.FakeCritic(second_changes if index else None),
                                       model_identity={"revision": "fixture"}, verify_endpoint=False)
            with contextlib.redirect_stdout(io.StringIO()):
                fixtures.quality_filter.filter_sft(inputs, output, admission, verifier)
            self.shards.append(output)

    def merge(self, shards=None):
        return merge.merge_quality_shards(self.parent, self.selection, shards or self.shards,
                                          self.root / "merged", self.root, self.run)

    @staticmethod
    def refresh_manifest(shard):
        path = shard / "quality_manifest.json"
        value = json.loads(path.read_text())
        value["outputs"] = {name: file_sha256(shard / name) for name in value["outputs"]}
        value["manifest_id"] = content_hash({key: item for key, item in value.items() if key != "manifest_id"})
        atomic_json(path, value)

    def test_merge_preserves_exact_coverage_and_reindexes_accepted_decisions(self):
        self.filter_shards()
        report = self.merge()
        self.assertEqual(report["input_chains"], 2)
        self.assertEqual(report["accepted_chains"], 2)
        self.assertEqual(report["accepted_unique_questions"], 1)
        self.assertEqual(report["verifier_errors"], 0)
        self.assertEqual(report["model_metrics"]["api_calls"], 10)
        self.assertTrue(report["all_parent_candidates_covered_once"])
        ledger = [json.loads(line) for line in (self.root / "merged/quality_decisions.jsonl").read_text().splitlines()]
        kept = json.loads((self.root / "merged/reasoning_chains_train.json").read_text())
        self.assertEqual([entry["entry_index"] for entry in ledger], [0, 1])
        self.assertEqual([entry["metadata"]["quality_filter"]["ledger_entry_index"] for entry in kept], [0, 1])
        self.assertEqual([entry["metadata"]["quality_filter"]["shard_provenance"]["ledger_entry_index"] for entry in kept], [0, 0])
        for name, expected in report["outputs"].items():
            self.assertEqual(file_sha256(self.root / "merged" / name), expected)
        self.assertEqual(json.loads((self.root / "merged/reasoning_chains_val.json").read_text()), [])
        self.assertEqual(json.loads((self.root / "merged/reasoning_chains_test.json").read_text()), [])

    def test_protocol_errors_are_separate_from_semantic_rejections(self):
        self.filter_shards({"scene_claims": lambda value: "{}"})
        report = self.merge()
        self.assertEqual(report["accepted_chains"], 1)
        self.assertEqual(report["verifier_errors"], 1)
        self.assertEqual(report["semantic_rejected_chains"], 0)
        self.assertFalse(report["all_candidates_semantically_evaluated"])
        self.assertEqual(report["protocol_errors"][0]["failed_stage"], "scene_claims")

    def test_semantic_rejection_has_no_protocol_error(self):
        def unsupported(value):
            value["claims"][0]["status"] = "unsupported"
            return value
        self.filter_shards({"scene_claims": unsupported})
        report = self.merge()
        self.assertEqual(report["accepted_chains"], 1)
        self.assertEqual(report["semantic_rejected_chains"], 1)
        self.assertEqual(report["verifier_errors"], 0)

    def test_missing_or_duplicate_shard_coverage_fails_before_publishing(self):
        self.filter_shards()
        duplicate = self.root / "duplicate"
        import shutil
        shutil.copytree(self.shards[0], duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate sample/chain"):
            self.merge([self.shards[0], duplicate])
        self.assertFalse((self.root / "merged").exists())
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.merge([self.shards[0]])

    def test_incomplete_manifest_and_output_tampering_fail(self):
        self.filter_shards()
        manifest = self.shards[0] / "quality_manifest.json"
        original = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, "Missing completed artifact"):
            self.merge()
        manifest.write_bytes(original)
        path = self.shards[0] / "reasoning_chains_train.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "output checksum changed"):
            self.merge()

    def test_forged_accepted_chain_cannot_be_admitted_by_rehashing_outputs(self):
        self.filter_shards()
        path = self.shards[0] / "reasoning_chains_train.json"
        value = json.loads(path.read_text())
        value[0]["messages"][1]["content"] = value[0]["messages"][1]["content"].replace("red", "blue")
        path.write_text(json.dumps(value))
        self.refresh_manifest(self.shards[0])
        with self.assertRaisesRegex(ValueError, "Accepted shard chains"):
            self.merge()

    def test_forged_candidate_binding_cannot_be_admitted_by_rehashing_decision(self):
        self.filter_shards()
        path = self.shards[0] / "quality_decisions.jsonl"
        decision = json.loads(path.read_text())
        verdict = decision["verdict"]
        verdict["binding"]["candidate_sha256"] = "0" * 64
        verdict["decision_sha256"] = content_hash({key: item for key, item in verdict.items() if key != "decision_sha256"})
        path.write_text(json.dumps(decision) + "\n")
        self.refresh_manifest(self.shards[0])
        with self.assertRaisesRegex(ValueError, "Candidate/source/critic binding mismatch"):
            self.merge()

    def test_source_or_cache_record_mutation_fails(self):
        self.filter_shards()
        cache = next((self.root / "cache_0").glob("*.json"))
        original = cache.read_bytes()
        cache.write_text(cache.read_text() + " ")
        # Whitespace preserves canonical cache content, so the bound attempt and
        # cache content remain valid; alter actual content to exercise the guard.
        value = json.loads(cache.read_text())
        value["raw_response"] += " altered"
        cache.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "record_sha256"):
            self.merge()
        cache.write_bytes(original)
        self.fixture.source.write_text(self.fixture.source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.merge()

    def test_fine_claim_inventory_cannot_be_removed_by_rehashing(self):
        self.filter_shards()
        path = self.shards[0] / "quality_decisions.jsonl"
        decision = json.loads(path.read_text())
        verdict = decision["verdict"]
        del verdict["fine_attribute_targets"]
        verdict["decision_sha256"] = content_hash({key: item for key, item in verdict.items() if key != "decision_sha256"})
        path.write_text(json.dumps(decision) + "\n")
        self.refresh_manifest(self.shards[0])
        with self.assertRaisesRegex(ValueError, "Fine-attribute source inventory changed"):
            self.merge()

    def test_completed_decision_cannot_omit_its_fine_attribute_pass(self):
        self.filter_shards()
        path = self.shards[0] / "quality_decisions.jsonl"
        decision = json.loads(path.read_text())
        verdict = decision["verdict"]
        del verdict["passes"]["fine_attributes"]
        verdict["decision_sha256"] = content_hash({key: item for key, item in verdict.items() if key != "decision_sha256"})
        path.write_text(json.dumps(decision) + "\n")
        self.refresh_manifest(self.shards[0])
        with self.assertRaisesRegex(ValueError, "lacks fine-attribute pass"):
            self.merge()


if __name__ == "__main__":
    unittest.main()
