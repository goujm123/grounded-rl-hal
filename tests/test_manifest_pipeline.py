import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from vlmsearch.data_manifest import FrozenDataManifest, attach_manifest, file_sha256, tag_rollout
from vlmsearch.reward_funcs.judge import Judge
from vlmsearch.tree_search.mcts_search import MonteCarloTreeSearch, TreeNode
from vlmsearch.tree_search.single_path_rollouts import SinglePathRollouts, TreeNode as SinglePathNode
from vlmsearch.__main__ import rollout_diagnostics, tag_model_output


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("linearizer", ROOT / "scripts/mcts/build_reasoning_chains_from_mcts.py")
linearizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linearizer)
spec = importlib.util.spec_from_file_location("rl_exporter", ROOT / "scripts/rl/build_amber_discriminative_rl.py")
rl_exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_exporter)


class ManifestPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "split_manifest.json"
        images, samples = {}, {}
        for index, split in enumerate(("train", "dev", "test")):
            name = f"{split}.png"
            Image.new("RGB", (16, 16), (index * 80, 20, 40)).save(self.root / name)
            images[name] = {"split": split, "cluster_id": split, "file_sha256": file_sha256(self.root / name)}
            samples[f"amber:{split}"] = {
                "image": name, "split": split, "cluster_id": split, "dataset": "amber",
                "task": "amber_discriminative", "question": "Is there a car?", "answer": "yes",
            }
        self.manifest_path.write_text(json.dumps({"samples": samples, "images": images}))
        self.manifest = FrozenDataManifest(self.manifest_path, self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def sample(self, split):
        return {"id": f"amber:{split}", "image": str(self.root / f"{split}.png"),
                "input_query": "<image>\nIs there a car?", "true_answer": "yes"}

    def test_holdout_and_duplicate_inputs_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "expected train"):
            attach_manifest([self.sample("test")], self.manifest, "train")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            attach_manifest([self.sample("train"), self.sample("train")], self.manifest, "train")
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            attach_manifest([self.sample("train"), self.sample("dev")], self.manifest)

    def test_changed_gold_and_image_are_rejected(self):
        sample = self.sample("train")
        sample["true_answer"] = "no"
        with self.assertRaisesRegex(ValueError, "answer changed"):
            attach_manifest([sample], self.manifest, "train")
        Image.new("RGB", (16, 16), "white").save(self.root / "train.png")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            attach_manifest([self.sample("train")], self.manifest, "train")

    def test_provenance_survives_output_and_cannot_be_reassigned(self):
        sample = attach_manifest([self.sample("train")], self.manifest, "train")[0]
        output = tag_rollout({"question": "Is there a car?", "true_answer": "yes", "image": sample["image"]}, sample)
        self.assertEqual(self.manifest.validate_rollout(output)["data_split"], "train")
        output["data_split"] = "test"
        with self.assertRaisesRegex(ValueError, "incompatible data_split"):
            self.manifest.validate_rollout(output)

    def test_runtime_geometry_and_request_metrics_are_copied_per_question(self):
        sample = attach_manifest([self.sample("train")], self.manifest, "train")[0]
        geometry = {"original_size": [16, 16], "coordinate_frame": "normalized_1000"}
        model = SimpleNamespace(last_image_geometry=geometry, metrics={"api_calls": 5, "completion_tokens": 120})
        result = tag_model_output({}, sample, model, time.perf_counter(), {"api_calls": 2, "completion_tokens": 50})
        self.assertEqual(result["model_diagnostics"], {"api_calls": 3, "completion_tokens": 70})
        self.assertGreaterEqual(result["search_elapsed_seconds"], 0)
        geometry["original_size"][0] = 999
        self.assertEqual(result["image_geometry"]["original_size"], [16, 16])

    def test_runtime_rejects_geometry_for_a_different_source_image(self):
        model = SimpleNamespace(last_image_geometry={"original_size": [32, 16]}, metrics={})
        with self.assertRaisesRegex(ValueError, "does not match"):
            tag_model_output({}, self.sample("train"), model, time.perf_counter(), {})

    def test_sft_preserves_all_three_frozen_partitions(self):
        run = self.root / "run"
        run.mkdir()
        (run / "split_manifest.json").write_bytes(self.manifest_path.read_bytes())
        rows = []
        for split in ("train", "dev", "test"):
            sample = attach_manifest([self.sample(split)], self.manifest, split)[0]
            raw = "<think>A car is visible at (8, 8).</think><answer>yes</answer>"
            rows.append(tag_rollout({
                "question": "Is there a car?", "true_answer": "yes", "image": sample["image"],
                "tree": {"value": 1, "children": [], "rollouts": [{"reward": 1, "final_answer": "yes",
                         "raw_final_answer": raw, "ephemeral_texts": ["Is there a car?", raw]}]},
            }, sample))
        (run / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = self.root / "sft"
        with patch.object(sys, "argv", ["linearizer", "--data-str", str(run), "--data-root", str(self.root), "--output-dir", str(output),
                                         "--legacy-coordinate-frame", "original_pixels"]):
            with contextlib.redirect_stdout(io.StringIO()):
                linearizer.main()
        for split, filename in (("train", "train"), ("dev", "val"), ("test", "test")):
            entries = json.loads((output / f"reasoning_chains_{filename}.json").read_text())
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["metadata"]["data_split"], split)
            self.assertEqual(entries[0]["metadata"]["sample_key"], f"amber:{split}")

    def test_sft_does_not_rescue_a_malformed_raw_answer(self):
        rollout = {"final_answer": "yes", "raw_final_answer": "<answer>yes</answer><answer>no</answer>",
                   "ephemeral_texts": ["<think>A car is at (8, 8).</think>"]}
        self.assertEqual(linearizer.normalize_rollout_chain(rollout, "Question")[2], "invalid_raw_final_answer")

    def test_visible_evidence_is_converted_to_student_thought_format(self):
        rollout = {"final_answer": "yes", "raw_final_answer": "<answer>yes</answer>",
                   "ephemeral_texts": ["Question", "<evidence>A car is visible at (8, 8).</evidence>", "<answer>yes</answer>"]}
        chain, answer, error = linearizer.normalize_rollout_chain(
            rollout, "Question", original_size=(16, 16), legacy_coordinate_frame="original_pixels")
        self.assertIsNone(error)
        self.assertEqual(answer, "yes")
        self.assertIn("<think>", chain)
        self.assertNotIn("<evidence>", chain)
        self.assertIsNone(linearizer.extract_reasoning_step("<think>car at (8, 8)</evidence>"))

    def test_rl_manifest_export_retains_all_partitions_and_writes_report(self):
        output = self.root / "rl"
        with patch.object(sys, "argv", ["rl_exporter", "--data-root", str(self.root), "--split-manifest", str(self.manifest_path), "--output-dir", str(output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                rl_exporter.main()
        for split, name in (("train", "train"), ("dev", "val"), ("test", "test")):
            entries = [json.loads(line) for line in (output / f"amber_discriminative_{name}.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["data_split"], split)
            self.assertEqual(entries[0]["id"], f"amber:{split}")
        self.assertEqual(json.loads((output / "amber_discriminative_split_report.json").read_text())["split_policy"], "frozen_manifest")


class RawSearchScoringTests(unittest.TestCase):
    def test_diagnostics_do_not_confuse_http_success_with_valid_answers(self):
        result = rollout_diagnostics({"true_answer": "no", "tree": {"rollouts": [
            {"final_answer": "no", "raw_final_answer": "<evidence>No ship is visible.</evidence>", "reward": 0},
            {"final_answer": "no", "raw_final_answer": "<answer>no</answer>", "reward": 1},
        ]}})
        self.assertEqual(result, {"trajectories": 2, "valid_final_answers": 1, "positive_trajectories": 1})

    def search(self, response):
        class FakeModel:
            def generate_single_thought(self, **kwargs):
                return response
        return MonteCarloTreeSearch(llm_wrapper=FakeModel(), judge=Judge(None, "string_match"), system_prompt="Check image")

    def test_expansion_preserves_ambiguous_raw_response_for_scoring(self):
        raw = "<answer>yes</answer><answer>no</answer>"
        search = self.search(raw)
        root = TreeNode("Question", image=Image.new("RGB", (16, 16)))
        child = search._expand(root, "Check", 1)[0]
        reward, _ = search._rollout(child, "Check", "Question", root.image, "yes")
        self.assertEqual(reward, 0)
        self.assertEqual(child.rollouts[0]["raw_final_answer"], raw)

    def test_ephemeral_final_response_uses_the_same_strict_scorer(self):
        search = self.search("<answer>yes</answer> trailing no")
        root = TreeNode("Question", image=Image.new("RGB", (16, 16)))
        child = TreeNode("<think>A car is at (8, 8).</think>", parent=root)
        reward, _ = search._rollout(child, "Check", "Question", root.image, "yes")
        self.assertEqual(reward, 0)

    def test_single_path_upfront_keeps_the_entire_response_for_scoring(self):
        raw = "<think>A car is at (8, 8).</think><answer>yes</answer><answer>no</answer>"
        class FakeModel:
            def generate_single_thought(self, **kwargs):
                return raw
        judge = Judge(None, "string_match")
        with contextlib.redirect_stdout(io.StringIO()):
            search = SinglePathRollouts(FakeModel(), judge, "Check", generate_upfront=True)
        root = SinglePathNode("Question", image=Image.new("RGB", (16, 16)))
        path, _ = search._do_single_rollout(root, "Check")
        self.assertEqual(path[-1].raw_final_answer, raw)
        self.assertEqual(judge.judge("yes", path[-1].raw_final_answer, "Question"), 0)


if __name__ == "__main__":
    unittest.main()
