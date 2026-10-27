"""Check frozen input admission, endpoint identity, and process ownership."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from vlmsearch.data_review import content_id


SPEC = importlib.util.spec_from_file_location("qwen38_rollout", Path(__file__).resolve().parents[1] / "scripts/mcts/qwen38_rollout.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        root_patch = patch.object(launcher, "REPO_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.data = self.root / "train.jsonl"
        self.manifest = self.root / "manifest.json"
        self.image = self.root / "amber/images/a.jpg"
        self.image.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16), "red").save(self.image)
        self.row = {"id": "amber:1005", "image": "amber/images/a.jpg", "conversations": [
            {"from": "human", "value": "<image>\nIs it red?"}, {"from": "gpt", "value": "yes"}]}
        self.assignment = {"split": "train", "dataset": "amber", "task": "amber_discriminative",
                           "image": "amber/images/a.jpg", "cluster_id": "image-cluster",
                           "question": "Is it red?", "answer": "yes"}
        self.data.write_text(json.dumps(self.row) + "\n")
        self.write_manifest()
        self.review = self.root / "review.json"
        review = {"schema_version": 1, "entries": [], "excluded_ids": [], "retained_reviewed_ids": [],
                  "source_files": {}, "frozen_manifest": {"sha256": launcher.FrozenDataManifest(self.manifest, self.root).sha256,
                  "manifest_id": json.loads(self.manifest.read_text())["manifest_id"]}}
        review["overlay_id"] = content_id(review)
        self.review.write_text(json.dumps(review))
        self.environment = {"DATA_ROOT": str(self.root), "DATA_FILE": str(self.data),
                            "SPLIT_MANIFEST": str(self.manifest), "QWEN38_MODEL_PATH": str(self.root / "model"),
                            "AMBER_REVIEW_OVERLAY": str(self.review)}

    def write_manifest(self):
        content = {"samples": {self.row["id"]: self.assignment}, "images": {
            "amber/images/a.jpg": {"split": "train", "cluster_id": "image-cluster",
                                   "file_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()}}}
        content["manifest_id"] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.manifest.write_text(json.dumps(content))

    def test_input_cannot_cross_splits_or_change_labels(self):
        self.assertEqual(launcher.validate_input(self.data, self.manifest, "train"), 1)
        with self.assertRaisesRegex(ValueError, "belongs to train"):
            launcher.validate_input(self.data, self.manifest, "dev")
        self.row["conversations"][1]["value"] = "no"
        self.data.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "Ground-truth answer changed"):
            launcher.validate_input(self.data, self.manifest, "train")

    def test_corrupt_images_and_manifest_are_rejected_before_gpu_start(self):
        self.image.write_bytes(self.image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Image checksum changed"):
            launcher.validate_input(self.data, self.manifest, "train")
        content = json.loads(self.manifest.read_text())
        content["samples"][self.row["id"]]["answer"] = "no"
        self.manifest.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "manifest_id"):
            launcher.validate_input(self.data, self.manifest, "train")

    def test_endpoint_checks_authorization_alias_and_checkpoint(self):
        response = io.BytesIO(json.dumps({"data": [{"id": "qwen_vllm", "root": str(self.root / "model")}]}).encode())
        with patch.object(launcher.urllib.request, "urlopen", return_value=response) as request:
            self.assertTrue(launcher.verify_endpoint("http://localhost:9011/v1", "secret", "qwen_vllm", self.root / "model"))
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer secret")
        wrong = io.BytesIO(json.dumps({"data": [{"id": "qwen_vllm", "root": "/old/Qwen2.5"}]}).encode())
        with patch.object(launcher.urllib.request, "urlopen", return_value=wrong):
            with self.assertRaises(launcher.EndpointMismatch):
                launcher.verify_endpoint("http://localhost:9011/v1", "secret", "qwen_vllm", self.root / "model")

    def test_reused_server_survives_failed_rollout_and_failure_propagates(self):
        rollout = MagicMock(pid=900001)
        rollout.wait.return_value = 7
        rollout.poll.return_value = 7
        with patch.dict(os.environ, self.environment, clear=True), patch.object(launcher.sys, "argv", ["launcher"]), \
             patch.object(launcher, "verify_endpoint", return_value=True), \
             patch.object(launcher.subprocess, "Popen", return_value=rollout) as spawn, \
             patch.object(launcher.os, "killpg") as kill, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(), 7)
        self.assertEqual(spawn.call_count, 1)
        self.assertIn("vlmsearch", spawn.call_args.args[0])
        self.assertNotIn("--add_thought_number_system_prompt", spawn.call_args.args[0])
        self.assertIn("--review_overlay", spawn.call_args.args[0])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        kill.assert_not_called()

    def test_only_owned_server_group_is_stopped(self):
        server = MagicMock(pid=900002)
        server.poll.return_value = None
        server.wait.return_value = 0
        rollout = MagicMock(pid=900003)
        rollout.wait.return_value = 0
        rollout.poll.return_value = 0
        with patch.dict(os.environ, self.environment, clear=True), patch.object(launcher.sys, "argv", ["launcher"]), \
             patch.object(launcher, "verify_endpoint", side_effect=[False, True]), \
             patch.object(launcher.subprocess, "Popen", side_effect=[server, rollout]) as spawn, \
             patch.object(launcher.os, "killpg") as kill, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(), 0)
        self.assertEqual(spawn.call_count, 2)
        kill.assert_called_once_with(server.pid, signal.SIGTERM)

    def test_dry_run_does_not_start_processes(self):
        with patch.dict(os.environ, self.environment, clear=True), patch.object(launcher.sys, "argv", ["launcher", "--dry-run"]), \
             patch.object(launcher.subprocess, "Popen") as spawn, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(), 0)
        spawn.assert_not_called()
        command = json.loads(output.getvalue())["command"]
        self.assertIn("--split_manifest", command)
        self.assertIn("--data_split", command)

    def test_missing_review_blocks_teacher_start(self):
        self.review.unlink()
        with patch.dict(os.environ, self.environment, clear=True), patch.object(launcher.sys, "argv", ["launcher"]), \
             patch.object(launcher.subprocess, "Popen") as spawn:
            with self.assertRaises(FileNotFoundError):
                launcher.main()
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
