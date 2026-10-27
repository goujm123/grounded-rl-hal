"""Coordinate-frame and bounded image-upload regressions for grounded traces."""

import base64
import contextlib
from copy import deepcopy
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from vlmsearch.image_geometry import (
    CoordinateError, ImagePayloadCache, NORMALIZED_COORDINATE_INSTRUCTION,
    convert_coordinate_text, parse_coordinate_points, smart_resize_size, validate_image_geometry,
)
from vlmsearch.data_manifest import FrozenDataManifest, file_sha256
from vlmsearch.data_review import content_id

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("geometry_linearizer", ROOT / "scripts/mcts/build_reasoning_chains_from_mcts.py")
linearizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linearizer)


def geometry(size=(500, 334)):
    return {"version": 1, "coordinate_frame": "normalized_1000", "coordinate_extent": [1000, 1000],
            "original_size": list(size), "transport_size": list(smart_resize_size(*size)),
            "preprocessing": {"policy": "rgb_smart_resize_v1", "min_pixels": 65536,
                              "max_pixels": 1048576, "resize_factor": 32, "format": "PNG"}}


class CoordinateGeometryTests(unittest.TestCase):
    def test_normalized_points_map_to_original_pixels_and_endpoints(self):
        text, audit = convert_coordinate_text("board (520, 513); corners (0, 0), (1000, 1000)", geometry())
        self.assertEqual(text, "board (259, 171); corners (0, 0), (499, 333)")
        self.assertEqual(audit[0], {"source": [520.0, 513.0], "original_pixels": [259, 171]})

    def test_all_coordinates_are_checked(self):
        for text in ("good (50, 50); bad (1001, 10)", "(-1, 0)", "(0, -0.1)",
                     "(NaN, 12)", "(inf, 12)", "(1e309, 12)", "(1, invalid)", "(1..2, 3..4)",
                     "(1, 2, 3)", "(1,000, 2,000)", "(−1, −2)"):
            with self.subTest(text=text), self.assertRaises(CoordinateError):
                convert_coordinate_text(text, geometry())

    def test_coordinate_frame_is_never_guessed_for_legacy_data(self):
        with self.assertRaisesRegex(CoordinateError, "missing_coordinate_frame"):
            convert_coordinate_text("point (8, 8)", original_size=(16, 16))
        text, _ = convert_coordinate_text("point (8, 8)", original_size=(16, 16), legacy_coordinate_frame="original_pixels")
        self.assertEqual(text, "point (8, 8)")
        with self.assertRaisesRegex(CoordinateError, "coordinate_out_of_bounds"):
            convert_coordinate_text("point (520, 513)", original_size=(500, 334), legacy_coordinate_frame="original_pixels")

    def test_geometry_is_verified_against_actual_original_image(self):
        with self.assertRaisesRegex(CoordinateError, "original_size_mismatch"):
            validate_image_geometry(geometry(), (600, 400))
        broken = geometry()
        broken["transport_size"][0] += 32
        with self.assertRaisesRegex(CoordinateError, "transport_size_mismatch"):
            validate_image_geometry(broken)
        broken = geometry()
        broken["coordinate_frame"] = "guessed"
        with self.assertRaisesRegex(CoordinateError, "unsupported_coordinate_frame"):
            validate_image_geometry(broken)

    def test_ordinary_parenthetical_prose_is_not_a_point(self):
        self.assertEqual(parse_coordinate_points("I checked both entities (sky, cloud)."), [])

    def test_absence_can_omit_points_but_cannot_hide_invalid_ones(self):
        rollout = {"final_answer": "no", "raw_final_answer": "<answer>no</answer>",
                   "ephemeral_texts": ["<evidence>No bicycle is visible anywhere in the image.</evidence>"]}
        chain, answer, error = linearizer.normalize_rollout_chain(
            rollout, "Question", require_coordinate=False, image_geometry=geometry(), original_size=(500, 334))
        self.assertIsNone(error)
        self.assertEqual(answer, "no")
        rollout["ephemeral_texts"] = ["<evidence>No bicycle is visible near (-1, 99).</evidence>"]
        self.assertEqual(linearizer.normalize_rollout_chain(
            rollout, "Question", require_coordinate=False, image_geometry=geometry())[2], "coordinate_out_of_bounds")

    def test_linearizer_converts_before_coordinate_retention(self):
        rollout = {"final_answer": "yes", "raw_final_answer": "<answer>yes</answer>",
                   "ephemeral_texts": ["<evidence>A board is at (520, 513).</evidence>"]}
        audit = {}
        chain, answer, error = linearizer.normalize_rollout_chain(
            rollout, "Question", image_geometry=geometry(), original_size=(500, 334), coordinate_audit=audit)
        self.assertIsNone(error)
        self.assertIn("(259, 171)", chain)
        self.assertEqual(audit["mapping"][0]["source"], [520.0, 513.0])
        prompt = linearizer.normalize_system_prompt(NORMALIZED_COORDINATE_INSTRUCTION, (500, 334))
        self.assertNotIn("normalized 0-1000", prompt)
        self.assertIn("zero-based pixels", prompt)

    def test_student_prompt_matches_complete_target_not_teacher_step_loop(self):
        source = importlib.util.spec_from_file_location("geometry_teacher_launcher", ROOT / "scripts/mcts/qwen38_rollout.py")
        teacher = importlib.util.module_from_spec(source)
        source.loader.exec_module(teacher)
        self.assertIn("Return exactly ONE of these blocks", teacher.SYSTEM_PROMPT)
        prompt = linearizer.normalize_system_prompt(teacher.SYSTEM_PROMPT, (500, 334))
        self.assertIn("one complete response", prompt)
        self.assertIn("exactly one <think> block followed by exactly one <answer> block", prompt)
        self.assertIn("coordinates may be omitted", prompt)
        self.assertIn("Never invent a point for an absent object", prompt)
        self.assertIn("0 <= x <= 499 and 0 <= y <= 333", prompt)
        self.assertNotIn("Return exactly ONE of these blocks", prompt)
        self.assertNotIn("Previous evidence", prompt)
        self.assertNotIn("normalized 0-1000", prompt)
        self.assertNotIn("<evidence>", prompt)


class ImageTransportTests(unittest.TestCase):
    def test_resize_matches_qwen_factor_and_pixel_limits(self):
        for size in [(6000, 9000), (500, 334), (142, 175), (4000, 4168), (1248, 702)]:
            with self.subTest(size=size):
                width, height = smart_resize_size(*size)
                self.assertEqual(width % 32, 0)
                self.assertEqual(height % 32, 0)
                self.assertTrue(65536 <= width * height <= 1048576)
                self.assertLess(abs((width / height) / (size[0] / size[1]) - 1), .13)
                self.assertEqual(smart_resize_size(width, height), (width, height))

    def test_rgb_alpha_and_grayscale_uploads_decode_and_record_geometry(self):
        cache = ImagePayloadCache()
        for mode, color in [("RGB", "red"), ("RGBA", (255, 0, 0, 0)), ("L", 50)]:
            with self.subTest(mode=mode):
                image = Image.new(mode, (142, 175), color)
                uri, metadata = cache.encode(image)
                payload = base64.b64decode(uri.split(",", 1)[1])
                with Image.open(BytesIO(payload)) as decoded:
                    self.assertEqual(decoded.mode, "RGB")
                    self.assertEqual(list(decoded.size), metadata["transport_size"])
                    if mode == "RGBA":
                        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(metadata["original_size"], [142, 175])
                self.assertEqual(metadata["transport_bytes"], len(payload))

    def test_repeated_requests_reuse_encoding_and_cache_is_bounded(self):
        image = Image.new("RGB", (2000, 1000), "red")
        cache = ImagePayloadCache(max_entries=1)
        first, geometry_first = cache.encode(image)
        second, geometry_second = cache.encode(image)
        self.assertEqual(first, second)
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(cache.stats()["misses"], 1)
        geometry_second["original_size"][0] = 9
        self.assertEqual(cache.encode(image)[1]["original_size"], [2000, 1000])
        other = Image.new("RGB", (32, 32), "blue")
        cache.encode(other)
        self.assertEqual(cache.stats()["entries"], 1)
        self.assertLessEqual(cache.stats()["bytes"], cache.max_bytes)

    def test_same_immutable_source_file_reuses_cache_across_pil_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 32), "red").save(path)
            cache = ImagePayloadCache()
            with Image.open(path) as first:
                uri, _ = cache.encode(first)
            with Image.open(path) as second:
                self.assertEqual(cache.encode(second)[0], uri)
            self.assertEqual(cache.stats()["hits"], 1)

    def test_client_declares_geometry_only_with_explicit_protocol(self):
        from vlmsearch.models import qwen_vllm
        fake = MagicMock()
        fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<answer>yes</answer>"))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=8))
        with patch.object(qwen_vllm, "client", fake):
            client = qwen_vllm.Qwen_VLLM(thought_token_begin="<evidence>", thought_token_end="</evidence>")
            image = Image.new("RGB", (500, 334), "red")
            client.generate_single_thought("Check. " + NORMALIZED_COORDINATE_INSTRUCTION, [("Question", image)])
            self.assertEqual(client.last_image_geometry["coordinate_frame"], "normalized_1000")
            self.assertEqual(client.metrics["prompt_tokens"], 30)
            client.generate_single_thought("Legacy pixel prompt.", [("Question", image)])
            self.assertIsNone(client.last_image_geometry)
            self.assertEqual(client.metrics["image_cache_hits"], 1)
            self.assertEqual(client.metrics["api_calls"], 2)


class ConverterReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        Image.new("RGB", (500, 334), "red").save(self.root / "image.png")
        self.image_sha = file_sha256(self.root / "image.png")
        data = {"images": {"image.png": {"split": "train", "cluster_id": "group", "file_sha256": self.image_sha}},
                "samples": {"amber:1": {"image": "image.png", "split": "train", "cluster_id": "group",
                                         "dataset": "amber", "task": "amber_discriminative",
                                         "question": "Is there a bicycle?", "answer": "no", "category": "existence"}}}
        (self.run / "split_manifest.json").write_text(json.dumps(data))
        self.manifest = FrozenDataManifest(self.run / "split_manifest.json", self.root)
        self.row = {"id": "amber:1", "question": "Is there a bicycle?", "true_answer": "no", "image": "image.png",
                    "image_geometry": geometry(), "system_prompt": NORMALIZED_COORDINATE_INSTRUCTION,
                    "tree": {"value": 1, "children": [], "rollouts": [{"reward": 1,
                             "final_answer": "no", "raw_final_answer": "<answer>no</answer>",
                             "ephemeral_texts": ["<evidence>A red area is at (500, 500); no bicycle is visible.</evidence>"]}]}}

    def tearDown(self):
        self.temporary.cleanup()

    def write_review(self, exclude=False):
        review = {"schema_version": 1,
                  "frozen_manifest": {"sha256": self.manifest.sha256, "manifest_id": None},
                  "entries": [{"id": "amber:1", "split": "train", "decision": "exclude" if exclude else "retain",
                               "image": "image.png", "question": "Is there a bicycle?", "original_label": "no",
                               "category": "existence", "image_sha256": self.image_sha,
                               "rationale": "Reviewed example", "confidence": "high"}],
                  "excluded_ids": ["amber:1"] if exclude else [],
                  "retained_reviewed_ids": [] if exclude else ["amber:1"]}
        review["overlay_id"] = content_id(review)
        path = self.run / "training_review.json"
        path.write_text(json.dumps(review))
        self.row.update(review_overlay_id=review["overlay_id"], review_overlay_sha256=file_sha256(path))

    def convert(self):
        (self.run / "rollouts.jsonl").write_text(json.dumps(self.row) + "\n")
        with patch.object(sys, "argv", ["linearizer", "--data-str", str(self.run), "--data-root", str(self.root),
                                         "--output-dir", str(self.root / "sft")]):
            with contextlib.redirect_stdout(StringIO()):
                linearizer.main()

    def test_reviewed_geometry_is_preserved_and_sft_points_are_pixels(self):
        self.write_review()
        self.convert()
        entries = json.loads((self.root / "sft/reasoning_chains_train.json").read_text())
        self.assertEqual(len(entries), 1)
        metadata = entries[0]["metadata"]
        self.assertEqual(metadata["review_overlay_id"], self.row["review_overlay_id"])
        self.assertEqual(metadata["coordinate_frame"], "original_pixels")
        self.assertEqual(metadata["student_prompt_version"], "amber_grounded_full_response_v1")
        self.assertEqual(metadata["source_image_geometry"]["coordinate_frame"], "normalized_1000")
        self.assertIn("(250, 166)", entries[0]["messages"][2]["content"])
        self.assertNotIn("normalized 0-1000", entries[0]["messages"][0]["content"])

    def test_excluded_or_forged_review_cannot_be_converted(self):
        self.write_review(exclude=True)
        with self.assertRaisesRegex(ValueError, "excluded"):
            self.convert()
        self.write_review(exclude=False)
        self.row["review_overlay_sha256"] = "forged"
        with self.assertRaisesRegex(ValueError, "incompatible review_overlay_sha256"):
            self.convert()

    def test_reviewed_rows_require_their_overlay_file(self):
        self.write_review()
        (self.run / "training_review.json").unlink()
        with self.assertRaisesRegex(ValueError, "require their --review-overlay"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
