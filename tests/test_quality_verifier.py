"""Gold-blind critic, point coverage, fail-closed cache and source-admission tests."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from vlmsearch.data_manifest import file_sha256
from vlmsearch.data_review import content_id
from vlmsearch.image_geometry import ImagePayloadCache
from vlmsearch import quality_verifier as verifier_module
from vlmsearch.quality_verifier import QualityVerifier, VerifierConfig, content_hash, strict_json, observability_risk_targets


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quality_filter", ROOT / "scripts/mcts/filter_reasoning_quality.py")
quality_filter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_filter)


class FakeCritic:
    def __init__(self, changes=None, sleep=0):
        self.calls = []
        self.changes = changes or {}
        self.sleep = sleep
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        time.sleep(self.sleep)
        self.calls.append(kwargs)
        stage = kwargs["response_format"]["json_schema"]["name"]
        payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
        if stage == "blind_answer":
            value = {"label": "yes", "unambiguous": True, "facts": ["A red square is visible."], "reason": "Visible red square."}
        elif stage == "scene_claims":
            observations = payload["observations"]
            def checks(observation):
                risks = [r for r in payload.get("observability_risk_targets", []) if r["observation_id"] == observation["observation_id"]]
                if not risks:
                    return []
                return [{"risk_target_ids": [r["risk_target_id"] for r in risks], "quote": observation["text"],
                         "assertion_scope": "physical_or_hidden" if any(r["requires_physical_scope"] for r in risks) else "visible_scene_only",
                         "visibility_basis": "direct_visible_evidence", "literal_scope_preserved": True,
                         "scale_reference_kind": "none_or_undetermined" if any(r.get("requires_visible_scale_reference") for r in risks) else "not_applicable",
                         "scale_reference_evidence": "No applicable visible scale reference in this mock adjudication.",
                         "motion_evidence_kind": "blur_or_sharpness_only" if any(r.get("requires_independent_motion_evidence") for r in risks) else "not_applicable",
                         "status": "supported", "reason": "Directly visible evidence establishes the claim."}]
            value = {"image_answer": "yes", "question_unambiguous": True,
                     "observations": [{"observation_id": o["observation_id"], "full_text": o["text"],
                                       "observability_checks": checks(o),
                                       "all_material_claims_checked": True, "whole_observation_support": "clear", "reason": "The whole statement is visible."} for o in observations],
                     "claims": [{"observation_id": o["observation_id"], "quote": o["text"], "status": "supported", "reason": "Pixels support the claim."} for o in observations],
                     "all_observations_checked": True, "explanation_support": "clear", "answer_support": "clear",
                     "repetition": False, "repetition_reason": "No repeated assertion.", "summary": "Supported."}
        elif stage == "fine_attributes":
            value = {"attributes": [{"target_id": t["target_id"], "observed_value": {"orientation": "horizontal", "lying_posture": "side_lying", "material": "dirt", "chromatic_color": "red"}[t["feature_kind"]], "clarity": "clear", "visible_basis": "Directly identifiable image feature."} for t in payload["attribute_targets"]]}
        elif stage == "blind_centers":
            value = {"points": [{"point_id": p["point_id"], "center_description": "Red square surface.",
                                  "clarity": "clear", "reason": "The center is visibly red."} for p in payload["points"]]}
        elif stage == "point_referents":
            value = {"points": [{"point_id": p["point_id"], "referent_quote": "red square",
                                  "checked_attachment_quote": p["attachment"]["attachment_quote"], "attachment_scope_preserved": True, "status": "correct", "precision": "precise", "reason": "The center is on the named square."} for p in payload["point_claims"]]}
        else:
            raise AssertionError(stage)
        if stage in self.changes:
            value = self.changes[stage](value)
        raw = value if isinstance(value, str) else json.dumps(value)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=raw))],
                               usage=SimpleNamespace(prompt_tokens=30, completion_tokens=20))


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (64, 64), "red").save(self.image)
        self.sha = file_sha256(self.image)
        self.chain = "<think>A red square is visible at (32, 32).</think><answer>yes</answer>"

    def verifier(self, fake=None, **config):
        return QualityVerifier(VerifierConfig(**config), self.root / "cache", client=fake or FakeCritic(),
                               model_identity={"revision": "pinned-test", "identity_file_sha256": {"config": "fixture"}}, verify_endpoint=False)

    def verify(self, verifier, chain=None):
        return verifier.verify_candidate(self.image, "Is there a red square?", chain or self.chain, self.sha,
                                          original_size=(64, 64), source_fingerprints={"must_not_leak_gold": "SECRET_LABEL"})

    def test_material_claims_outside_the_single_think_block_cannot_be_ignored(self):
        extra = "<evidence>An unsupported extra claim.</evidence>" + self.chain
        with self.assertRaisesRegex(ValueError, "one valid complete"):
            self.verify(self.verifier(), extra)

    def test_gold_blind_independent_passes_and_cache_reuse(self):
        fake = FakeCritic()
        verifier = self.verifier(fake)
        first = self.verify(verifier)
        self.assertTrue(first["accepted"], first["reasons"])
        self.assertEqual(len(fake.calls), 5)
        payloads = [json.loads(c["messages"][1]["content"][0]["text"]) for c in fake.calls]
        self.assertEqual(set(payloads[0]), {"question"})
        self.assertNotIn("candidate", json.dumps(payloads[2]))
        self.assertIn("[point_1]", payloads[1]["observations"][0]["text"])
        self.assertNotIn("(32, 32)", payloads[1]["observations"][0]["text"])
        self.assertEqual(payloads[3]["coordinate_frame"], "original_pixels")
        self.assertNotIn("SECRET_LABEL", json.dumps(fake.calls))
        self.assertIn("blind_center_observations", payloads[4])
        self.assertTrue(self.verify(verifier)["accepted"])
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual(verifier.metrics["cache_hits"], 5)
        self.assertEqual(file_sha256(self.image), self.sha)

    def test_unsupported_extra_claim_rejects_even_when_answer_is_supported(self):
        def unsupported(value):
            value["claims"][0]["status"] = "unsupported"
            return value
        result = self.verify(self.verifier(FakeCritic({"scene_claims": unsupported})))
        self.assertFalse(result["accepted"])
        self.assertIn("material_claim_unsupported", result["reasons"])

    def test_fine_orientation_is_blindly_measured_even_if_scene_passes(self):
        chain = "<think>A square is visible; the paddle is held vertically.</think><answer>yes</answer>"
        fake = FakeCritic({"fine_attributes": lambda v: {"attributes": [
            {**v["attributes"][0], "observed_value": "diagonal", "visible_basis": "Shaft slopes across the image."}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertFalse(result["accepted"])
        self.assertIn("fine_attribute_value_mismatch", result["reasons"])
        call = next(c for c in fake.calls if c["response_format"]["json_schema"]["name"] == "fine_attributes")
        payload = json.loads(call["messages"][1]["content"][0]["text"])
        public = json.dumps(payload)
        self.assertNotIn("vertically", public)
        self.assertNotIn("expected_value", public)
        self.assertNotIn("source_quote", public)
        self.assertIn("[ATTRIBUTE]", public)
        self.assertEqual(result["fine_attribute_targets"][0]["expected_value"], "vertical")

    def test_chromatic_inventory_is_explicit_limited_and_backward_compatible(self):
        observations = ["Pink flowers and red, orange, yellow, green, blue, purple, brown panels are visible.",
                        "A black train has a white front and gray wheels.", "The sky is blue-gray."]
        targets = verifier_module.fine_attribute_targets(observations)
        self.assertEqual([t["expected_value"] for t in targets],
                         ["pink", "red", "orange", "yellow", "green", "blue", "purple", "brown", "blue_gray"])
        self.assertTrue(all(t["feature_kind"] == "chromatic_color" for t in targets))
        self.assertEqual(verifier_module.fine_attribute_targets(observations, include_color=False), [])

    def test_neutral_object_with_small_colored_patch_cannot_satisfy_whole_object_color(self):
        chain = "<think>Only one train is visible, with a pink locomotive in front.</think><answer>yes</answer>"
        fake = FakeCritic({"fine_attributes": lambda v: {"attributes": [
            {**v["attributes"][0], "observed_value": "black", "visible_basis": "The main body is black; only its small front panel is red."}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["accepted"])
        self.assertIn("fine_attribute_value_mismatch", result["reasons"])
        call = next(c for c in fake.calls if c["response_format"]["json_schema"]["name"] == "fine_attributes")
        payload = json.loads(call["messages"][1]["content"][0]["text"])
        self.assertNotIn("pink", json.dumps(payload))
        self.assertIn("[ATTRIBUTE] locomotive", payload["attribute_targets"][0]["masked_context"])
        self.assertIn("dominant visible surface", call["messages"][0]["content"])

    def test_correct_explicit_green_color_remains_eligible(self):
        chain = "<think>The grass is green.</think><answer>yes</answer>"
        fake = FakeCritic({"fine_attributes": lambda v: {"attributes": [
            {**v["attributes"][0], "observed_value": "green", "visible_basis": "The grass surface is predominantly green."}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertTrue(result["accepted"], result["reasons"])

    def test_named_colored_part_is_checked_without_coloring_the_whole_object(self):
        chain = "<think>The locomotive's front panel is red.</think><answer>yes</answer>"
        result = self.verify(self.verifier(), chain)
        self.assertTrue(result["accepted"], result["reasons"])
        target = result["fine_attribute_targets"][0]
        self.assertIn("front panel", target["masked_context"])
        self.assertIn("assess only that named part", verifier_module.FINE_ATTRIBUTE_PROMPT)

    def test_orange_fruit_name_does_not_invent_a_color_qualifier(self):
        texts = ["One orange is on the table.", "The orange at [point_1] is visible.",
                 "An orange locomotive is visible.", "The fruit is orange."]
        targets = verifier_module.fine_attribute_targets(texts)
        self.assertEqual([t["observation_id"] for t in targets], [3, 4])
        self.assertEqual([t["expected_value"] for t in targets], ["orange", "orange"])

    def test_lying_alone_cannot_establish_a_specific_side_posture(self):
        chain = "<think>The person lies on his side beside a square.</think><answer>yes</answer>"
        fake = FakeCritic({"fine_attributes": lambda v: {"attributes": [
            {**v["attributes"][0], "observed_value": "lying_unspecified", "visible_basis": "Body is horizontal; front versus side is obscured."}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertFalse(result["accepted"])
        self.assertIn("fine_attribute_uncertain", result["reasons"])

    def test_fine_material_uncertainty_is_not_silently_accepted(self):
        chain = "<think>A square is beside a dirt path.</think><answer>yes</answer>"
        fake = FakeCritic({"fine_attributes": lambda v: {"attributes": [
            {**v["attributes"][0], "observed_value": "other", "clarity": "uncertain", "visible_basis": "Debris obscures the underlying surface."}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertFalse(result["accepted"])
        self.assertIn("fine_attribute_uncertain", result["reasons"])

    def test_fine_targets_mask_all_values_and_keep_target_entity_location(self):
        targets = verifier_module.fine_attribute_targets(["A vertical pole stands beside a horizontal rail on a wooden fence."])
        self.assertEqual([t["expected_value"] for t in targets], ["vertical", "horizontal", "wood"])
        self.assertIn("[ATTRIBUTE] pole", targets[0]["masked_context"])
        self.assertIn("[ATTRIBUTE] rail", targets[1]["masked_context"])
        for target in targets:
            self.assertNotRegex(target["masked_context"], r"vertical|horizontal|wooden")
        self.assertFalse(verifier_module.fine_attribute_targets(["A glass is on a table; the bag is on his back."]))

    def test_missing_fine_inventory_is_a_protocol_error_not_a_pass(self):
        chain = "<think>A red square is on a horizontal surface.</think><answer>yes</answer>"
        result = self.verify(self.verifier(FakeCritic({"fine_attributes": lambda v: {"attributes": []}})), chain)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["verification_complete"])
        self.assertEqual(result["failed_stage"], "fine_attributes")

    def test_fine_negation_is_preserved_without_forcing_the_positive_value(self):
        chain = "<think>The paddle is not vertical beside a square.</think><answer>yes</answer>"
        # FakeCritic independently describes a horizontal axis.
        result = self.verify(self.verifier(), chain)
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertTrue(result["fine_attribute_targets"][0]["negated"])

    def test_posture_front_and_back_direction_have_anatomical_meaning(self):
        descriptions = ["The person lies on his back.", "The person lies back-up.",
                        "The person lies back-down.", "The person lies face-up.",
                        "The person lies face-down.", "The person lies front-up."]
        targets = verifier_module.fine_attribute_targets(descriptions)
        self.assertEqual([t["expected_value"] for t in targets],
                         ["supine", "prone", "supine", "supine", "prone", "supine"])

    def test_fine_negation_scope_preserves_additive_and_longer_pose_phrases(self):
        descriptions = ["The person is not lying down on his side.",
                        "The pole is not only vertical.", "The pole is not just vertical.",
                        "The pole is not necessarily vertical."]
        targets = verifier_module.fine_attribute_targets(descriptions)
        self.assertEqual([t["negated"] for t in targets], [True, False, False, False])
        self.assertEqual([t["polarity_uncertain"] for t in targets], [False, False, False, True])

    def test_local_part_attachment_cannot_be_replaced_by_broad_entity(self):
        chain = "<think>A white sheep stands with its hooves resting directly on the grass at (32, 32).</think><answer>yes</answer>"
        fake = FakeCritic({"point_referents": lambda v: {"points": [
            {**v["points"][0], "referent_quote": "white sheep"}]}})
        result = self.verify(self.verifier(fake), chain)
        self.assertFalse(result["accepted"])
        self.assertIn("point_specific_feature_broadened", result["reasons"])
        call = next(c for c in fake.calls if c["response_format"]["json_schema"]["name"] == "point_referents")
        point = json.loads(call["messages"][1]["content"][0]["text"])["point_claims"][0]
        self.assertEqual(point["attachment"]["scope"], "feature")
        self.assertEqual(point["attachment"]["required_feature_terms"], ["hooves"])

    def test_broad_action_anchor_does_not_require_unclaimed_diagnostic_limb(self):
        text = "A person is jumping with legs bent and body extended upward at [point_1]."
        attachment = verifier_module.point_attachment(text, 1)
        self.assertEqual(attachment["scope"], "object")
        self.assertEqual(attachment["required_feature_terms"], [])
        self.assertEqual(verifier_module.point_attachment("A head rests on a pillow at [point_1].", 1)["scope"], "object")

    def test_contact_region_requires_preserved_local_scope(self):
        text = "The red square rests directly on a plate around [point_1]."
        self.assertEqual(verifier_module.point_attachment(text, 1)["scope"], "contact_region")
        fake = FakeCritic({"point_referents": lambda v: {"points": [
            {**v["points"][0], "attachment_scope_preserved": False}]}})
        result = self.verify(self.verifier(fake), "<think>" + text.replace("[point_1]", "(32, 32)") + "</think><answer>yes</answer>")
        self.assertFalse(result["accepted"])
        self.assertIn("point_attachment_scope_not_preserved", result["reasons"])

    def test_separately_anchored_relation_arguments_do_not_force_contact_centers(self):
        text = "The toy [point_1] sits directly on the bicycle crossbar [point_2]."
        first = verifier_module.point_attachment(text, 1)
        second = verifier_module.point_attachment(text, 2)
        self.assertEqual(first["scope"], "object")
        self.assertEqual(second["scope"], "object")
        explicit = "The toy [point_1] rests directly on the plate at the contact region [point_2]."
        self.assertEqual(verifier_module.point_attachment(explicit, 2)["scope"], "contact_region")

    def test_coordinate_lists_preserve_the_same_feature_attachment(self):
        text = "The hooves at [point_1], [point_2] meet the grass."
        first = verifier_module.point_attachment(text, 1)
        second = verifier_module.point_attachment(text, 2)
        self.assertEqual(first["scope"], "feature")
        self.assertEqual(first, second)

    def test_whole_observation_completeness_is_mandatory(self):
        for change in [lambda v: {**v, "observations": []},
                       lambda v: {**v, "observations": [{**v["observations"][0], "all_material_claims_checked": False}]},
                       lambda v: {**v, "observations": [{**v["observations"][0], "full_text": "A fragment."}]}]:
            with tempfile.TemporaryDirectory() as directory:
                verifier = self.verifier(FakeCritic({"scene_claims": change}))
                verifier.cache_dir = Path(directory)
                self.assertFalse(self.verify(verifier)["accepted"])

    def test_incorrect_or_imprecise_point_rejects(self):
        for status, precision in [("incorrect", "precise"), ("correct", "imprecise"), ("uncertain", "uncertain")]:
            def change(value):
                value["points"][0].update(status=status, precision=precision)
                return value
            with tempfile.TemporaryDirectory() as directory:
                verifier = self.verifier(FakeCritic({"point_referents": change}))
                verifier.cache_dir = Path(directory)
                result = self.verify(verifier)
                self.assertFalse(result["accepted"])
                self.assertIn("point_referent_not_precisely_correct", result["reasons"])

    def test_uncertain_blind_answer_and_centers_fail_closed(self):
        for changes in [{"blind_answer": lambda v: {**v, "label": "uncertain"}},
                        {"blind_centers": lambda v: {"points": [{**v["points"][0], "clarity": "uncertain"}]}}]:
            with tempfile.TemporaryDirectory() as directory:
                verifier = self.verifier(FakeCritic(changes))
                verifier.cache_dir = Path(directory)
                self.assertFalse(self.verify(verifier)["accepted"])

    def test_every_point_occurrence_is_checked_with_eight_image_limit(self):
        chain = "<think>A red square is visible at " + ", ".join(f"({i}, 32)" for i in range(1, 8)) + ".</think><answer>yes</answer>"
        fake = FakeCritic()
        result = self.verify(self.verifier(fake), chain)
        self.assertTrue(result["accepted"], result["reasons"])
        checked = [p for b in result["passes"]["localization_batches"] for p in b["point_ids"]]
        self.assertEqual(checked, list(range(1, 8)))
        self.assertTrue(all(sum(c["type"] == "image_url" for c in call["messages"][1]["content"]) <= 8 for call in fake.calls))

    def test_repetition_is_reported_separately_from_semantic_accept(self):
        fake = FakeCritic({"scene_claims": lambda v: {**v, "repetition": True}})
        result = self.verify(self.verifier(fake))
        self.assertTrue(result["semantic_accept"])
        self.assertFalse(result["accepted"])
        self.assertIn("repeated_observations", result["reasons"])

    def test_malformed_duplicate_json_and_missing_points_are_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"label":"yes","label":"no"}')
        for changes in [{"scene_claims": lambda v: "```json\n" + json.dumps(v) + "\n```"},
                        {"point_referents": lambda v: {"points": []}}]:
            with tempfile.TemporaryDirectory() as directory:
                verifier = self.verifier(FakeCritic(changes))
                verifier.cache_dir = Path(directory)
                self.assertFalse(self.verify(verifier)["accepted"])

    def test_changed_image_is_rejected_before_model_calls(self):
        fake = FakeCritic()
        verifier = self.verifier(fake)
        Image.new("RGB", (64, 64), "blue").save(self.image)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.verify(verifier)
        self.assertEqual(len(fake.calls), 0)

    def test_tampered_cache_does_not_get_trusted(self):
        verifier = self.verifier()
        result = self.verify(verifier)
        path = Path(result["cache_records"][0]["cache_file"])
        value = json.loads(path.read_text())
        value["raw_response"] = '{}'
        path.write_text(json.dumps(value))
        self.assertFalse(self.verify(verifier)["accepted"])

    def test_concurrent_instances_generate_each_shared_stage_once(self):
        fake = FakeCritic(sleep=.01)
        first, second = self.verifier(fake), self.verifier(fake)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.verify, [first, second]))
        self.assertTrue(all(result["accepted"] for result in results))
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual(first.metrics["api_calls"] + second.metrics["api_calls"], 5)

    def test_parallel_cli_helper_owns_counters_and_preserves_input_order(self):
        fake = FakeCritic(sleep=.01)
        verifier = self.verifier(fake)
        clones = []
        def clone():
            worker = self.verifier(fake)
            clones.append(worker)
            return worker
        with patch.object(verifier, "clone", side_effect=clone):
            results = list(quality_filter.parallel_verdicts(
                ["first", "second"], verifier, 2, lambda worker, item: (item, self.verify(worker))))
        self.assertEqual([item[0] for item in results], ["first", "second"])
        self.assertEqual(len(clones), 2)
        self.assertEqual(verifier.metrics["api_calls"], 5)
        self.assertEqual(verifier.metrics["cache_hits"], 5)

    def test_prompt_policy_distinguishes_broad_anchors_and_coarse_visibility(self):
        scene = verifier_module.SCENE_PROMPT
        referent = verifier_module.REFERENT_PROMPT
        self.assertIn('"Clearly visible" means identifiable, not necessarily fully unoccluded', scene)
        self.assertIn("approximate regions, not exact rectangles", scene)
        self.assertIn("Ignore opaque [point_N] placeholders completely", scene)
        self.assertIn("ANY visible part", referent)
        self.assertIn("MINIMAL source noun span", referent)
        self.assertIn("genuine target-scope ambiguity must be uncertain", referent)
        self.assertIn("most diagnostic feature", referent)
        self.assertEqual(verifier_module.scene_schema(["Observation"])["properties"]["claims"]["maxItems"], 6)

    def test_stage_cache_reuses_unchanged_blind_stages_after_critic_prompt_changes(self):
        fake = FakeCritic()
        first = self.verifier(fake)
        self.assertTrue(self.verify(first)["accepted"])
        with patch.object(verifier_module, "SCENE_PROMPT", verifier_module.SCENE_PROMPT + " Revised generic scene policy."), \
             patch.object(verifier_module, "REFERENT_PROMPT", verifier_module.REFERENT_PROMPT + " Revised generic point policy."):
            second = self.verifier(fake)
            self.assertNotEqual(first.fingerprint, second.fingerprint)
            self.assertTrue(self.verify(second)["accepted"])
        self.assertEqual(second.metrics["api_calls"], 2)
        self.assertEqual(second.metrics["cache_hits"], 3)

    def test_one_bounded_retry_preserves_actual_failed_response_and_finish_reason(self):
        class InitiallyTruncated(FakeCritic):
            def create(self, **kwargs):
                completion = super().create(**kwargs)
                if kwargs["response_format"]["json_schema"]["name"] == "scene_claims" and \
                        sum(c["response_format"]["json_schema"]["name"] == "scene_claims" for c in self.calls) == 1:
                    completion.choices[0].finish_reason = "length"
                    completion.choices[0].message.content = '{"incomplete":'
                return completion
        fake = InitiallyTruncated()
        result = self.verify(self.verifier(fake))
        self.assertTrue(result["accepted"], result["reasons"])
        scene_calls = [c for c in fake.calls if c["response_format"]["json_schema"]["name"] == "scene_claims"]
        self.assertEqual([c["max_tokens"] for c in scene_calls], [3072, 4608])
        records = result["cache_records"][1]["attempt_records"]
        self.assertEqual(len(records), 2)
        failed = json.loads(Path(records[0]["file"]).read_text())
        self.assertEqual(failed["finish_reason"], "length")
        self.assertEqual(failed["raw_response"], '{"incomplete":')

    def test_contradictory_unsupported_whole_requires_protocol_retry_not_automatic_accept(self):
        class InitiallyContradictory(FakeCritic):
            def create(self, **kwargs):
                completion = super().create(**kwargs)
                if kwargs["response_format"]["json_schema"]["name"] == "scene_claims" and \
                        sum(c["response_format"]["json_schema"]["name"] == "scene_claims" for c in self.calls) == 1:
                    value = json.loads(completion.choices[0].message.content)
                    value["observations"][0]["whole_observation_support"] = "unsupported"
                    completion.choices[0].message.content = json.dumps(value)
                return completion
        fake = InitiallyContradictory()
        result = self.verify(self.verifier(fake))
        self.assertTrue(result["accepted"], result["reasons"])
        scene_calls = [c for c in fake.calls if c["response_format"]["json_schema"]["name"] == "scene_claims"]
        self.assertEqual(len(scene_calls), 2)
        failed = json.loads(Path(result["cache_records"][1]["attempt_records"][0]["file"]).read_text())
        self.assertIn("contradicts", failed["error"])

    def test_consistency_retry_can_find_a_real_extra_unsupported_claim(self):
        class FindsUnsupportedClaim(FakeCritic):
            def create(self, **kwargs):
                completion = super().create(**kwargs)
                if kwargs["response_format"]["json_schema"]["name"] == "scene_claims":
                    value = json.loads(completion.choices[0].message.content)
                    value["observations"][0]["whole_observation_support"] = "unsupported"
                    if sum(c["response_format"]["json_schema"]["name"] == "scene_claims" for c in self.calls) > 1:
                        value["claims"][0]["status"] = "unsupported"
                    completion.choices[0].message.content = json.dumps(value)
                return completion
        result = self.verify(self.verifier(FindsUnsupportedClaim()))
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["accepted"])
        self.assertIn("material_claim_unsupported", result["reasons"])

    def test_coarse_color_activity_and_idiom_scope_does_not_relax_precise_claims(self):
        scene = verifier_module.SCENE_PROMPT
        self.assertIn('such as "is moving"', scene)
        self.assertIn('Explicit "both hands", "two hands"', scene)
        self.assertIn("Preserve obvious hue or light/dark contradictions", scene)
        self.assertIn("actual pose does not distinguish", verifier_module.IMAGE_QUESTION_SCOPE_POLICY)
        self.assertIn("closeness, an unresolved gap", verifier_module.IMAGE_QUESTION_SCOPE_POLICY)

    def test_exhausted_protocol_retry_is_an_error_not_a_semantic_rejection(self):
        class AlwaysTruncated(FakeCritic):
            def create(self, **kwargs):
                completion = super().create(**kwargs)
                if kwargs["response_format"]["json_schema"]["name"] == "scene_claims":
                    completion.choices[0].finish_reason = "length"
                return completion
        fake = AlwaysTruncated()
        verifier = self.verifier(fake)
        result = self.verify(verifier)
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["semantic_accept"])
        self.assertFalse(result["verification_complete"])
        self.assertEqual(result["status"], "verifier_error")
        self.assertEqual(len(result["failed_attempt_records"]), 2)
        calls = len(fake.calls)
        self.verify(verifier)
        self.assertEqual(len(fake.calls), calls)

    def test_generic_observability_risks_are_checks_not_automatic_rejections(self):
        risks = observability_risk_targets(["A shallow layer and its bottom are visible beneath the red square.",
                                           "No other red square is visible."])
        self.assertTrue(any(r["kind"] == "depth" for r in risks))
        self.assertTrue(any(r["kind"] == "occlusion_or_hidden_location" for r in risks))
        self.assertFalse(any(r["observation_id"] == 2 for r in risks))
        chain = "<think>A red square is visible above a shallow layer.</think><answer>yes</answer>"
        # A mock affirmative image adjudication proves the lexical flag is not
        # itself a veto; real acceptance still requires the supplied pixels.
        self.assertTrue(self.verify(self.verifier(), chain)["accepted"])

    def test_physical_units_require_scope_checks_but_body_part_feet_do_not(self):
        risks = observability_risk_targets(["The person is several meters away.", "A six-foot-wide board is visible.",
                                           "The ruler indicates 2 cm.", "The dog has two feet touching the sand."])
        measurements = [r for r in risks if r["kind"] == "physical_measurement"]
        self.assertEqual({r["observation_id"] for r in measurements}, {1, 2, 3})
        self.assertTrue(all(r["requires_physical_scope"] and r["requires_visible_scale_reference"] for r in measurements))

    def test_unmeasured_meters_cannot_pass_with_a_scene_scan(self):
        chain = "<think>The red square is several meters away.</think><answer>yes</answer>"
        result = self.verify(self.verifier(), chain)
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["accepted"])
        self.assertIn("physical_measurement_without_visible_scale_reference", result["reasons"])

    def test_visible_ruler_measurement_is_not_a_lexical_unit_veto(self):
        chain = "<think>The red square is 2 cm wide beside a calibrated ruler.</think><answer>yes</answer>"
        def calibrated(value):
            for observation in value["observations"]:
                for check in observation["observability_checks"]:
                    check["scale_reference_kind"] = "visible_calibrated_ruler_or_grid"
                    check["scale_reference_evidence"] = "Readable 0 and 2 cm ticks align with the square edges."
            return value
        # The mock explicitly adjudicates a visible calibration. This tests that
        # the policy permits grounded measurements rather than banning unit words.
        result = self.verify(self.verifier(FakeCritic({"scene_claims": calibrated})), chain)
        self.assertTrue(result["accepted"], result["reasons"])

    def test_lack_of_blur_does_not_establish_stillness(self):
        statement = "The red square is not moving because no motion blur is visible."
        risks = observability_risk_targets([statement, "No motion blur is visible.", "A running pose is depicted."])
        motion = [r for r in risks if r["kind"] == "blur_to_motion_inference"]
        self.assertEqual(len(motion), 1)
        self.assertEqual(motion[0]["observation_id"], 1)
        result = self.verify(self.verifier(), "<think>" + statement + "</think><answer>yes</answer>")
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["accepted"])
        self.assertIn("motion_inference_not_established_by_blur_or_pose", result["reasons"])

    def test_image_presence_count_scope_does_not_weaken_hidden_physical_assertions(self):
        for prompt in (verifier_module.BLIND_PROMPT, verifier_module.SCENE_PROMPT):
            self.assertIn("concern visibly depicted instances", prompt)
            self.assertIn("head or partial body can establish one depicted instance", prompt)
            self.assertIn("Do not demand proof that no additional wholly occluded/offscreen instance exists", prompt)
            self.assertIn("If the actual depicted instances cannot be distinguished or counted reliably, retain uncertainty", prompt)
            self.assertIn("ground/support beneath opaque water still needs literal evidence", prompt)
        self.assertNotIn("Image-question scope:", verifier_module.CENTER_PROMPT)
        self.assertNotIn("Image-question scope:", verifier_module.REFERENT_PROMPT)
        risks = observability_risk_targets(["There is no ground beneath the animal."])
        self.assertTrue(any(r["kind"] == "absolute_hidden_absence" and r["requires_physical_scope"] for r in risks))

    def test_hidden_absolute_claim_cannot_be_weakened_into_no_visible_support(self):
        chain = "<think>A red square is visible; there is no ground beneath it.</think><answer>yes</answer>"
        def weaken(value):
            for observation in value["observations"]:
                for check in observation["observability_checks"]:
                    check["assertion_scope"] = "visible_scene_only"
                    check["visibility_basis"] = "visible_scene_scan"
                    check["reason"] = "No ground is visible."
            return value
        result = self.verify(self.verifier(FakeCritic({"scene_claims": weaken})), chain)
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["semantic_accept"])
        self.assertIn("source_assertion_scope_weakened", result["reasons"])

    def test_occluded_basis_and_missing_source_risk_coverage_fail_closed(self):
        chain = "<think>A red square is visible in shallow water.</think><answer>yes</answer>"
        def occluded(value):
            value["observations"][0]["observability_checks"][0]["visibility_basis"] = "occluded"
            return value
        result = self.verify(self.verifier(FakeCritic({"scene_claims": occluded})), chain)
        self.assertTrue(result["verification_complete"])
        self.assertFalse(result["semantic_accept"])
        self.assertIn("unobservable_or_unsupported_material_claim", result["reasons"])
        with tempfile.TemporaryDirectory() as directory:
            def omit(value):
                value["observations"][0]["observability_checks"] = []
                return value
            verifier = self.verifier(FakeCritic({"scene_claims": omit}))
            verifier.cache_dir = Path(directory)
            missing = self.verify(verifier, chain)
            self.assertFalse(missing["accepted"])
            self.assertIsNone(missing["semantic_accept"])
            self.assertEqual(missing["failed_stage"], "scene_claims")


class SourceFilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.sft = self.root / "sft"
        self.sft.mkdir()
        self.image = self.root / "image.png"
        Image.new("RGB", (64, 64), "red").save(self.image)
        sha = file_sha256(self.image)
        sample = {"image": "image.png", "split": "train", "cluster_id": "group", "dataset": "amber",
                  "task": "amber_discriminative", "question": "Is there a red square?", "answer": "yes",
                  "category": "discriminative-attribute-state"}
        manifest = {"images": {"image.png": {"file_sha256": sha, "split": "train", "cluster_id": "group"}},
                    "samples": {"amber:1": sample}}
        manifest["manifest_id"] = content_hash(manifest)
        (self.run / "split_manifest.json").write_text(json.dumps(manifest))
        review = {"schema_version": 1, "frozen_manifest": {"sha256": file_sha256(self.run / "split_manifest.json"),
                  "manifest_id": manifest["manifest_id"]}, "entries": [], "excluded_ids": [], "retained_reviewed_ids": []}
        review["overlay_id"] = content_id(review)
        (self.run / "training_review.json").write_text(json.dumps(review))
        with Image.open(self.image) as image:
            geometry = ImagePayloadCache().encode(image)[1]
        self.row = {"id": "amber:1", "image": "image.png", "question": sample["question"], "true_answer": "yes",
                    "data_split": "train", "image_group": "group", "split_manifest_sha256": file_sha256(self.run / "split_manifest.json"),
                    "dataset": "amber", "task": "amber_discriminative", "review_overlay_id": review["overlay_id"],
                    "review_overlay_sha256": file_sha256(self.run / "training_review.json"), "image_geometry": geometry,
                    "tree": {"value": 1, "rollouts": [{"reward": 1, "final_answer": "yes", "raw_final_answer": "<answer>yes</answer>",
                             "ephemeral_texts": ["<evidence>A red square is visible at (500, 500).</evidence>"]}], "children": []}}
        self.source = self.run / "rollouts.jsonl"
        self.source.write_text(json.dumps(self.row) + "\n")
        self.admission = quality_filter.SourceAdmission(self.root, self.run)
        rebuilt = self.admission.linearizer.build_all_reasoning_for_sample(self.row, original_size=(64, 64))[0][0]
        self.entry = {"id": "entry", "images": ["image.png"], "gt_answer": "yes",
                      "messages": [{"role": "user", "content": sample["question"]}, {"role": "assistant", "content": rebuilt["chain"]}],
                      "metadata": {key: self.row[key] for key in ["data_split", "image_group", "split_manifest_sha256", "dataset", "task", "review_overlay_id", "review_overlay_sha256"]}}
        self.entry["metadata"].update(source_file=str(self.source), source_line=1, sample_key="amber:1", rollout_idx=0,
                                       coordinate_frame="original_pixels", image_geometry=geometry, source_image_geometry=geometry,
                                       original_image_size=[64, 64], evidence_normalization=rebuilt["evidence_normalization"])

    def test_exact_source_derivation_is_required(self):
        self.admission.validate(self.entry)
        changed = deepcopy(self.entry)
        changed["messages"][1]["content"] = changed["messages"][1]["content"].replace("red square", "blue square")
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self.admission.validate(changed)

    def test_outputs_preserve_sources_and_emit_audited_manifest(self):
        for split, entries in [("train", [self.entry]), ("val", []), ("test", [])]:
            (self.sft / f"reasoning_chains_{split}.json").write_text(json.dumps(entries))
        source_hash = file_sha256(self.source)
        verifier = QualityVerifier(VerifierConfig(), self.root / "cache", client=FakeCritic(),
                                   model_identity={"revision": "fixture"}, verify_endpoint=False)
        report = quality_filter.filter_sft(self.sft, self.root / "filtered", self.admission, verifier)
        self.assertEqual(report["accepted_chains"], 1)
        self.assertEqual(file_sha256(self.source), source_hash)
        self.assertTrue((self.root / "filtered/quality_manifest.json").is_file())
        kept = json.loads((self.root / "filtered/reasoning_chains_train.json").read_text())
        self.assertEqual(kept[0]["metadata"]["quality_status"], "accepted_by_automated_visual_verifier")
        self.assertEqual(json.loads((self.root / "filtered/reasoning_chains_val.json").read_text()), [])

    def test_all_sources_are_validated_before_any_critic_calls(self):
        tampered = deepcopy(self.entry)
        tampered["messages"][1]["content"] = tampered["messages"][1]["content"].replace("red square", "blue square")
        for split, entries in [("train", [self.entry, tampered]), ("val", []), ("test", [])]:
            (self.sft / f"reasoning_chains_{split}.json").write_text(json.dumps(entries))
        fake = FakeCritic()
        verifier = QualityVerifier(VerifierConfig(), self.root / "cache", client=fake,
                                   model_identity={"revision": "fixture"}, verify_endpoint=False)
        with self.assertRaisesRegex(ValueError, "reproduce"):
            quality_filter.filter_sft(self.sft, self.root / "filtered", self.admission, verifier)
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.root / "filtered").exists())


if __name__ == "__main__":
    unittest.main()
