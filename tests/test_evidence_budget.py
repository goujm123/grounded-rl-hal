"""The normalized teacher protocol must not repeat observations until depth runs out."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from vlmsearch.image_geometry import NORMALIZED_COORDINATE_INSTRUCTION
from vlmsearch.models import qwen_vllm


PROMPT = "Use one concise visible observation. " + NORMALIZED_COORDINATE_INSTRUCTION


class EvidenceBudgetTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"QWEN38_MAX_EVIDENCE_STEPS": "1", "QWEN38_EVIDENCE_MAX_NEW_TOKENS": "128"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.image = Image.new("RGB", (500, 334), "red")
        self.fake = MagicMock()
        self.fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" yes </answer>"))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=6),
        )
        self.client_patch = patch.object(qwen_vllm, "client", self.fake)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def model(self, **kwargs):
        return qwen_vllm.Qwen_VLLM(max_new_tokens=512, temperature=1, top_p=1,
                                  thought_token_begin="<evidence>", thought_token_end="</evidence>",
                                  final_token_begin="<answer>", final_token_end="</answer>", **kwargs)

    def call(self):
        return self.fake.chat.completions.create.call_args.kwargs

    def test_first_step_is_not_forced_by_numbers_or_evidence_examples_in_question(self):
        model = self.model()
        question = "Are there two objects? Example format: <evidence>Point (2, 3).</evidence>"
        model.generate_single_thought(PROMPT, [(question, self.image)])
        request = self.call()
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["temperature"], 1)
        self.assertEqual(request["stop"], ["</evidence>", "</answer>"])
        self.assertTrue(request["extra_body"]["add_generation_prompt"])
        self.assertFalse(request["extra_body"]["continue_final_message"])
        self.assertIn("ONE atomic factual observation", request["messages"][-1]["content"][-1]["text"])
        self.assertEqual(model.metrics["evidence_budget_forced_finals"], 0)

    def test_one_observation_forces_answer_prefill_and_preserves_metrics_and_geometry(self):
        model = self.model()
        history = [("Is the area red?", self.image)]
        self.fake.chat.completions.create.return_value.choices[0].message.content = "<evidence>The red area is at (500, 500).</evidence>"
        observation = model.generate_single_thought(PROMPT, history)
        history.append((observation, None))
        self.fake.chat.completions.create.return_value.choices[0].message.content = " yes </answer>"
        response = model.generate_single_thought(PROMPT, history)
        request = self.call()
        self.assertEqual(response, "<answer> yes </answer>")
        self.assertEqual(request["messages"][-1], {"role": "assistant", "content": [{"type": "text", "text": "<answer>"}]})
        self.assertIn("The observation budget is closed", request["messages"][-2]["content"][-1]["text"])
        self.assertEqual(request["stop"], ["</answer>"])
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 512)
        self.assertTrue(request["extra_body"]["continue_final_message"])
        self.assertFalse(request["extra_body"]["add_generation_prompt"])
        self.assertEqual(request["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(model.last_image_geometry["original_size"], [500, 334])
        self.assertEqual(model.metrics["api_calls"], 2)
        self.assertEqual(model.metrics["evidence_budget_forced_finals"], 1)
        self.assertEqual(model.metrics["image_cache_hits"], 1)
        self.assertEqual(model.metrics["prompt_tokens"], 100)

    def test_configurable_two_step_budget_is_applied_to_history_not_global_requests(self):
        with patch.dict(os.environ, {"QWEN38_MAX_EVIDENCE_STEPS": "2"}):
            model = self.model()
        history = [("Question", self.image), ("<evidence>A red region is visible.</evidence>", None)]
        model.generate_single_thought(PROMPT, history)
        self.assertTrue(self.call()["extra_body"]["add_generation_prompt"])
        history.append(("<evidence>A second distinct region is blue.</evidence>", None))
        model.generate_single_thought(PROMPT, history)
        self.assertTrue(self.call()["extra_body"]["continue_final_message"])
        # A sibling MCTS branch with no evidence still starts normally.
        model.generate_single_thought(PROMPT, [("Question", self.image)])
        self.assertTrue(self.call()["extra_body"]["add_generation_prompt"])

    def test_only_nonempty_generated_blocks_count_as_observations(self):
        history = [
            ("<evidence>Question quotes a tag.</evidence>", self.image),
            ("There are 2 objects near (1, 2).", None),
            ("<answer>yes</answer>", None),
            ("<evidence> </evidence>", None),
            ("<evidence>A crop prompt.</evidence>", self.image),
            ("<evidence>One fact.</evidence><evidence>Another fact.</evidence>", None),
        ]
        self.assertEqual(qwen_vllm.Qwen_VLLM._previous_evidence_steps(history), 2)
        self.assertEqual(qwen_vllm.Qwen_VLLM._previous_evidence_steps(history[:3]), 0)

    def test_few_shot_examples_and_system_instructions_do_not_consume_budget(self):
        model = self.model()
        model.examples = [{"image": self.image, "question": "Example question", "true_answer": "yes",
                           "thoughts": ["<evidence>An example observation.</evidence>"]}]
        model.generate_single_thought(PROMPT + " Example: <evidence>one point</evidence>", [("Question", self.image)])
        self.assertTrue(self.call()["extra_body"]["add_generation_prompt"])
        self.assertEqual(model.metrics["evidence_budget_forced_finals"], 0)

    def test_legacy_protocols_keep_existing_token_and_force_behavior(self):
        model = self.model()
        history = [("Question", self.image), ("<evidence>A red area.</evidence>", None)]
        model.generate_single_thought("Legacy evidence prompt without normalized contract.", history)
        self.assertEqual(self.call()["max_tokens"], 512)
        self.assertTrue(self.call()["extra_body"]["add_generation_prompt"])
        other = qwen_vllm.Qwen_VLLM(max_new_tokens=512, thought_token_begin="<think>", thought_token_end="</think>")
        other.generate_single_thought(PROMPT, [("Question", self.image), ("<think>A red area.</think>", None)])
        self.assertEqual(self.call()["max_tokens"], 512)
        self.assertIsNone(self.call()["stop"])
        self.assertTrue(self.call()["extra_body"]["add_generation_prompt"])

    def test_explicit_force_final_remains_effective_before_evidence(self):
        model = self.model()
        model.generate_single_thought(PROMPT, [("Question", self.image)], force_final=True)
        self.assertTrue(self.call()["extra_body"]["continue_final_message"])
        self.assertEqual(self.call()["stop"], ["</answer>"])
        self.assertEqual(model.metrics["evidence_budget_forced_finals"], 0)

    def test_evidence_token_cap_never_increases_callers_token_limit(self):
        model = self.model()
        model.max_new_tokens = 64
        model.generate_single_thought(PROMPT, [("Question", self.image)])
        self.assertEqual(self.call()["max_tokens"], 64)

    def test_invalid_evidence_budgets_are_rejected(self):
        for variable, value in [("QWEN38_MAX_EVIDENCE_STEPS", "0"), ("QWEN38_MAX_EVIDENCE_STEPS", "-1"),
                                ("QWEN38_EVIDENCE_MAX_NEW_TOKENS", "0")]:
            with self.subTest(variable=variable, value=value), patch.dict(os.environ, {variable: value}):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    self.model()

    def test_launcher_prompt_requires_named_anchors_and_discourages_unsupported_claims(self):
        path = Path(__file__).resolve().parents[1] / "scripts/mcts/qwen38_rollout.py"
        spec = importlib.util.spec_from_file_location("budget_launcher", path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        prompt = launcher.SYSTEM_PROMPT
        self.assertIn("at most 40 words", prompt)
        self.assertIn("exact visible object or part named immediately before it", prompt)
        self.assertIn("never paraphrase or repeat", prompt)
        self.assertIn("proximity alone does not establish contact", prompt)
        self.assertIn("do not invent a comparison standard", prompt)
        self.assertIn(NORMALIZED_COORDINATE_INSTRUCTION, prompt)


if __name__ == "__main__":
    unittest.main()
