"""Regression coverage for hallucination labels and the teacher request."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from vlmsearch.reward_funcs.answers import parse_final_answer, strict_match_answer
from vlmsearch.reward_funcs.judge import Judge


class StrictAnswerTests(unittest.TestCase):
    def test_atomic_binary_and_choice_answers(self):
        for ground, response in [("no", "NO."), ("yes", "<answer> yes </answer>"),
                                 ("C", "<think>I checked the image at (10, 20).</think>\n<answer>C</answer>"),
                                 ("no", "<think>Object count < 2.</think><answer>no</answer>")]:
            with self.subTest(response=response):
                self.assertEqual(strict_match_answer(ground, response), 1)

    def test_substrings_and_ambiguous_outputs_fail(self):
        for response in ("unknown", "No final node found in this rollout.", "yes or no", "no, but yes",
                         "<answer>no</answer><answer>yes</answer>",
                         "<answer>no</answer><answer>no</answer>",
                         "<answer>no</answer> yes", "<answer>no", "</answer><answer>no",
                         "<think>no</think>", "<think>unclosed<answer>no</answer>",
                         "<answer>unknown</answer>", "reasoning <answer>no</answer>"):
            with self.subTest(response=response):
                self.assertEqual(strict_match_answer("no", response), 0)

    def test_wrong_choices_and_letter_substrings_fail(self):
        for response in ("cat", "Answer C", "<answer>C or D</answer>", "<answer>C</answer> D", "E"):
            with self.subTest(response=response):
                self.assertEqual(strict_match_answer("C", response), 0)
        self.assertEqual(strict_match_answer("C", "<answer>D</answer>"), 0)

    def test_general_text_is_whole_answer_equality(self):
        self.assertEqual(strict_match_answer("red car", "<answer>Red car.</answer>"), 1)
        self.assertEqual(strict_match_answer("red car", "a red car beside a truck"), 0)

    def test_custom_tags_and_empty_values(self):
        self.assertEqual(strict_match_answer("yes", "<final>YES</final>", "<final>", "</final>"), 1)
        for value in (None, "", " ", "<answer></answer>"):
            self.assertIsNone(parse_final_answer(value))

    def test_native_independent_evidence_prefixes(self):
        for response in (
            "<evidence>A red car is visible at (8, 8).</evidence><answer>yes</answer>",
            "<evidence>Object count < 2.</evidence>\n<answer>no</answer>",
            "<evidence>I checked the car.</evidence><think>I checked again.</think><answer>yes</answer>",
        ):
            with self.subTest(response=response):
                expected = "no" if "<answer>no" in response else "yes"
                self.assertEqual(strict_match_answer(expected, response), 1)

    def test_mixed_unclosed_or_nested_evidence_is_rejected(self):
        for response in (
            "<evidence>I checked the car.</think><answer>yes</answer>",
            "<think>I checked the car.</evidence><answer>yes</answer>",
            "<evidence>I checked the car.<answer>yes</answer>",
            "<evidence><think>I checked.</think></evidence><answer>yes</answer>",
            "<evidence>I checked.</evidence> free prose <answer>yes</answer>",
            "<evidence>I checked.</evidence><answer>yes</answer><answer>no</answer>",
        ):
            with self.subTest(response=response):
                self.assertEqual(strict_match_answer("yes", response), 0)

    def test_judge_public_entrypoint_uses_strict_parser(self):
        judge = Judge(None, "string_match")
        self.assertEqual(judge.judge("no", "unknown", "Is there a dog?"), 0)
        self.assertEqual(judge.judge("no", "<answer>no</answer><answer>yes</answer>", "Question"), 0)
        self.assertEqual(judge.judge("no", "<answer>no</answer>", "Question"), 1)


class TeacherClientTests(unittest.TestCase):
    def test_force_final_prefills_the_answer_instead_of_starting_evidence_again(self):
        from vlmsearch.models import qwen_vllm
        fake = MagicMock()
        fake.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" no </answer>"))])
        with patch.object(qwen_vllm, "client", fake):
            model = qwen_vllm.Qwen_VLLM(top_p=1, repetition_penalty=1.05,
                                        thought_token_begin="<evidence>", thought_token_end="</evidence>",
                                        final_token_begin="<answer>", final_token_end="</answer>")
            response = model.generate_single_thought("Check.", [("Is it red?", Image.new("RGB", (16, 16), "blue")),
                                                               ("<evidence>The image is blue.</evidence>", None)], force_final=True)
        self.assertEqual(strict_match_answer("no", response), 1)
        call = fake.chat.completions.create.call_args.kwargs
        self.assertEqual(call["messages"][-1]["role"], "assistant")
        self.assertEqual(call["messages"][-1]["content"][0]["text"], "<answer>")
        self.assertTrue(call["extra_body"]["continue_final_message"])
        self.assertFalse(call["extra_body"]["add_generation_prompt"])
        self.assertEqual(call["stop"], ["</answer>"])

    def test_evidence_protocol_stops_after_one_complete_block(self):
        from vlmsearch.models import qwen_vllm
        fake = MagicMock()
        text = "<evidence>A red area is visible at (8, 8).</evidence>"
        fake.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        with patch.object(qwen_vllm, "client", fake):
            model = qwen_vllm.Qwen_VLLM(top_p=1, repetition_penalty=1.05,
                                        thought_token_begin="<evidence>", thought_token_end="</evidence>",
                                        final_token_begin="<answer>", final_token_end="</answer>")
            self.assertEqual(model.generate_single_thought("Check.", [("Is it red?", Image.new("RGB", (16, 16), "red"))]), text)
        call = fake.chat.completions.create.call_args.kwargs
        self.assertEqual(call["stop"], ["</evidence>", "</answer>"])
        self.assertTrue(call["extra_body"]["include_stop_str_in_output"])

    def test_literal_thoughts_are_preserved_and_native_thinking_is_disabled(self):
        from vlmsearch.models import qwen_vllm
        fake = MagicMock()
        text = "<think>I see a red area at (8, 8).</think>"
        fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
        with patch.object(qwen_vllm, "client", fake):
            model = qwen_vllm.Qwen_VLLM(top_p=1, top_k=10, repetition_penalty=1.05)
            answer = model.generate_single_thought("Check visible evidence.", [("Is it red?", Image.new("RGB", (16, 16), "red"))])
        self.assertEqual(answer, text)
        body = fake.chat.completions.create.call_args.kwargs["extra_body"]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["top_k"], 10)


if __name__ == "__main__":
    unittest.main()
