import importlib.util
from pathlib import Path

import pytest


REWARD_PATH = (
    Path(__file__).resolve().parents[1] / "examples/reward_function/amber_discriminative.py"
)
SPEC = importlib.util.spec_from_file_location("amber_discriminative_reward", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
REWARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARD)


@pytest.mark.parametrize(
    "response",
    [
        "<answer>yes and no</answer>",
        "<answer>yesterday</answer>",
        "<answer>yes</answer><answer>no</answer>",
        "<answer>yes</answer> trailing text",
        "yes",
        "<answer>A. yes</answer>",
    ],
)
def test_answer_parser_rejects_ambiguous_or_nonterminal_answers(response: str) -> None:
    assert REWARD.parse_final_answer(response) is None


def test_reward_requires_accuracy_to_dominate_style() -> None:
    correct_minimal = REWARD.amber_compute_score("<answer>yes</answer>", "yes")
    incorrect_polished = REWARD.amber_compute_score(
        "<think>The object is visible at (10, 20).</think><answer>no</answer>", "yes"
    )

    assert correct_minimal["overall"] == pytest.approx(0.80)
    assert incorrect_polished["overall"] == pytest.approx(0.20)
    assert correct_minimal["overall"] > incorrect_polished["overall"]


def test_complete_correct_response_receives_full_reward() -> None:
    score = REWARD.amber_compute_score(
        "\n<think>The queried object is visible near (12.5, .75).</think>\n<answer> YES </answer>\n",
        "yes",
    )

    assert score == {
        "overall": pytest.approx(1.0),
        "accuracy": 1.0,
        "format": 1.0,
        "coordinate": 1.0,
        "answer_valid": 1.0,
        "predicted_no": 0.0,
        "label_no": 0.0,
        "correct_no": 0.0,
    }


def test_coordinate_must_be_nonnegative_and_inside_think_block() -> None:
    outside = REWARD.amber_compute_score(
        "<think>No coordinate here.</think><answer>yes</answer> (1, 2)", "yes"
    )
    negative = REWARD.amber_compute_score(
        "<think>The region is at (-1, 2).</think><answer>yes</answer>", "yes"
    )

    assert outside["coordinate"] == 0.0
    assert negative["coordinate"] == 0.0


def test_no_class_diagnostics_are_explicit() -> None:
    score = REWARD.amber_compute_score(
        "<think>The claimed item is absent at (4, 5).</think><answer>no</answer>", "no"
    )

    assert score["predicted_no"] == 1.0
    assert score["label_no"] == 1.0
    assert score["correct_no"] == 1.0


def test_invalid_ground_truth_raises() -> None:
    with pytest.raises(ValueError, match="must be yes or no"):
        REWARD.amber_compute_score("<answer>yes</answer>", "maybe")