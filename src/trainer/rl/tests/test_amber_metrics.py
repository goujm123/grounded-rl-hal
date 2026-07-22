import pytest

from verl.trainer.metrics import compute_amber_validation_metrics, compute_group_reward_metrics


def test_group_reward_metrics_expose_zero_variance_fraction() -> None:
    metrics = compute_group_reward_metrics(
        uids=["first", "first", "second", "second"],
        overall_scores=[0.0, 1.0, 0.5, 0.5],
        accuracy_scores=[0.0, 1.0, 1.0, 1.0],
    )

    assert metrics["grpo/reward_group_std_mean"] == pytest.approx(2**-0.5 / 2)
    assert metrics["grpo/reward_zero_variance_fraction"] == pytest.approx(0.5)
    assert metrics["grpo/accuracy_group_std_mean"] == pytest.approx(2**-0.5 / 2)
    assert metrics["grpo/accuracy_zero_variance_fraction"] == pytest.approx(0.5)


def test_group_reward_metrics_reject_singleton_groups() -> None:
    with pytest.raises(ValueError, match="at least two rollouts"):
        compute_group_reward_metrics(["first"], [1.0])


def test_amber_validation_metrics_include_no_class_and_categories() -> None:
    metrics = compute_amber_validation_metrics(
        reward_metrics={
            "accuracy": [1.0, 1.0, 1.0, 0.0],
            "answer_valid": [1.0, 1.0, 1.0, 0.0],
            "predicted_no": [0.0, 1.0, 1.0, 0.0],
            "label_no": [0.0, 1.0, 0.0, 1.0],
            "correct_no": [0.0, 1.0, 0.0, 0.0],
        },
        categories=["existence", "existence", "relation", "relation"],
        amber_types=[
            "discriminative-hallucination",
            "discriminative-hallucination",
            "discriminative-relation",
            "discriminative-relation",
        ],
    )

    assert metrics["amber/accuracy"] == pytest.approx(0.75)
    assert metrics["amber/invalid_answer_rate"] == pytest.approx(0.25)
    assert metrics["amber/no_precision"] == pytest.approx(0.5)
    assert metrics["amber/no_recall"] == pytest.approx(0.5)
    assert metrics["amber/no_f1"] == pytest.approx(0.5)
    assert metrics["amber/category_accuracy/existence"] == pytest.approx(1.0)
    assert metrics["amber/category_accuracy/relation"] == pytest.approx(0.5)


def test_non_amber_reward_metrics_are_ignored() -> None:
    assert compute_amber_validation_metrics({"accuracy": [1.0]}, [], []) == {}