from pathlib import Path

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.workers.reward.config import RewardConfig
from verl.workers.reward.function import FunctionRewardManager, validate_reward_score


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded" if len(token_ids) else ""


def test_empty_response_stays_at_zero_reward(tmp_path: Path) -> None:
    reward_module = tmp_path / "reward.py"
    reward_module.write_text(
        "def score(response_str, ground_truth):\n"
        "    return {'overall': 0.5 if not response_str else 1.0, 'answer_valid': float(bool(response_str))}\n",
        encoding="utf-8",
    )
    config = RewardConfig(reward_function=f"{reward_module}:score")
    config.post_init()
    manager = FunctionRewardManager(config, FakeTokenizer())
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[0, 0, 0], [4, 5, 0]]),
            "response_mask": torch.tensor([[0, 0, 0], [1, 1, 0]]),
        },
        non_tensors={"ground_truth": np.array(["yes", "yes"], dtype=object)},
    )

    reward_tensor, metrics = manager.compute_reward(data)

    assert reward_tensor[0].tolist() == [0.0, 0.0, 0.0]
    assert reward_tensor[1].tolist() == [0.0, 1.0, 0.0]
    assert metrics["overall"] == [0.0, 1.0]
    assert metrics["answer_valid"] == [0.0, 1.0]


@pytest.mark.parametrize(
    "score",
    [
        {},
        {"overall": float("nan")},
        {"overall": float("inf")},
        {"overall": "one"},
    ],
)
def test_reward_score_validation_rejects_invalid_values(score) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_reward_score(score)