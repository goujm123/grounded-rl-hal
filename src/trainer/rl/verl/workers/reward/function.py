# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import math
import numbers
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Mapping, Tuple

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


RewardScore = Dict[str, float]
RewardFunction = Callable[[str, str], RewardScore]


def validate_reward_score(score: Mapping[str, float]) -> RewardScore:
    if not isinstance(score, Mapping) or "overall" not in score:
        raise ValueError("Reward function must return a mapping containing `overall`.")

    validated_score = {}
    for key, value in score.items():
        if not isinstance(key, str) or not isinstance(value, numbers.Real):
            raise TypeError(f"Reward metric {key!r} must be a real number, got {value!r}.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Reward metric {key!r} must be finite, got {value!r}.")
        validated_score[key] = numeric_value
    return validated_score


class FunctionRewardManager:
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn: RewardFunction = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        print_first_n = 5
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            # valid_response_length = response_mask.sum()
            # get last non-zero index
            valid_response_length = response_mask.nonzero()[-1].item() + 1 if torch.any(response_mask) else 0
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]

            score = validate_reward_score(self.reward_fn(response_str, ground_truth))
            if valid_response_length == 0:
                score["overall"] = 0.0

            if i < print_first_n:
                print(f"response_str: {response_str}; score: {score}")
            
            # print(f"response_str: {response_str}; valid_response_length: {valid_response_length}; score: {score}")
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
