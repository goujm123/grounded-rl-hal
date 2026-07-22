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

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_group_reward_metrics(
    uids: Sequence[Any],
    overall_scores: Sequence[float],
    accuracy_scores: Optional[Sequence[float]] = None,
    zero_tolerance: float = 1e-12,
) -> Dict[str, float]:
    if len(uids) != len(overall_scores):
        raise ValueError("uids and overall_scores must have the same length.")
    if accuracy_scores is not None and len(uids) != len(accuracy_scores):
        raise ValueError("uids and accuracy_scores must have the same length.")

    grouped_overall = defaultdict(list)
    grouped_accuracy = defaultdict(list)
    for index, uid in enumerate(uids):
        grouped_overall[str(uid)].append(float(overall_scores[index]))
        if accuracy_scores is not None:
            grouped_accuracy[str(uid)].append(float(accuracy_scores[index]))

    if any(len(scores) < 2 for scores in grouped_overall.values()):
        raise ValueError("GRPO reward groups must contain at least two rollouts.")

    def summarize(groups: Mapping[str, Sequence[float]], prefix: str) -> Dict[str, float]:
        standard_deviations = np.array(
            [np.std(scores, ddof=1) for scores in groups.values()], dtype=np.float64
        )
        return {
            f"grpo/{prefix}_group_std_mean": float(np.mean(standard_deviations)),
            f"grpo/{prefix}_zero_variance_fraction": float(
                np.mean(np.isclose(standard_deviations, 0.0, atol=zero_tolerance))
            ),
        }

    metrics = summarize(grouped_overall, "reward")
    if grouped_accuracy:
        metrics.update(summarize(grouped_accuracy, "accuracy"))
    return metrics


def compute_amber_validation_metrics(
    reward_metrics: Mapping[str, Sequence[float]],
    categories: Sequence[Any],
    amber_types: Sequence[Any],
) -> Dict[str, float]:
    required_metrics = {"accuracy", "answer_valid", "predicted_no", "label_no", "correct_no"}
    if not required_metrics.issubset(reward_metrics):
        return {}

    accuracy = np.asarray(reward_metrics["accuracy"], dtype=np.float64)
    answer_valid = np.asarray(reward_metrics["answer_valid"], dtype=np.float64)
    predicted_no = np.asarray(reward_metrics["predicted_no"], dtype=np.float64)
    label_no = np.asarray(reward_metrics["label_no"], dtype=np.float64)
    correct_no = np.asarray(reward_metrics["correct_no"], dtype=np.float64)
    expected_length = len(accuracy)
    lengths = {
        expected_length,
        len(answer_valid),
        len(predicted_no),
        len(label_no),
        len(correct_no),
        len(categories),
        len(amber_types),
    }
    if len(lengths) != 1:
        raise ValueError("AMBER reward diagnostics and metadata must have matching lengths.")
    if expected_length == 0:
        return {}

    predicted_no_count = float(np.sum(predicted_no))
    label_no_count = float(np.sum(label_no))
    correct_no_count = float(np.sum(correct_no))
    no_precision = correct_no_count / predicted_no_count if predicted_no_count else 0.0
    no_recall = correct_no_count / label_no_count if label_no_count else 0.0
    no_f1 = (
        2.0 * no_precision * no_recall / (no_precision + no_recall)
        if no_precision + no_recall
        else 0.0
    )

    metrics = {
        "amber/accuracy": float(np.mean(accuracy)),
        "amber/invalid_answer_rate": float(1.0 - np.mean(answer_valid)),
        "amber/no_precision": no_precision,
        "amber/no_recall": no_recall,
        "amber/no_f1": no_f1,
    }

    def add_grouped_accuracy(values: Sequence[Any], metric_prefix: str) -> None:
        grouped_indices = defaultdict(list)
        for index, value in enumerate(values):
            grouped_indices[str(value)].append(index)
        for value, indices in sorted(grouped_indices.items()):
            metrics[f"amber/{metric_prefix}_accuracy/{value}"] = float(np.mean(accuracy[indices]))

    add_grouped_accuracy(categories, "category")
    add_grouped_accuracy(amber_types, "type")
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    # acc_reward = batch.batch["acc_reward"].sum(-1)
    # format_reward = batch.batch["format_reward"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # "critic/acc_reward/mean": torch.mean(acc_reward).detach().item(),
        # "critic/format_reward/mean": torch.mean(format_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], num_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
