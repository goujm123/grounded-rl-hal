"""Deterministic reward for AMBER discriminative yes/no responses."""

import math
import re
from typing import Dict, Optional, Set, Tuple


ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
STRICT_RESPONSE_RE = re.compile(
    r"\s*<think>(?P<think>.*?)</think>\s*<answer>\s*(?P<answer>yes|no)\s*</answer>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)"
COORDINATE_RE = re.compile(rf"\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)")


def normalize_ground_truth(ground_truth: str) -> str:
    answer = str(ground_truth).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"AMBER ground truth must be yes or no, got {ground_truth!r}.")
    return answer


def parse_final_answer(response: str) -> Optional[str]:
    matches = list(ANSWER_BLOCK_RE.finditer(response))
    if len(matches) != 1 or response[matches[0].end() :].strip():
        return None

    answer = matches[0].group(1).strip().lower()
    return answer if answer in {"yes", "no"} else None


def has_strict_format(response: str) -> bool:
    match = STRICT_RESPONSE_RE.fullmatch(response)
    return bool(match and match.group("think").strip())


def extract_coordinates(response: str) -> Set[Tuple[float, float]]:
    coordinates: Set[Tuple[float, float]] = set()
    for think_block in THINK_BLOCK_RE.findall(response):
        for x_value, y_value in COORDINATE_RE.findall(think_block):
            coordinate = (float(x_value), float(y_value))
            if all(math.isfinite(value) and value >= 0 for value in coordinate):
                coordinates.add(coordinate)
    return coordinates


def amber_compute_score(
    response_str: str,
    ground_truth: str,
    accuracy_weight: float = 0.80,
    format_weight: float = 0.15,
    coordinate_weight: float = 0.05,
) -> Dict[str, float]:
    if not math.isclose(accuracy_weight + format_weight + coordinate_weight, 1.0):
        raise ValueError("AMBER reward weights must sum to 1.0.")
    if min(accuracy_weight, format_weight, coordinate_weight) < 0:
        raise ValueError("AMBER reward weights must be nonnegative.")

    label = normalize_ground_truth(ground_truth)
    predicted_answer = parse_final_answer(response_str)
    answer_valid = float(predicted_answer is not None)
    accuracy = float(predicted_answer == label)
    strict_format = float(has_strict_format(response_str))
    coordinate = float(bool(extract_coordinates(response_str)))
    predicted_no = float(predicted_answer == "no")
    label_no = float(label == "no")
    correct_no = float(predicted_answer == label == "no")

    overall = (
        accuracy_weight * accuracy
        + format_weight * strict_format
        + coordinate_weight * strict_format * coordinate
    )
    return {
        "overall": overall,
        "accuracy": accuracy,
        "format": strict_format,
        "coordinate": coordinate,
        "answer_valid": answer_valid,
        "predicted_no": predicted_no,
        "label_no": label_no,
        "correct_no": correct_no,
    }