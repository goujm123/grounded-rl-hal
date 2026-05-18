import re


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
COORDINATE_PATTERN = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")


def _answer_to_option(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"^[\s\"'{}\[\]()]+|[\s\"'{}\[\]().!,;:]+$", "", normalized)

    choice_match = re.match(r"^([ab])\s*[\.)]?\s*(.*)$", normalized)
    if choice_match:
        choice, rest = choice_match.groups()
        rest = rest.strip()
        if not rest or rest in {"yes", "no"}:
            return choice.upper()

    if normalized == "yes":
        return "A"
    if normalized == "no":
        return "B"

    yes_no_match = re.search(r"\b(yes|no)\b", normalized)
    if yes_no_match:
        return "A" if yes_no_match.group(1) == "yes" else "B"

    return normalized.upper()


def _normalize_answer(answer: str) -> str:
    return _answer_to_option(answer)


def _has_option_letter_answer(answer: str) -> bool:
    return re.fullmatch(r"\s*[ABab]\s*", answer.strip()) is not None


def _extract_answer(predict_str: str) -> str:
    matches = ANSWER_PATTERN.findall(predict_str)
    if matches:
        return matches[-1]
    return predict_str


def _has_thinking_coordinates(predict_str: str) -> bool:
    think_blocks = THINK_PATTERN.findall(predict_str)
    if think_blocks:
        thinking_text = "\n".join(think_blocks)
    else:
        answer_match = ANSWER_PATTERN.search(predict_str)
        thinking_text = predict_str[: answer_match.start()] if answer_match else predict_str
    return COORDINATE_PATTERN.search(thinking_text) is not None


def amber_compute_score(predict_str: str, ground_truth: str):
    pred_answer = _normalize_answer(_extract_answer(predict_str))
    gt_answer = _normalize_answer(ground_truth)
    is_correct = pred_answer == gt_answer
    has_coordinate_evidence = _has_thinking_coordinates(predict_str)
    extracted_answer = _extract_answer(predict_str)
    has_option_answer = _has_option_letter_answer(extracted_answer)
    has_required_format = bool(THINK_PATTERN.search(predict_str) and ANSWER_PATTERN.search(predict_str) and has_option_answer)

    if not is_correct:
        overall = -1.0
    elif not has_required_format:
        overall = 0
    elif not has_coordinate_evidence:
        overall = 0.5
    else:
        overall = 1.0

    return {
        "overall": overall,
        "format": float(has_required_format),
        "accuracy": 1.0 if is_correct else 0.0,
        "coordinate_evidence": 1.0 if has_coordinate_evidence else 0.0,
    }
