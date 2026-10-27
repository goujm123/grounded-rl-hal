"""Strict final-answer parsing shared by search and its deterministic scorer."""

import re
from typing import Iterable, Optional


def _normalize_payload(value: str) -> str:
    # Formatting around an atomic label is harmless; words inside it are not.
    value = value.strip()
    pairs = {'(': ')', '[': ']', '{': '}', '"': '"', "'": "'"}
    while len(value) > 1 and value[0] in pairs and value[-1] == pairs[value[0]]:
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value.rstrip(".!?,;:").strip()).casefold()


def parse_final_answer(
    text: str,
    allowed_answers: Optional[Iterable[str]] = None,
    final_token_begin: str = "<answer>",
    final_token_end: str = "</answer>",
) -> Optional[str]:
    """Read one unambiguous final answer, without searching prose for labels.

    A tagged answer can be preceded by balanced ``<think>`` or ``<evidence>``
    blocks, but cannot have a second answer, malformed/nested tags, or trailing
    text. Plain labels remain supported for already-parsed terminal nodes.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    value = text.strip()
    if final_token_begin in value or final_token_end in value:
        if value.count(final_token_begin) != 1 or value.count(final_token_end) != 1:
            return None
        if value.find(final_token_end) < value.find(final_token_begin):
            return None
        before, remainder = value.split(final_token_begin, 1)
        payload, suffix = remainder.split(final_token_end, 1)
        if suffix.strip() or "<" in payload or ">" in payload:
            return None
        # Permit visible evidence statements only in their declared blocks.
        prefix = re.sub(
            r"<(?P<tag>think|evidence)>(?:(?!</?(?:think|evidence|answer)>).)*</(?P=tag)>",
            "", before, flags=re.DOTALL,
        )
        if prefix.strip():
            return None
        value = payload
    elif "<" in value or ">" in value:
        return None
    normalized = _normalize_payload(value)
    if not normalized:
        return None
    if allowed_answers is not None:
        allowed = {_normalize_payload(answer) for answer in allowed_answers}
        if normalized not in allowed:
            return None
    return normalized


def strict_match_answer(
    ground_truth: str,
    prediction: str,
    final_token_begin: str = "<answer>",
    final_token_end: str = "</answer>",
) -> float:
    """Compare complete answers; binary and choice labels must be atomic."""
    expected = parse_final_answer(ground_truth, final_token_begin=final_token_begin,
                                  final_token_end=final_token_end)
    if expected is None:
        return 0.0
    if expected in {"yes", "no"}:
        allowed = {"yes", "no"}
    elif expected in {"a", "b", "c", "d"}:
        allowed = {"a", "b", "c", "d"}
    else:
        allowed = None
    actual = parse_final_answer(prediction, allowed, final_token_begin, final_token_end)
    return float(actual is not None and actual == expected)
