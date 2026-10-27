"""Conservatively remove repeated observations without merging distinct facts."""

import hashlib
import re

from vlmsearch.image_geometry import coordinate_matches


VERSION = "conservative_observation_dedup_v1"
FILLER = {"a", "an", "the", "is", "are", "was", "were", "there", "visible", "clearly", "image", "picture", "at", "around", "approximately"}
NUMBER_WORDS = {word: str(number) for number, word in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}


def _fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _signature(text):
    points = list(coordinate_matches(text))
    content, cursor = [], 0
    for match, _ in points:
        content.append(text[cursor:match.start()])
        content.append(" coordinateanchor ")
        cursor = match.end()
    content.append(text[cursor:])
    # Never delete negation, counts, colors, subjects, or other content words.
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:n't)?", " ".join(content).casefold().replace("n't", " not"))
    tokens = [NUMBER_WORDS.get(token, token) for token in tokens if token not in FILLER]
    return tuple(tokens), [point for _, point in points]


def _same_points(left, right, tolerance):
    if len(left) != len(right):
        return False
    for (x, y), (xx, yy) in zip(left, right):
        if (x - xx) ** 2 + (y - yy) ** 2 > tolerance ** 2:
            return False
    return True


def deduplicate_observations(text, coordinate_extent=1000):
    """Drop only equivalent ordered content tokens with nearby matching points.

    This is deliberately not a semantic truth check. All source coordinates
    must be validated before calling; callers must validate retained text too.
    """
    spans = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]
    tolerance = max(1.0, float(coordinate_extent) * 0.02)
    kept, signatures, removed = [], [], []
    for index, span in enumerate(spans):
        signature, points = _signature(span)
        canonical = re.sub(r"\s+", " ", span).strip().casefold()
        duplicate = None
        for previous_index, (old_signature, old_points, old_canonical) in enumerate(signatures):
            if canonical == old_canonical:
                duplicate = (previous_index, "exact_observation")
                break
            if signature and signature == old_signature and _same_points(points, old_points, tolerance):
                duplicate = (previous_index, "same_content_sequence_nearby_points")
                break
        if duplicate is not None:
            removed.append({"input_span": index, "text": span, "kept_span": duplicate[0], "reason": duplicate[1]})
        else:
            kept.append(span)
            signatures.append((signature, points, canonical))
    cleaned = "\n".join(kept)
    return cleaned, {"version": VERSION, "input_spans": len(spans), "kept_spans": len(kept),
                     "removed_spans": removed, "removed_count": len(removed),
                     "input_sha256": _fingerprint(text), "output_sha256": _fingerprint(cleaned),
                     "point_tolerance": tolerance}
