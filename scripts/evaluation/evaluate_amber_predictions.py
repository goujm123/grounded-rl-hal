#!/usr/bin/env python3
"""Evaluate AMBER discriminative predictions with AMBER and repo-style metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score AMBER discriminative predictions on normalized A/B answers."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL to score.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional frozen AMBER split JSONL for ids, gold answers, and amber_type metadata.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional metrics JSON output path.",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=None,
        help="Optional normalized per-sample JSONL output path.",
    )
    parser.add_argument(
        "--amber-original-output",
        type=Path,
        default=None,
        help="Optional original-AMBER JSON output with id/response rows for compatible predictions.",
    )
    parser.add_argument(
        "--allow-order-reference-alignment",
        action="store_true",
        help="Allow rows without ids to inherit reference metadata by row order. Use only for known same-order files.",
    )
    parser.add_argument(
        "--no-grounded-score",
        action="store_true",
        help="Skip repo-style amber_compute_score aggregation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a row cannot be aligned to a gold answer.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_amber_reward_module() -> ModuleType:
    module_path = repo_root() / "src/trainer/rl/examples/reward_function/amber_hallucination.py"
    spec = importlib.util.spec_from_file_location("amber_hallucination", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AMBER reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_answer_text(text: Any) -> str:
    value = "" if text is None else str(text)
    matches = ANSWER_PATTERN.findall(value)
    if matches:
        return matches[-1]
    outside_think = THINK_PATTERN.sub(" ", value).strip()
    trailing_option = re.search(r"(?:^|\s)([ABab])\s*$", outside_think)
    if trailing_option:
        return trailing_option.group(1)
    return value


def local_answer_to_option(answer: Any) -> str:
    normalized = str(answer).strip().lower()
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


def normalize_option(answer: Any, reward_module: ModuleType) -> str:
    extracted = extract_answer_text(answer)
    if hasattr(reward_module, "_normalize_answer"):
        option = reward_module._normalize_answer(extracted)
    else:
        option = local_answer_to_option(extracted)
    return option if option in {"A", "B"} else ""


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def reference_metadata(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations") or []
    images = record.get("images") or []
    image = first_present(record, ("image",))
    if image is None and isinstance(images, list) and images:
        image = images[0]

    question = first_present(record, ("problem", "question", "prompt"))
    gold_answer = first_present(record, ("answer", "gold_answer", "true_answer", "label"))
    if isinstance(conversations, list) and conversations:
        if question is None and len(conversations) >= 1:
            question = conversations[0].get("value")
        if gold_answer is None and len(conversations) >= 2:
            gold_answer = conversations[1].get("value")

    return {
        "id": str(record["id"]) if "id" in record else None,
        "question": question,
        "image": image,
        "gold_answer_text": extract_answer_text(gold_answer),
        "amber_type": record.get("amber_type", "unknown"),
    }


def build_reference_index(reference_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    metadata = [reference_metadata(record) for record in reference_records]
    by_id = {str(item["id"]): item for item in metadata if item.get("id") is not None}
    return metadata, by_id


def prediction_text(record: dict[str, Any]) -> str:
    direct = first_present(
        record,
        ("prediction_text", "predict", "response", "output", "generated_text", "text"),
    )
    if direct is not None:
        return str(direct)

    thoughts = record.get("thoughts")
    final_answer = record.get("final_answer")
    if isinstance(thoughts, list) or final_answer is not None:
        thought_text = "\n".join(str(thought) for thought in thoughts or [] if thought is not None)
        if ANSWER_PATTERN.search(str(final_answer)):
            answer_text = str(final_answer)
        elif final_answer is not None:
            answer_text = f"<answer> {final_answer} </answer>"
        else:
            answer_text = ""
        if thought_text.strip():
            return f"<think>\n{thought_text}\n</think>\n{answer_text}".strip()
        return answer_text

    return ""


def prediction_answer_text(record: dict[str, Any], text: str) -> str:
    direct = first_present(record, ("pred_answer", "final_answer", "prediction_answer"))
    if direct is not None:
        return extract_answer_text(direct)
    return extract_answer_text(text)


def gold_answer_text(record: dict[str, Any], reference: dict[str, Any] | None) -> str:
    direct = first_present(record, ("gold_answer", "label", "true_answer"))
    if direct is not None:
        return extract_answer_text(direct)
    if reference is not None:
        return str(reference.get("gold_answer_text", ""))
    return ""


def align_reference(
    record: dict[str, Any],
    index: int,
    reference_records: list[dict[str, Any]],
    reference_by_id: dict[str, dict[str, Any]],
    allow_order_alignment: bool,
) -> tuple[dict[str, Any] | None, str]:
    record_id = record.get("id")
    if record_id is not None and str(record_id) in reference_by_id:
        return reference_by_id[str(record_id)], "id"
    if allow_order_alignment and index < len(reference_records):
        return reference_records[index], "order"
    return None, "none"


def normalize_rows(
    prediction_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    reference_by_id: dict[str, dict[str, Any]],
    reward_module: ModuleType,
    strict: bool,
    include_grounded_score: bool,
    allow_order_alignment: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, record in enumerate(prediction_records):
        reference, alignment = align_reference(
            record, index, reference_records, reference_by_id, allow_order_alignment
        )
        text = prediction_text(record)
        pred_answer_text = prediction_answer_text(record, text)
        gold_text = gold_answer_text(record, reference)
        pred_answer = normalize_option(pred_answer_text, reward_module)
        gold_answer = normalize_option(gold_text, reward_module)

        if not gold_answer:
            message = f"Row {index} has no valid gold A/B answer"
            if strict:
                raise ValueError(message)
            warnings.append(message)

        row_id = str(record.get("id")) if record.get("id") is not None else None
        if row_id is None and reference is not None and reference.get("id") is not None:
            row_id = str(reference["id"])
        if row_id is None:
            row_id = str(index)

        normalized = {
            "id": row_id,
            "alignment": alignment,
            "amber_type": record.get("amber_type") or (reference or {}).get("amber_type", "unknown"),
            "question": record.get("question") or record.get("prompt") or (reference or {}).get("question"),
            "image": record.get("image") or (reference or {}).get("image"),
            "prediction_text": text,
            "prediction_answer_text": pred_answer_text,
            "pred_answer": pred_answer,
            "gold_answer_text": gold_text,
            "gold_answer": gold_answer,
            "correct": bool(pred_answer and gold_answer and pred_answer == gold_answer),
        }

        if include_grounded_score:
            normalized["grounded_score"] = reward_module.amber_compute_score(text, gold_answer or gold_text)

        normalized_rows.append(normalized)

    return normalized_rows, warnings


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_answer_option_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row["gold_answer"] in {"A", "B"}]
    correct_count = sum(1 for row in evaluated if row["correct"])
    confusion: Counter[str] = Counter()
    invalid_pred_count = 0

    for row in evaluated:
        pred = row["pred_answer"] if row["pred_answer"] in {"A", "B"} else "invalid"
        if pred == "invalid":
            invalid_pred_count += 1
        confusion[f"gold_{row['gold_answer']}_pred_{pred}"] += 1

    tp = confusion["gold_A_pred_A"]
    fp = confusion["gold_B_pred_A"]
    tn = confusion["gold_B_pred_B"]
    fn = confusion["gold_A_pred_B"] + confusion["gold_A_pred_invalid"]

    return {
        "count": len(rows),
        "evaluated_count": len(evaluated),
        "correct_count": correct_count,
        "invalid_prediction_count": invalid_pred_count,
        "accuracy": safe_divide(correct_count, len(evaluated)),
        "precision_yes": safe_divide(tp, tp + fp),
        "recall_yes": safe_divide(tp, tp + fn),
        "f1_yes": safe_divide(2 * tp, 2 * tp + fp + fn),
        "unsupported_yes_rate": safe_divide(fp, fp + tn),
        "confusion": dict(sorted(confusion.items())),
    }


def aggregate_original_group(rows: list[dict[str, Any]], f1_epsilon: float = 0.0001) -> dict[str, Any]:
    evaluated = [row for row in rows if row["gold_answer"] in {"A", "B"}]
    correct_count = sum(1 for row in evaluated if row["pred_answer"] == row["gold_answer"])
    gold_no_count = sum(1 for row in evaluated if row["gold_answer"] == "B")
    pred_no_count = sum(1 for row in evaluated if row["pred_answer"] == "B")
    correct_no_count = sum(1 for row in evaluated if row["gold_answer"] == "B" and row["pred_answer"] == "B")

    accuracy = round(safe_divide(correct_count, len(evaluated)) * 100, 1)
    precision = round(safe_divide(correct_no_count, pred_no_count) * 100, 1)
    recall = round(safe_divide(correct_no_count, gold_no_count) * 100, 1)
    f1 = round(
        safe_divide(
            2 * (precision / 100) * (recall / 100),
            (precision / 100) + (recall / 100) + f1_epsilon,
        )
        * 100,
        1,
    )

    return {
        "count": len(rows),
        "evaluated_count": len(evaluated),
        "correct_count": correct_count,
        "gold_no_count": gold_no_count,
        "pred_no_count": pred_no_count,
        "correct_no_count": correct_no_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_original_amber_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attribute_types = {
        "discriminative-attribute-state",
        "discriminative-attribute-number",
        "discriminative-attribute-action",
    }
    known_type_rows = [row for row in rows if str(row.get("amber_type") or "unknown") != "unknown"]
    groups = {
        "discriminative_task": rows,
        "existence": [row for row in known_type_rows if row.get("amber_type") == "discriminative-hallucination"],
        "attribute": [row for row in known_type_rows if row.get("amber_type") in attribute_types],
        "state": [row for row in known_type_rows if row.get("amber_type") == "discriminative-attribute-state"],
        "number": [row for row in known_type_rows if row.get("amber_type") == "discriminative-attribute-number"],
        "action": [row for row in known_type_rows if row.get("amber_type") == "discriminative-attribute-action"],
        "relation": [
            row
            for row in known_type_rows
            if row.get("amber_type") not in attribute_types and row.get("amber_type") != "discriminative-hallucination"
        ],
    }
    return {
        key: aggregate_original_group(value, f1_epsilon=0.001 if key == "existence" else 0.0001)
        for key, value in groups.items()
        if value
    }


def aggregate_by_type(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("amber_type") or "unknown")].append(row)
    return {key: aggregate_answer_option_metrics(value) for key, value in sorted(grouped.items())}


def aggregate_grounded_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    scored = [row["grounded_score"] for row in rows if "grounded_score" in row]
    if not scored:
        return {}
    keys = sorted({key for score in scored for key in score})
    return {key: safe_divide(sum(float(score.get(key, 0.0)) for score in scored), len(scored)) for key in keys}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_amber_original_json(path: Path, rows: list[dict[str, Any]]) -> int:
    output_rows = []
    for row in rows:
        pred_answer = row.get("pred_answer")
        row_id = str(row.get("id", ""))
        if not row_id.isdigit() or pred_answer not in {"A", "B"}:
            continue
        output_rows.append({"id": int(row_id), "response": "Yes" if pred_answer == "A" else "No"})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return len(output_rows)


def print_summary(metrics: dict[str, Any]) -> None:
    summary = metrics["answer_option"]
    amber_summary = metrics["original_amber_discriminative"]["discriminative_task"]
    print(f"Predictions: {metrics['prediction_file']}")
    if metrics.get("reference_file"):
        print(f"Reference: {metrics['reference_file']}")
    print(f"Rows: {summary['count']} total, {summary['evaluated_count']} evaluated")
    print(
        "Original AMBER discriminative: "
        f"accuracy={amber_summary['accuracy']:.1f}, "
        f"precision={amber_summary['precision']:.1f}, "
        f"recall={amber_summary['recall']:.1f}, "
        f"F1={amber_summary['f1']:.1f}"
    )
    print(f"Answer accuracy: {summary['accuracy']:.4f} ({summary['correct_count']}/{summary['evaluated_count']})")
    print(f"Precision yes: {summary['precision_yes']:.4f}")
    print(f"Recall yes: {summary['recall_yes']:.4f}")
    print(f"F1 yes: {summary['f1_yes']:.4f}")
    print(f"Unsupported yes rate: {summary['unsupported_yes_rate']:.4f}")
    print("Alignment counts:", metrics["alignment_counts"])
    if metrics.get("grounded_score"):
        grounded = metrics["grounded_score"]
        print("Repo-style grounded score:")
        for key in sorted(grounded):
            print(f"  {key}: {grounded[key]:.4f}")
    if metrics.get("warnings"):
        print(f"Warnings: {len(metrics['warnings'])}")


def main() -> None:
    args = parse_args()
    reward_module = load_amber_reward_module()
    prediction_records = read_jsonl(args.predictions)
    raw_reference_records = read_jsonl(args.reference) if args.reference else []
    reference_records, reference_by_id = build_reference_index(raw_reference_records)

    rows, warnings = normalize_rows(
        prediction_records=prediction_records,
        reference_records=reference_records,
        reference_by_id=reference_by_id,
        reward_module=reward_module,
        strict=args.strict,
        include_grounded_score=not args.no_grounded_score,
        allow_order_alignment=args.allow_order_reference_alignment,
    )

    metrics = {
        "prediction_file": str(args.predictions),
        "reference_file": str(args.reference) if args.reference else None,
        "original_amber_discriminative": aggregate_original_amber_metrics(rows),
        "answer_option": aggregate_answer_option_metrics(rows),
        "by_amber_type": aggregate_by_type(rows),
        "grounded_score": aggregate_grounded_scores(rows),
        "alignment_counts": dict(sorted(Counter(row["alignment"] for row in rows).items())),
        "warnings": warnings[:50],
    }

    print_summary(metrics)

    if args.output_json is not None:
        write_json(args.output_json, metrics)
        print(f"Wrote metrics to {args.output_json}")
    if args.normalized_output is not None:
        write_jsonl(args.normalized_output, rows)
        print(f"Wrote normalized predictions to {args.normalized_output}")
    if args.amber_original_output is not None:
        output_count = write_amber_original_json(args.amber_original_output, rows)
        print(f"Wrote {output_count} original-AMBER rows to {args.amber_original_output}")


if __name__ == "__main__":
    main()