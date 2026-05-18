#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path


def normalize_amber_prompt(prompt: str) -> str:
    return prompt.replace("Answer with the text of the option.", "Answer with the option letter only.")


def answer_to_option(answer: str) -> str:
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

    raise ValueError(f"Unsupported AMBER answer: {answer!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert AMBER MCTS JSONL to the RL trainer JSONL schema.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def convert_record(record):
    conversations = record["conversations"]
    return {
        "id": record.get("id"),
        "problem": normalize_amber_prompt(conversations[0]["value"]),
        "images": [record["image"]],
        "answer": answer_to_option(conversations[1]["value"]),
        "source": record.get("source", "AMBER"),
        "amber_type": record.get("amber_type", "unknown"),
    }


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    records = [convert_record(record) for record in read_records(args.input)]
    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_size = min(max(args.val_size, 1), len(records) - 1)
    val_records = records[:val_size]
    train_records = records[val_size:]

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)
    print(f"Wrote {len(train_records)} train records to {args.train_output}")
    print(f"Wrote {len(val_records)} val records to {args.val_output}")


if __name__ == "__main__":
    main()
