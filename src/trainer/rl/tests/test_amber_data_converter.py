import json
import importlib.util
from pathlib import Path

import pytest


CONVERTER_PATH = Path(__file__).resolve().parents[4] / "scripts/rl/build_amber_discriminative_rl.py"
SPEC = importlib.util.spec_from_file_location("build_amber_discriminative_rl", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
build_rows = CONVERTER.build_rows
write_outputs = CONVERTER.write_outputs


def create_images(data_root: Path) -> None:
    image_dir = data_root / "mllm_hal/images"
    image_dir.mkdir(parents=True)
    (image_dir / "AMBER_1.jpg").write_bytes(b"image-one")
    (image_dir / "AMBER_2.jpg").write_bytes(b"image-two")


def test_converter_joins_by_id_and_groups_images_deterministically(tmp_path: Path) -> None:
    create_images(tmp_path)
    queries = [
        {"id": 1005, "image": "AMBER_1.jpg", "query": "Is the sky sunny?"},
        {"id": 1006, "image": "AMBER_1.jpg", "query": "Is the sky gloomy?"},
        {"id": 13557, "image": "AMBER_2.jpg", "query": "Are the person and grass touching?"},
    ]
    annotations = [
        {"id": 13557, "type": "relation", "truth": "no"},
        {"id": 1006, "type": "discriminative-attribute-state", "truth": "no"},
        {"id": 1005, "type": "discriminative-attribute-state", "truth": "yes"},
    ]

    first = build_rows(queries, annotations, tmp_path, val_fraction=0.5, seed=42)
    second = build_rows(queries, list(reversed(annotations)), tmp_path, val_fraction=0.5, seed=42)

    assert first == second
    train_rows, val_rows, report = first
    assert {row["image_key"] for row in train_rows}.isdisjoint(
        {row["image_key"] for row in val_rows}
    )
    assert report["train_validation_image_overlap"] == 0
    assert report["total"]["rows"] == 3
    assert {row["id"]: row["answer"] for row in train_rows + val_rows} == {
        1005: "yes",
        1006: "no",
        13557: "no",
    }
    assert all(row["prompt"].count("<image>") == 1 for row in train_rows + val_rows)


def test_output_files_are_byte_deterministic(tmp_path: Path) -> None:
    create_images(tmp_path)
    queries = [
        {"id": 1005, "image": "AMBER_1.jpg", "query": "Question one?"},
        {"id": 1006, "image": "AMBER_2.jpg", "query": "Question two?"},
    ]
    annotations = [
        {"id": 1005, "type": "discriminative-hallucination", "truth": "yes"},
        {"id": 1006, "type": "discriminative-relation", "truth": "no"},
    ]
    query_file = tmp_path / "queries.json"
    annotation_file = tmp_path / "annotations.json"
    query_file.write_text(json.dumps(queries), encoding="utf-8")
    annotation_file.write_text(json.dumps(annotations), encoding="utf-8")
    train_rows, val_rows, report = build_rows(queries, annotations, tmp_path, 0.5, 7)

    first_paths = write_outputs(
        tmp_path / "first", train_rows, val_rows, report, query_file, annotation_file
    )
    second_paths = write_outputs(
        tmp_path / "second", train_rows, val_rows, report, query_file, annotation_file
    )

    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()


def test_converter_deduplicates_matching_official_rows(tmp_path: Path) -> None:
    create_images(tmp_path)
    queries = [
        {"id": 1005, "image": "AMBER_1.jpg", "query": "Is there one lemon?"},
        {"id": 1006, "image": "AMBER_1.jpg", "query": "Is there one lemon?"},
    ]
    annotations = [
        {"id": 1005, "type": "discriminative-attribute-number", "truth": "yes"},
        {"id": 1006, "type": "discriminative-attribute-number", "truth": "yes"},
    ]

    train_rows, val_rows, report = build_rows(queries, annotations, tmp_path, 0.0, 42)

    assert [row["id"] for row in train_rows + val_rows] == [1005]
    assert report["source_query_rows"] == 2
    assert report["deduplicated_row_count"] == 1
    assert report["deduplicated_rows"][0]["dropped_id"] == 1006


def test_converter_rejects_conflicting_duplicate_rows(tmp_path: Path) -> None:
    create_images(tmp_path)
    queries = [
        {"id": 1005, "image": "AMBER_1.jpg", "query": "Is there one lemon?"},
        {"id": 1006, "image": "AMBER_1.jpg", "query": "Is there one lemon?"},
    ]
    annotations = [
        {"id": 1005, "type": "discriminative-attribute-number", "truth": "yes"},
        {"id": 1006, "type": "discriminative-attribute-number", "truth": "no"},
    ]

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        build_rows(queries, annotations, tmp_path, 0.0, 42)


@pytest.mark.parametrize(
    ("annotations", "message"),
    [
        ([], "No annotation found"),
        ([{"id": 1005, "type": "discriminative-hallucination", "truth": "maybe"}], "non-binary"),
        ([{"id": 1005, "type": "unsupported", "truth": "yes"}], "unknown discriminative type"),
    ],
)
def test_converter_rejects_invalid_annotations(tmp_path: Path, annotations, message: str) -> None:
    create_images(tmp_path)
    queries = [{"id": 1005, "image": "AMBER_1.jpg", "query": "Is there a cloud?"}]

    with pytest.raises(ValueError, match=message):
        build_rows(queries, annotations, tmp_path, val_fraction=0.0, seed=42)