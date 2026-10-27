#!/usr/bin/env python3
"""Fetch pinned AMBER, COCO POPE, and seven official LongHalQA releases.

Dependencies: requests, gdown, huggingface_hub, pyarrow, Pillow. Files are
resumable/reused, original annotations are retained, and output images are fully
decoded before the final manifest is published. No dataset loading code runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
import zipfile

import requests
from PIL import Image


AMBER_REPO = "junyangwang0410/AMBER"
AMBER_REVISION = "534babf6bbfcce2e735c26289dedfb21cef3c939"
AMBER_DRIVE_ID = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
POPE_REPO = "AoiDragon/POPE"
POPE_REVISION = "08d957b917e5a378a2f99d35b6293c536a66298b"
LONGHALQA = {
    "object_hall_binary_v1": "7f30df615e5856018b92892db208e9466bc21b11",
    "description_hall_binary_v3": "7466a86d94980760af58dd49ad48f98b19bbb831",
    "conversation_hall_binary_v1": "407b138996effdecf2607793db178a547787dbe8",
    "description_hall_choice_v3": "73a427bd2dde197a7ec54dacdb95b43fc4fae392",
    "conversation_hall_choice_v2": "2cac5c2720a4336b8d2d74a45fa4b756cf3c96da",
    "description_hall_completion_v4": "224f9be65517fef129f3dfee4637416b9b91d24b",
    "conversation_hall_completion_v3": "ca8260b36cd6520c81556f864ed077c3a427e1e6",
}


def digest(path: Path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_json(url):
    response = requests.get(url, timeout=(15, 120))
    response.raise_for_status()
    return response.json()


def fetch(url: str, destination: Path, expected_blob=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        if expected_blob is None or digest(destination, "sha1", True) == expected_blob:
            return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            if expected_blob and digest(temporary, "sha1", True) != expected_blob:
                raise ValueError(f"Git blob hash mismatch: {url}")
            temporary.replace(destination)
            return destination
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def github_files(repo, revision, prefix, destination):
    tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")
    provenance = []
    for entry in tree["tree"]:
        if entry["type"] != "blob" or not entry["path"].startswith(prefix):
            continue
        relative = entry["path"][len(prefix):]
        target = destination / relative
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{entry['path']}"
        fetch(url, target, expected_blob=entry["sha"])
        provenance.append({"path": str(target), "url": url, "git_blob_sha1": entry["sha"]})
    return provenance


def binary_answer(value):
    answer = str(value).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"Invalid binary answer: {value!r}")
    return answer


def download_amber(data_root):
    target = data_root / "hallucination/raw/amber"
    provenance = github_files(AMBER_REPO, AMBER_REVISION, "data/", target)
    archive = target / "images.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        import gdown

        result = gdown.download(id=AMBER_DRIVE_ID, output=str(archive), resume=True)
        if result is None:
            raise RuntimeError("Official AMBER image archive download failed")
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member:
            raise ValueError(f"AMBER archive CRC mismatch: {bad_member}")
        members = {Path(member.filename).name: member for member in handle.infolist()
                   if re.fullmatch(r"AMBER_\d+\.jpg", Path(member.filename).name)}
        expected = {f"AMBER_{index}.jpg" for index in range(1, 1005)}
        if set(members) != expected:
            raise ValueError(f"Unexpected AMBER archive image inventory: {len(members)}")
        images_dir = target / "images"
        images_dir.mkdir(exist_ok=True)
        for name, member in members.items():
            path = images_dir / name
            if not path.exists() or path.stat().st_size != member.file_size:
                path.write_bytes(handle.read(member))
    annotation_rows = json.loads((target / "annotations.json").read_text())
    annotations = {str(row["id"]): row for row in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise ValueError("Duplicate AMBER annotation IDs")
    queries = json.loads((target / "query/query_discriminative.json").read_text())
    rows = []
    for query in queries:
        source_id = str(query["id"])
        annotation = annotations[source_id]
        name = Path(query["image"]).name
        rows.append({
            "id": f"amber:{source_id}", "dataset": "amber", "task": "amber_discriminative",
            "image": (images_dir / name).relative_to(data_root).as_posix(),
            "question": query["query"], "answer": binary_answer(annotation["truth"]),
            "answer_type": "binary", "category": annotation["type"],
            "source_id": source_id, "source_image_id": f"amber:{name}",
            "source_revision": AMBER_REVISION,
        })
    if len(rows) != 14216:
        raise ValueError(f"Expected 14216 AMBER discriminative rows, found {len(rows)}")
    provenance.append({"path": str(archive), "url": f"https://drive.google.com/file/d/{AMBER_DRIVE_ID}/view",
                       "sha256": digest(archive),
                       "note": "Official author Drive file has no immutable revision or published checksum; local SHA256 recorded."})
    return rows, provenance


def download_pope(data_root, workers):
    target = data_root / "hallucination/raw/pope"
    provenance = github_files(POPE_REPO, POPE_REVISION, "output/coco/", target / "annotations")
    images_dir = target / "images"
    rows = []
    for variant in ("random", "popular", "adversarial"):
        annotation = target / f"annotations/coco_pope_{variant}.json"
        source_rows = [json.loads(line) for line in annotation.read_text().splitlines() if line.strip()]
        if len(source_rows) != 3000:
            raise ValueError(f"Unexpected POPE {variant} count: {len(source_rows)}")
        for source in source_rows:
            name = Path(source["image"]).name
            image_id = re.fullmatch(r"COCO_val2014_(\d+)\.jpg", name)
            if not image_id:
                raise ValueError(f"Unexpected COCO image name: {name}")
            rows.append({
                "id": f"pope:{variant}:{source['question_id']}", "dataset": "pope",
                "task": f"pope_{variant}", "image": (images_dir / name).relative_to(data_root).as_posix(),
                "question": source["text"], "answer": binary_answer(source["label"]),
                "answer_type": "binary", "category": "existence",
                "source_id": str(source["question_id"]),
                "source_image_id": f"coco2014:{image_id.group(1)}", "source_revision": POPE_REVISION,
            })
    names = sorted({Path(row["image"]).name for row in rows})
    if len(names) != 500:
        raise ValueError(f"Expected 500 POPE images, found {len(names)}")
    def fetch_image(name):
        # Use the COCO public S3 bucket's path-style TLS endpoint. The image
        # host itself is commonly linked over HTTP and has a mismatched cert.
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2014/{name}"
        path = fetch(url, images_dir / name)
        return {"path": str(path), "url": url,
                "official_image_url": f"http://images.cocodataset.org/val2014/{name}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        provenance.extend(pool.map(fetch_image, names))
    return rows, provenance


def source_image_key(name):
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    if "objects365" in name.lower():
        return f"objects365:{name}"
    numeric = re.fullmatch(r"(\d+)\.(?:jpg|jpeg|png)", name, re.I)
    if numeric:
        return f"visualgenome:{numeric.group(1)}"
    return f"longhalqa:{name}"


def download_longhalqa(data_root):
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    target = data_root / "hallucination/raw/longhalqa"
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    provenance, rows = [], []
    api = HfApi()
    for task, revision in LONGHALQA.items():
        repo = f"QHQK/{task}"
        print(f"Downloading {repo}@{revision}", flush=True)
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError(f"HF revision mismatch: {repo}")
        local = target / task
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=local, max_workers=4)
        for entry in info.siblings:
            path = local / entry.rfilename
            if path.stat().st_size != entry.size:
                raise ValueError(f"HF file size mismatch: {path}")
            expected_hash = entry.lfs.sha256 if entry.lfs else entry.blob_id
            actual_hash = digest(path) if entry.lfs else digest(path, "sha1", True)
            if actual_hash != expected_hash:
                raise ValueError(f"HF upstream hash mismatch: {path}")
            provenance.append({"path": str(path), "repo_id": repo, "revision": revision,
                               "upstream_hash": expected_hash,
                               "upstream_algorithm": "sha256" if entry.lfs else "git_blob_sha1"})
        for parquet_path in sorted(local.glob("data/*.parquet")):
            # Read in batches to bound memory independently of release size.
            for batch in parquet.ParquetFile(parquet_path).iter_batches(batch_size=64):
                for source in batch.to_pylist():
                    image_blob = source["image"]["bytes"]
                    if not image_blob:
                        raise ValueError(f"Missing embedded image in {repo}")
                    image_hash = hashlib.sha256(image_blob).hexdigest()
                    with Image.open(BytesIO(image_blob)) as image:
                        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
                        if not suffix:
                            raise ValueError(f"Unsupported embedded image format: {image.format}")
                    image_path = images_dir / (image_hash + suffix)
                    if not image_path.exists():
                        image_path.write_bytes(image_blob)
                    answer_type = "binary" if "binary" in task else "choice"
                    answer = binary_answer(source["answer"]) if answer_type == "binary" else str(source["answer"]).strip().upper()
                    if answer_type == "choice" and answer not in {"A", "B", "C", "D"}:
                        raise ValueError(f"Invalid LongHalQA choice: {answer}")
                    question = source["question"].strip()
                    if answer_type == "choice":
                        question += "\n" + "\n".join(f"{letter}. {source['choice_' + letter.lower()]}" for letter in "ABCD")
                    rows.append({
                        "id": f"longhalqa:{task}:{source['question_id']}", "dataset": "longhalqa",
                        "task": task, "image": image_path.relative_to(data_root).as_posix(),
                        "question": question, "answer": answer, "answer_type": answer_type,
                        "category": source.get("category", ""), "source_id": str(source["question_id"]),
                        "source_image_id": source_image_key(source["image_source"]),
                        "source_revision": revision,
                    })
    if len(rows) != 9461:
        raise ValueError(f"Expected 9461 LongHalQA task rows, found {len(rows)}")
    return rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")))
    parser.add_argument("--datasets", default="amber,pope,longhalqa")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    base = data_root / "hallucination"
    all_files, summary = [], {}
    for dataset in args.datasets.split(","):
        print(f"Preparing official {dataset} data", flush=True)
        if dataset == "amber":
            rows, provenance = download_amber(data_root)
        elif dataset == "pope":
            rows, provenance = download_pope(data_root, args.workers)
        elif dataset == "longhalqa":
            rows, provenance = download_longhalqa(data_root)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Non-unique namespaced IDs: {dataset}")
        image_paths = sorted({row["image"] for row in rows})
        image_records = []
        for relative in image_paths:
            path = data_root / relative
            with Image.open(path) as image:
                image.load()
                dimensions = list(image.size)
            image_records.append({"path": relative, "bytes": path.stat().st_size,
                                  "sha256": digest(path), "dimensions": dimensions, "decoded": True})
        source_records = []
        for entry in provenance:
            path = Path(entry.pop("path"))
            source_records.append({"path": path.relative_to(data_root).as_posix(),
                                   "bytes": path.stat().st_size, "sha256": digest(path), **entry})
        normalized = base / f"normalized/{dataset}.jsonl"
        write_rows(normalized, rows)
        result = {"dataset": dataset, "records": len(rows), "unique_image_files": len(image_paths),
                  "normalized": normalized.relative_to(data_root).as_posix(),
                  "normalized_sha256": digest(normalized), "sources": source_records,
                  "images": image_records}
        atomic_json(base / f"raw/{dataset}/download-manifest.json", result)
        summary[dataset] = {key: result[key] for key in ("records", "unique_image_files", "normalized", "normalized_sha256")}
        all_files.append(result)
        print(f"Verified {dataset}: {len(rows)} records, {len(image_paths)} image files", flush=True)
    # Preserve verified manifests for datasets not selected in a resumed run.
    existing = {result["dataset"]: result for result in all_files}
    for dataset in ("amber", "pope", "longhalqa"):
        path = base / f"raw/{dataset}/download-manifest.json"
        if dataset not in existing and path.exists():
            existing[dataset] = json.loads(path.read_text())
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_root": str(data_root), "all_images_decoded": True,
                "datasets": existing}
    atomic_json(base / "raw/download-manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
