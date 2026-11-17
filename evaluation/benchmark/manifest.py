"""Maintain the two human-labeled query manifests."""

import argparse
import csv
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit

from AI.color_utils import PALETTE
from evaluation.run_m1_ablation import file_hash, load_labels

EVAL_DIR = Path(__file__).resolve().parents[1]
ROOT = EVAL_DIR.parent
FIELDS = ("query_id", "image", "lighting_group", "class_name", "color", "bbox",
          "source", "license", "author", "license_url")
SPLITS = ("validation", "test")
MIN_IMAGES = {"validation": 20, "test": 50}


def portable_path(path):
    """Store paths relative to the repo so saved runs can be scored on another machine."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"Unexpected CSV header: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def write_csv(path, fields, rows):
    path = Path(path)
    temp_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8",
                                newline="", delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_manifest(path):
    path = Path(path)
    for row in read_csv(path, FIELDS):
        qid = row["query_id"]
        image = Path(row["image"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", qid):
            raise ValueError(f"Invalid query_id: {qid!r}")
        if not row["image"] or image.is_absolute() or image.drive or ".." in image.parts:
            raise ValueError(f"Expected an image path relative to the manifest: {qid}")
        if row["color"] not in PALETTE:
            raise ValueError(f"Color outside PALETTE: {qid}")
    return load_labels(path)


def validate_splits(validation, test):
    for field, label in (("query_id", "query ID"), ("sha256", "image hash")):
        overlap = {row[field] for row in validation} & {row[field] for row in test}
        if overlap:
            raise ValueError(f"Duplicate {label} across validation and test: {', '.join(sorted(overlap))}")
    val_authors = {row["author"].strip().casefold() for row in validation}
    test_authors = {row["author"].strip().casefold() for row in test}
    shared_authors = val_authors & test_authors
    if shared_authors:
        warnings.warn(f"Authors shared by validation and test: {', '.join(sorted(shared_authors))}", stacklevel=2)


def source_key(url):
    parts = urlsplit(url)
    return (parts.netloc.casefold(), unquote(parts.path).replace("_", " ").casefold())


def reject_old_outfits(test):
    old_images = {file_hash(path) for path in (EVAL_DIR / "queries/real_outfits").iterdir()
                  if path.is_file() and path.name != ".gitkeep"}
    source_file = EVAL_DIR / "real_outfit_sources.json"
    old_sources = {source_key(row["source_url"]) for row in
                   json.loads(source_file.read_text(encoding="utf-8"))} if source_file.is_file() else set()
    for row in test:
        if (row["query_id"].startswith("real_") or row["sha256"] in old_images
                or source_key(row["source"]) in old_sources):
            raise ValueError(f"Old real_outfits image in test: {row['query_id']}")


def manifest_path(split):
    return EVAL_DIR / f"bench_{split}.csv"


def require_sample_size(split, labels):
    minimum = MIN_IMAGES[split]
    if len(labels) < minimum:
        raise ValueError(f"{split}: {len(labels)}/{minimum} labeled images")


def check():
    validation, test = [load_manifest(manifest_path(split)) for split in SPLITS]
    validate_splits(validation, test)
    reject_old_outfits(test)
    for split, rows in zip(SPLITS, (validation, test), strict=True):
        print(f"{split}: {len(rows)}/{MIN_IMAGES[split]} images")
        for field in ("lighting_group", "class_name", "color"):
            print(f"  {field}: {dict(sorted(Counter(row[field] for row in rows).items()))}")


def save_splits(rows_by_split):
    # Validate the entire batch at paths with the same parent as the real manifests.
    staged = {}
    try:
        for split, rows in rows_by_split.items():
            with NamedTemporaryFile(dir=EVAL_DIR, suffix=".csv", delete=False) as stream:
                staged[split] = Path(stream.name)
            write_csv(staged[split], FIELDS, rows)
        validation, test = [load_manifest(staged[split]) for split in SPLITS]
        validate_splits(validation, test)
        reject_old_outfits(test)
        for split in SPLITS:
            staged[split].replace(manifest_path(split))
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def init():
    if any(manifest_path(split).exists() for split in SPLITS):
        raise ValueError("Benchmark manifests already exist")
    old_val = read_csv(EVAL_DIR / "m1_validation.csv", FIELDS)
    old_test = read_csv(EVAL_DIR / "m1_test.csv", FIELDS)
    save_splits({
        "validation": old_val + [row for row in old_test if row["query_id"].startswith("real_")],
        "test": [row for row in old_test if row["query_id"].startswith("m1_")],
    })


def add(path):
    rows = {split: read_csv(manifest_path(split), FIELDS) for split in SPLITS}
    for row in read_csv(path, (*FIELDS, "split")):
        split = row.pop("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split!r}")
        rows[split].append(row)
    save_splits(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("add").add_argument("labeled_csv", type=Path)
    commands.add_parser("check")
    args = parser.parse_args()
    try:
        if args.command == "init":
            init()
        elif args.command == "add":
            add(args.labeled_csv)
        check()
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

