"""Pool blinded product cards and merge human judgments."""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from itertools import islice
from math import ceil, floor
from pathlib import Path
from urllib.parse import quote, urlsplit

from PIL import Image, ImageDraw

from object_detection.image_preprocessing import prepare_image

from .catalog import Catalog
from .manifest import EVAL_DIR, read_csv, resolve_path, write_csv
from .run import read_json

QREL_FIELDS = ("query_id", "product_id", "type_ok", "color_ok", "acceptable", "grader", "graded_at")
STATUS_FIELDS = ("query_id", "relevance_status", "checked_by", "note")
FLAGS = ("type_ok", "color_ok", "acceptable")
STATUSES = ("has_match", "no_match", "unconfirmed")
R2_VARS = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")
CACHE = EVAL_DIR / "benchmark/cache"
TEMPLATE = Path(__file__).with_name("grading_template.html")


def field_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}")
    return value.strip()


def validated_pair(row):
    qid, pid = (field_text(row.get(key), key) for key in QREL_FIELDS[:2])
    flags = {}
    for key in FLAGS:
        value = row.get(key)
        if not (type(value) in (int, str) and value in (0, 1, "0", "1")):
            raise ValueError(f"{qid}/{pid}: {key} must be 0 or 1")
        flags[key] = int(value)
    if flags["acceptable"] and not flags["type_ok"]:
        raise ValueError(f"{qid}/{pid}: acceptable requires type_ok")
    return {"query_id": qid, "product_id": pid, **flags}


def read_judgment_rows(root):
    paths = (root / "bench_qrels.csv", root / "bench_query_status.csv")
    rows, statuses = [read_csv(path, fields) if path.exists() else []
                      for path, fields in zip(paths, (QREL_FIELDS, STATUS_FIELDS), strict=True)]
    pairs = {}
    for row in rows:
        pair = validated_pair(row)
        key = pair["query_id"], pair["product_id"]
        if key in pairs:
            raise ValueError(f"Duplicate judgment: {key}")
        pairs[key] = row | pair
    by_query = {}
    for row in statuses:
        qid = field_text(row.get("query_id"), "query_id")
        if qid in by_query or row["relevance_status"] not in STATUSES:
            raise ValueError(f"Invalid or duplicate query status: {qid}")
        by_query[qid] = row
    return pairs, by_query


def load_judgments(root=EVAL_DIR):
    pairs, statuses = read_judgment_rows(Path(root))
    query_status = {qid: "no_match" if row["relevance_status"] == "no_match" else "unconfirmed"
                    for qid, row in statuses.items()}
    query_status.update({qid: "has_match" for (qid, _), pair in pairs.items() if pair["acceptable"]})
    return pairs, query_status


def merge_grades(document, root=EVAL_DIR):
    root = Path(root)
    grader = field_text(document.get("grader"), "grader")
    timestamp = field_text(document.get("exported_at"), "exported_at")
    if datetime.fromisoformat(timestamp).tzinfo is None:
        raise ValueError("exported_at must include a timezone")
    pairs, statuses = read_judgment_rows(root)
    seen = set()
    new = overwritten = 0
    for raw in document["pairs"]:
        row = validated_pair(raw)
        key = row["query_id"], row["product_id"]
        if key in seen:
            raise ValueError(f"Duplicate pair in export: {key}")
        seen.add(key)
        overwritten += key in pairs
        new += key not in pairs
        pairs[key] = row | {"grader": grader, "graded_at": timestamp}

    query_metadata = {}
    for row in document["queries"]:
        qid = field_text(row.get("query_id"), "query_id")
        if qid in query_metadata or type(row.get("no_match")) is not bool:
            raise ValueError(f"Invalid or duplicate exported query: {qid}")
        if row["no_match"] and row.get("expanded_reviewed") is not True:
            raise ValueError(f"{qid}: no_match requires reviewing the extended list")
        query_metadata[qid] = row
    if not {qid for qid, _ in seen} <= query_metadata.keys():
        raise ValueError("Export must include query metadata for graded pairs")

    relevant = {qid for (qid, _), pair in pairs.items() if pair["acceptable"]}
    for qid, row in query_metadata.items():
        if row["no_match"] and qid in relevant:
            raise ValueError(f"{qid}: no_match conflicts with an acceptable product")
        statuses[qid] = {
            "query_id": qid, "relevance_status": "no_match" if row["no_match"] else "unconfirmed",
            "checked_by": grader, "note": "Đã xem danh sách mở rộng" if row["no_match"] else "",
        }
    for qid in statuses.keys() | relevant:
        if qid in relevant:
            statuses[qid] = {"query_id": qid, "relevance_status": "has_match", "checked_by": grader, "note": ""} if (
                qid in query_metadata or qid not in statuses
            ) else statuses[qid] | {"relevance_status": "has_match"}
        elif statuses[qid]["relevance_status"] == "has_match":
            statuses[qid] = statuses[qid] | {"relevance_status": "unconfirmed"}

    write_csv(root / "bench_qrels.csv", QREL_FIELDS, [pairs[key] for key in sorted(pairs)])
    write_csv(root / "bench_query_status.csv", STATUS_FIELDS, [statuses[qid] for qid in sorted(statuses)])
    counts = Counter(row["relevance_status"] for row in statuses.values())
    return new, overwritten, {key: counts[key] for key in STATUSES}


def target_ranked_ids(query):
    ids = {}
    for result in query["results"].values():
        index = result["target"]["index"]
        if index is not None:
            ids.update(dict.fromkeys(str(product["id"]) for product in result["detections"][index]["ranked"][:10]))
    return list(ids)


def relative_url(path, folder):
    return quote(Path(os.path.relpath(path, folder)).as_posix(), safe="/")


def query_roi(query, path, thumb_path):
    pixels = path.read_bytes()
    if hashlib.sha256(pixels).hexdigest() != query["sha256"]:
        raise ValueError(f"Query image changed after inference: {query['query_id']}")
    image = Image.fromarray(prepare_image(pixels).source_rgb)
    x1, y1, x2, y2 = (val * length for val, length in zip(query["bbox"], image.size * 2, strict=True))
    crop = image.crop((floor(x1), floor(y1), ceil(x2), ceil(y2)))
    ImageDraw.Draw(image).rectangle((x1, y1, x2, y2), outline="#15df61", width=max(2, image.width // 160))
    image.thumbnail((760, 760), Image.Resampling.LANCZOS)
    image.save(thumb_path, quality=95)
    return crop


def card(product, folder, no_images):
    thumbnail = None
    pid = product["id"]
    if not no_images:
        from AI.vectorizer import fetch_image_for_ai

        if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            raise ValueError(f"Invalid product ID: {pid!r}")
        path = CACHE / "products" / f"{pid}.jpg"
        if not path.exists() and product.get("image_url"):
            image = fetch_image_for_ai(product["image_url"])
            if image is not None:
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                image.save(path, quality=90)
        if path.exists():
            thumbnail = relative_url(path, folder)
    url = product.get("product_url") or ""
    return {"id": pid, "title": product.get("title") or pid, "image": thumbnail,
            "url": url if urlsplit(url).scheme in ("http", "https") else None}


def pool_candidates(folder, stage, no_images):
    from AI.embedding_model import get_embedding_config
    from AI.vectorizer import extract_vector
    from api.category_mapping import get_category_terms
    from api.search_service import compute_category_score

    snapshot = read_json(folder / "results.json")
    if snapshot["stage"] != stage or snapshot.get("smoke"):
        raise ValueError("Use a completed, non-smoke run for grading")
    if not no_images:
        missing = [key for key in R2_VARS if not os.getenv(key, "").strip()]
        if missing:
            raise ValueError("Điền các biến R2: " + ", ".join(R2_VARS) + ". Có thể dùng --no-images.")
    catalog = Catalog.load(resolve_path(snapshot["catalog"]))
    if catalog.sha256 != snapshot["catalog_sha256"]:
        raise ValueError("Catalog changed after inference")
    config = get_embedding_config()
    catalog.check_embedding_space(config.model_name, config.version)
    judgments, statuses = load_judgments()
    (CACHE / "queries").mkdir(parents=True, exist_ok=True)
    if not no_images:
        (CACHE / "products").mkdir(parents=True, exist_ok=True)
    cards_cache = {}
    output = []
    for query in snapshot["queries"]:
        qid = query["query_id"]
        thumb = CACHE / "queries" / f"{qid}.jpg"
        roi = query_roi(query, resolve_path(snapshot["manifest"]).parent / query["image"], thumb)
        vector = extract_vector(roi)
        if vector is None:
            raise ValueError(f"ROI embedding failed for {qid}")
        cosine = list(catalog.ordered_by_similarity(vector))
        primary_ids = set(target_ranked_ids(query)) | {item["id"] for item in cosine[:20]}
        cls = query["class_name"]
        terms = get_category_terms(cls)
        expanded_ids = {item["id"] for item in islice(
            (item for item in cosine if compute_category_score(item, terms, cls) > 0), 60
        )}
        rng = random.Random(qid)

        def cards(ids):
            ordered = sorted(ids)
            rng.shuffle(ordered)
            result = []
            for pid in ordered:
                if (qid, pid) in judgments:
                    continue
                product = catalog.get(pid)
                if product is None:
                    raise ValueError(f"Ranked product outside catalog: {pid}")
                if pid not in cards_cache:
                    cards_cache[pid] = card(product, folder, no_images)
                result.append(cards_cache[pid])
            return result

        output.append({
            "id": qid, "image": relative_url(thumb, folder), "class_name": cls, "color": query["color"],
            "primary": cards(primary_ids), "expanded": cards(expanded_ids),
            "existing_status": statuses.get(qid, "unconfirmed"),
            "previously_acceptable": statuses.get(qid) == "has_match",
        })
        print(f"Pooled {qid}", flush=True)

    payload = json.dumps({"stage": stage, "queries": output}, ensure_ascii=False).replace("<", "\\u003c")
    path = folder / "grading.html"
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__GRADING_DATA__", payload), encoding="utf-8")
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pool = commands.add_parser("pool")
    pool.add_argument("--run-dir", type=Path, required=True)
    pool.add_argument("--stage", choices=("validation", "test"), required=True)
    pool.add_argument("--no-images", action="store_true")
    commands.add_parser("merge").add_argument("grades", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "pool":
            print(pool_candidates(args.run_dir / args.stage, args.stage, args.no_images))
        else:
            new, overwritten, statuses = merge_grades(read_json(args.grades))
            print(f"New: {new}; overwritten: {overwritten}; query statuses: {statuses}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

