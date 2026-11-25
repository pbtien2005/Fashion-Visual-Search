"""Download every catalog product photo once, so each embedding model reads the same bytes.

Runs inside the existing `fashion-visual-search-api` image (boto3, python-dotenv). Reads the R2
keys from .env exactly like AI/vectorizer.fetch_image_for_ai. Existing files are kept.

    evaluation_new/docker_run.sh api python -m evaluation_new.fetch_catalog_images
    evaluation_new/docker_run.sh api python -m evaluation_new.fetch_catalog_images --catalog data/fashion_with_hsv_vectors.json
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/fashion_with_vectors.json"
OUT_DIR = ROOT / "evaluation_new/cache/catalog_images"


def r2_client():
    import boto3
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return boto3.client(
        "s3", endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"), region_name="auto",
    )


def fetch(client, url: str) -> bytes:
    if url.startswith("s3://"):
        parsed = urlparse(url)
        return client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    import httpx

    with httpx.Client(timeout=15.0, follow_redirects=True) as http:
        response = http.get(url)
        response.raise_for_status()
        return response.content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = json.loads(args.catalog.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = r2_client()
    todo = [row for row in rows if row.get("r2_image_url") and not (OUT_DIR / f"{row['id']}.img").exists()]
    print(f"{len(rows)} products, {len(rows) - len(todo)} already cached, {len(todo)} to fetch", flush=True)

    def one(row):
        try:
            data = fetch(client, row["r2_image_url"])
        except Exception as exc:  # noqa: BLE001 - report and continue
            return row["id"], f"{type(exc).__name__}: {exc}"
        temp = OUT_DIR / f"{row['id']}.img.part"
        temp.write_bytes(data)
        temp.replace(OUT_DIR / f"{row['id']}.img")
        return row["id"], None

    failures = {}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for n, (product_id, error) in enumerate(pool.map(one, todo), 1):
            if error:
                failures[product_id] = error
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} (lỗi {len(failures)})", flush=True)

    index = {}
    for row in rows:
        path = OUT_DIR / f"{row['id']}.img"
        if path.exists():
            index[row["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT_DIR.parent / "catalog_images_index.json").write_text(
        json.dumps({"catalog": args.catalog.resolve().relative_to(ROOT).as_posix(), "count": len(index),
                    "failures": failures, "sha256": index}, indent=1), encoding="utf-8")
    print(f"Có {len(index)}/{len(rows)} ảnh. Lỗi: {len(failures)}")
    for product_id, error in list(failures.items())[:10]:
        print(f"  {product_id}: {error}")


if __name__ == "__main__":
    main()
