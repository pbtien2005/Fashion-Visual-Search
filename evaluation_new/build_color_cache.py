"""Rebuild the HSV147 colour cache for the full catalog with the unchanged production script.

No full cache exists (data/color_cache_hsv147.json covers only the 28-product sample), and
evaluation/BENCHMARK.md requires one: rerank_products puts each product's HSV colour name into
the text/style prompt, so a missing colour changes B0/C3 rankings.

This runs AI/rebuild_color_cache.py as is (catalog_color_inputs -> /detect/upload -> ROI ->
HSV147). The only change: images come from evaluation_new/cache/catalog_images/ (already
downloaded from R2 by fetch_catalog_images.py) instead of a second R2 download. The decode is
the same as AI.vectorizer.fetch_image_for_ai: Image.open(bytes).convert("RGB").

Run in the api image, next to a detection service (see README, step 2b):
    python -m evaluation_new.build_color_cache
Check: python -m evaluation_new.build_color_cache --verify
"""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # does not override variables set by docker run

CATALOG = ROOT / "data/fashion_with_vectors.json"
IMAGES = ROOT / "evaluation_new/cache/catalog_images"
OUTPUT = ROOT / "evaluation_new/cache/color_cache_hsv147_full.json"
SAMPLE = ROOT / "data/color_cache_hsv147.json"


def local_fetcher():
    from PIL import Image

    by_url = {row["r2_image_url"]: str(row["id"])
              for row in json.loads(CATALOG.read_text(encoding="utf-8")) if row.get("r2_image_url")}

    def fetch_image_for_ai(image_url):
        path = IMAGES / f"{by_url[image_url]}.img"
        return Image.open(BytesIO(path.read_bytes())).convert("RGB")

    return fetch_image_for_ai


def output_path(skip_detection):
    return OUTPUT.with_name(OUTPUT.stem + "_wholeimage.json") if skip_detection else OUTPUT


def build(skip_detection=False):
    import AI.vectorizer
    from AI import rebuild_color_cache

    AI.vectorizer.fetch_image_for_ai = local_fetcher()  # main() imports it from the module at call time
    sys.argv = ["rebuild_color_cache", "--catalog", str(CATALOG), "--output", str(output_path(skip_detection)),
                "--allow-remote"] + (["--skip-detection"] if skip_detection else [])
    rebuild_color_cache.main()


def verify(skip_detection=False):
    """Compare with the 28-product sample cache made by the team: same ROI pixels, same descriptor."""
    full = json.loads(output_path(skip_detection).read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    by_id = {entry["product_id"]: entry for entry in full["entries"].values()}
    same_roi = same_descriptor = 0
    for entry in sample["entries"].values():
        other = by_id.get(entry["product_id"])
        if other is None:
            print("thiếu", entry["product_id"])
            continue
        same_roi += other["source_hashes"] == entry["source_hashes"]
        same_descriptor += other["descriptor"] == entry["descriptor"]
    total = len(sample["entries"])
    print(f"config giống: {full['config'] == sample['config']}, color_version giống: "
          f"{full['color_version'] == sample['color_version']}")
    print(f"ROI pixel hash trùng: {same_roi}/{total}; descriptor trùng: {same_descriptor}/{total}")
    print(f"coverage: {full['coverage']}; lỗi nguồn: {len(full.get('source_failures', []))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--skip-detection", action="store_true",
                        help="whole-image ROI, like data/color_cache_hsv147.json (the 28-product sample)")
    args = parser.parse_args()
    if args.verify:
        verify(args.skip_detection)
    else:
        build(args.skip_detection)
