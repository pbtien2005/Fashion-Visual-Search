"""Download licensed Wikimedia candidates with all four annotation fields blank."""

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from evaluation.build_wikimedia_real_eval import (
    REAL_CATEGORIES, USER_AGENT, commons_file_page, download_image, list_category_files,
    metadata_value, query_commons,
)

from .manifest import EVAL_DIR, FIELDS, SPLITS, file_hash, manifest_path, read_csv, source_key, write_csv

EXTRA_CATEGORIES = {
    "shirt": "Men wearing shirts",
    "tshirt": "Men wearing T-shirts",
    "polo": "Men wearing polo shirts",
    "trousers": "Women wearing trousers",
    "jeans": "Women wearing jeans",
    "skirt": "Women wearing skirts",
    "dress": "Women wearing dresses",
    "cardigan": "Women wearing cardigans",
    "jacket": "Men wearing jackets",
}
MIN_CANDIDATES = 80


def throttle(interval):
    next_request = 0.0

    def pace(_request):
        nonlocal next_request
        sleep(max(0, next_request - monotonic()))
        next_request = monotonic() + interval

    return pace


def category_status(client, categories):
    payload = query_commons(client, {
        "action": "query", "titles": "|".join(categories.values()), "prop": "categoryinfo",
    })
    return {page["title"].casefold(): page.get("categoryinfo", {}).get("files", 0)
            for page in payload["query"]["pages"].values()}


def image_metadata(client, titles, width):
    payload = query_commons(client, {
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": width,
        "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
    })
    for page in payload["query"]["pages"].values():
        infos = page.get("imageinfo") or []
        if infos:
            yield page["title"], infos[0]


def download_candidates(client, per_category, width):
    categories = {row["slug"]: row["commons_category"] for row in REAL_CATEGORIES}
    categories.update({slug: f"Category:{name}" for slug, name in EXTRA_CATEGORIES.items()})
    existing_categories = category_status(client, categories)

    folder = EVAL_DIR / "queries" / "bench"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "candidates.csv"
    labeled = [row for split in SPLITS for row in read_csv(manifest_path(split), FIELDS)]
    hashes = {file_hash(EVAL_DIR / row["image"]) for row in labeled}
    sources = {source_key(row["source"]) for row in labeled}
    for path in (EVAL_DIR / "queries" / "real_outfits").glob("*.jpg"):
        hashes.add(file_hash(path))
    # The old sources include the one real_outfits image that has no bbox label.
    old_sources = json.loads((EVAL_DIR / "real_outfit_sources.json").read_text(encoding="utf-8"))
    sources.update(source_key(row["source_url"]) for row in old_sources)

    previous = read_csv(csv_path, FIELDS) if csv_path.exists() else []
    candidate_rows = []
    for row in previous:
        path = EVAL_DIR / row["image"]
        if source_key(row["source"]) in sources or not path.is_file():
            continue
        digest = file_hash(path)
        if digest not in hashes:
            hashes.add(digest)
            sources.add(source_key(row["source"]))
            candidate_rows.append(row)
    counts = Counter(row["query_id"].rsplit("_", 1)[0] for row in candidate_rows)
    # Labeled rows leave candidates.csv, so also reserve manifest IDs and files already on disk.
    ids = {row["query_id"] for row in (*previous, *labeled)} | {path.stem for path in folder.glob("*.jpg")}
    write_csv(csv_path, FIELDS, candidate_rows)

    with TemporaryDirectory(dir=folder) as temp_dir:
        downloaded = Path(temp_dir) / "download"
        converted = Path(temp_dir) / "converted.jpg"
        for slug, category in categories.items():
            limit = existing_categories.get(category.casefold(), 0)
            if not limit:
                print(f"{category}: skipped (missing or no direct images)", flush=True)
                continue
            prefix = f"bench_{slug}"
            if counts[prefix] >= per_category:
                continue
            titles = list_category_files(client, category, min(50, max(20, per_category * 4)))
            titles = [title for title in titles if source_key(commons_file_page(title)) not in sources]
            for offset in range(0, len(titles), 20):
                if counts[prefix] >= per_category:
                    break
                for title, info in image_metadata(client, titles[offset:offset + 20], width):
                    if counts[prefix] >= per_category:
                        break
                    source = info.get("descriptionurl") or commons_file_page(title)
                    meta = info.get("extmetadata") or {}
                    license_name = metadata_value(meta, "LicenseShortName")
                    author = metadata_value(meta, "Artist")
                    url = info.get("thumburl") or info.get("url")
                    if not (license_name and author and url and info.get("mime", "").startswith("image/")
                            and source_key(source) not in sources):
                        continue
                    try:
                        if not download_image(client, url, downloaded):
                            continue
                        raw_hash = file_hash(downloaded)
                        with Image.open(downloaded) as image:
                            # Keep real photographs and other valid raster files; verify decoding.
                            oriented = ImageOps.exif_transpose(image).convert("RGB")
                            oriented.thumbnail((width, width), Image.Resampling.LANCZOS)
                            oriented.save(converted, quality=95)
                    except (httpx.HTTPError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                        print(f"Skipping {title}: {exc}", flush=True)
                        continue
                    digest = file_hash(converted)
                    if raw_hash in hashes or digest in hashes:
                        continue
                    number = 1
                    while (query_id := f"{prefix}_{number:02}") in ids:
                        number += 1
                    path = folder / f"{query_id}.jpg"
                    converted.replace(path)
                    ids.add(query_id)
                    hashes.update((raw_hash, digest))
                    sources.add(source_key(source))
                    candidate_rows.append(dict.fromkeys(FIELDS, "") | {
                        "query_id": query_id, "image": path.relative_to(EVAL_DIR).as_posix(), "source": source,
                        "license": html.unescape(license_name), "author": html.unescape(author),
                        "license_url": metadata_value(meta, "LicenseUrl") or "",
                    })
                    counts[prefix] += 1
                    write_csv(csv_path, FIELDS, candidate_rows)
            print(f"{category}: {counts[prefix]} candidates; total {len(candidate_rows)}", flush=True)
    return len(candidate_rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--width", type=int, default=900)
    args = parser.parse_args()
    if args.per_category < 1 or args.width < 1:
        parser.error("--per-category and --width must be positive")
    try:
        with httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": USER_AGENT},
                          event_hooks={"request": [throttle(.8)]}) as client:
            count = download_candidates(client, args.per_category, args.width)
    except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    if count < MIN_CANDIDATES:
        parser.exit(1, f"Only {count}/{MIN_CANDIDATES} licensed candidates downloaded\n")
    print(f"{count} licensed candidates")


if __name__ == "__main__":
    main()

