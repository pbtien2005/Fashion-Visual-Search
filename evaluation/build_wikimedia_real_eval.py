import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import httpx


EVAL_DIR = Path(__file__).resolve().parent
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FashionVisualSearchEval/0.1 (local academic evaluation)"

REAL_CATEGORIES = [
    {
        "slug": "green_tshirt",
        "commons_category": "Category:Men wearing green T-shirts",
        "type": "short_sleeve_top",
        "expected_categories": ["shirt", "top", "t-shirt", "tee"],
        "expected_color": "green",
    },
    {
        "slug": "white_shirt",
        "commons_category": "Category:Women wearing white shirts",
        "type": "short_sleeve_top",
        "expected_categories": ["shirt", "top", "blouse"],
        "expected_color": "white",
    },
    {
        "slug": "blue_jeans",
        "commons_category": "Category:Men wearing blue jeans",
        "type": "trousers",
        "expected_categories": ["pants", "trousers", "jeans"],
        "expected_color": "blue",
    },
    {
        "slug": "black_trousers",
        "commons_category": "Category:Men wearing black trousers",
        "type": "trousers",
        "expected_categories": ["pants", "trousers"],
        "expected_color": "black",
    },
    {
        "slug": "shorts",
        "commons_category": "Category:Men wearing shorts",
        "type": "shorts",
        "expected_categories": ["shorts"],
        "expected_color": None,
    },
    {
        "slug": "black_skirt",
        "commons_category": "Category:Women wearing black skirts",
        "type": "skirt",
        "expected_categories": ["skirt"],
        "expected_color": "black",
    },
    {
        "slug": "red_dress",
        "commons_category": "Category:Women wearing red dresses",
        "type": "dress",
        "expected_categories": ["dress"],
        "expected_color": "red",
    },
    {
        "slug": "black_dress",
        "commons_category": "Category:Women wearing black dresses",
        "type": "dress",
        "expected_categories": ["dress"],
        "expected_color": "black",
    },
]


def safe_filename(value: str) -> str:
    value = value.replace("File:", "")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._")[:90] or "image"


def commons_file_page(title: str) -> str:
    file_name = title.replace("File:", "")
    return f"https://commons.wikimedia.org/wiki/File:{quote(file_name.replace(' ', '_'))}"


def query_commons(client: httpx.Client, params: dict) -> dict:
    for attempt in range(1, 4):
        response = client.get(
            COMMONS_API,
            params={"format": "json", **params},
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code == 429 and attempt < 3:
            time.sleep(2 * attempt)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Wikimedia Commons API did not return a response")


def safe_log(message: str):
    print(message.encode("ascii", errors="replace").decode("ascii"))


def list_category_files(client: httpx.Client, category: str, limit: int) -> list[str]:
    payload = query_commons(
        client,
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": limit,
        },
    )
    return [member["title"] for member in payload.get("query", {}).get("categorymembers", [])]


def get_image_info(client: httpx.Client, title: str, width: int) -> dict | None:
    payload = query_commons(
        client,
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": width,
        },
    )
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        imageinfo = page.get("imageinfo") or []
        if imageinfo:
            return imageinfo[0]
    return None


def metadata_value(extmetadata: dict, key: str) -> str | None:
    value = (extmetadata.get(key) or {}).get("value")
    if value is None:
        return None
    return re.sub(r"<[^>]+>", "", str(value)).strip() or None


def download_image(client: httpx.Client, url: str, output_path: Path, retries: int = 3) -> bool:
    for attempt in range(1, retries + 1):
        response = client.get(url, headers={"User-Agent": USER_AGENT})
        if response.status_code == 429 and attempt < retries:
            time.sleep(2 * attempt)
            continue

        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return False
        output_path.write_bytes(response.content)
        return True

    return False


def build_real_eval(api_limit_per_category: int, per_category: int, width: int, timeout: float):
    image_dir = EVAL_DIR / "queries" / "real_outfits"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    sources = []

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for config in REAL_CATEGORIES:
            titles = list_category_files(
                client,
                category=config["commons_category"],
                limit=api_limit_per_category,
            )
            selected_count = 0

            for title in titles:
                if selected_count >= per_category:
                    break

                try:
                    info = get_image_info(client, title=title, width=width)
                except Exception as exc:
                    safe_log(f"Skip {title}: {exc}")
                    time.sleep(1.0)
                    continue
                if not info:
                    continue

                image_url = info.get("thumburl") or info.get("url")
                if not image_url:
                    continue

                query_id = f"real_{config['slug']}_{selected_count + 1:02d}"
                output_name = f"{query_id}_{safe_filename(title)}.jpg"
                output_path = image_dir / output_name

                try:
                    if not output_path.exists():
                        ok = download_image(client, image_url, output_path)
                        if not ok:
                            continue
                except Exception as exc:
                    safe_log(f"Skip {title}: {exc}")
                    continue

                expected_item = {
                    "type": config["type"],
                    "expected_categories": config["expected_categories"],
                    "acceptable_product_ids": [],
                }
                if config.get("expected_color"):
                    expected_item["expected_color"] = config["expected_color"]

                labels.append(
                    {
                        "query_id": query_id,
                        "image": f"queries/real_outfits/{output_name}",
                        "source": "Wikimedia Commons",
                        "source_url": commons_file_page(title),
                        "expected_items": [expected_item],
                    }
                )

                extmetadata = info.get("extmetadata") or {}
                sources.append(
                    {
                        "query_id": query_id,
                        "file": output_name,
                        "commons_title": title,
                        "source_url": commons_file_page(title),
                        "author": metadata_value(extmetadata, "Artist"),
                        "license": metadata_value(extmetadata, "LicenseShortName"),
                        "license_url": metadata_value(extmetadata, "LicenseUrl"),
                        "description": metadata_value(extmetadata, "ImageDescription"),
                    }
                )
                selected_count += 1
                time.sleep(0.8)

    labels_path = EVAL_DIR / "labels.real.json"
    sources_json_path = EVAL_DIR / "real_outfit_sources.json"
    sources_md_path = EVAL_DIR / "real_outfit_sources.md"

    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    sources_json_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Real Outfit Evaluation Sources", ""]
    for source in sources:
        lines.append(
            f"- `{source['query_id']}`: [{source['commons_title']}]({source['source_url']})"
            f" — author: {source.get('author') or 'unknown'}"
            f" — license: {source.get('license') or 'unknown'}"
        )
    sources_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {labels_path} with {len(labels)} labels.")
    print(f"Wrote {sources_json_path}.")
    print(f"Wrote {sources_md_path}.")


def main():
    parser = argparse.ArgumentParser(description="Build real outfit eval images from Wikimedia Commons.")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--api-limit-per-category", type=int, default=30)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    build_real_eval(
        api_limit_per_category=args.api_limit_per_category,
        per_category=args.per_category,
        width=args.width,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
