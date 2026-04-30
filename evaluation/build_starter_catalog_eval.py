import argparse
import json
import re
from pathlib import Path

import httpx


EVAL_DIR = Path(__file__).resolve().parent

CATEGORY_GROUPS = {
    "short_sleeve_top": {
        "needles": ["t-shirts", "t-shirts", "polos", "button-down shirts", "tops & tees", "tanks & camis"],
        "terms": ["shirt", "top", "t-shirt", "tee", "polo", "blouse"],
    },
    "long_sleeve_top": {
        "needles": ["sweaters", "cardigans", "pullovers"],
        "terms": ["shirt", "top", "sweater", "cardigan", "pullover", "long sleeve"],
    },
    "long_sleeve_outwear": {
        "needles": ["jackets", "coats", "blazers"],
        "terms": ["jacket", "coat", "outwear", "outerwear", "blazer"],
    },
    "trousers": {
        "needles": ["pants", "jeans", "sweatpants", "trousers"],
        "terms": ["pants", "trousers", "jeans", "chino", "sweatpants"],
    },
    "shorts": {
        "needles": ["shorts"],
        "terms": ["shorts"],
    },
    "skirt": {
        "needles": ["skirts"],
        "terms": ["skirt"],
    },
    "dress": {
        "needles": ["dresses"],
        "terms": ["dress"],
    },
}


def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower())


def choose_group(item: dict) -> str | None:
    haystack = normalize_text(f"{item.get('category')} {item.get('title')}")
    for group_name, config in CATEGORY_GROUPS.items():
        if any(needle in haystack for needle in config["needles"]):
            return group_name
    return None


def load_products(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_selection(products: list[dict], per_group: int) -> list[tuple[str, dict]]:
    selected = []
    counts = {group_name: 0 for group_name in CATEGORY_GROUPS}

    for item in products:
        product_id = item.get("id") or item.get("asin")
        if not product_id or not item.get("dominant_color_name"):
            continue

        group_name = choose_group(item)
        if not group_name or counts[group_name] >= per_group:
            continue

        selected.append((group_name, item))
        counts[group_name] += 1

        if all(count >= per_group for count in counts.values()):
            break

    return selected


def download_product_image(client: httpx.Client, api_url: str, product_id: str, output_path: Path) -> bool:
    url = f"{api_url.rstrip('/')}/api/products/{product_id}/image"
    try:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        print(f"Skip {product_id}: image download failed: {exc}")
        return False

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        print(f"Skip {product_id}: unsupported content type {content_type}")
        return False

    output_path.write_bytes(response.content)
    return True


def build_labels(selected: list[tuple[str, dict]], image_dir_name: str) -> list[dict]:
    labels = []
    for group_name, item in selected:
        product_id = str(item.get("id") or item.get("asin"))
        labels.append(
            {
                "query_id": f"starter_{product_id}",
                "image": f"queries/{image_dir_name}/{product_id}.jpg",
                "expected_items": [
                    {
                        "type": group_name,
                        "expected_categories": CATEGORY_GROUPS[group_name]["terms"],
                        "expected_color": item.get("dominant_color_name"),
                        "acceptable_product_ids": [product_id],
                    }
                ],
            }
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build a starter catalog/model-photo evaluation set.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--source", default="data/fashion_with_vectors.json")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--image-dir", default="starter_catalog")
    parser.add_argument("--labels", default="evaluation/labels.starter_catalog.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    products = load_products(Path(args.source))
    selected = build_selection(products, per_group=args.per_group)
    image_dir = EVAL_DIR / "queries" / args.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    with httpx.Client(timeout=args.timeout) as client:
        for group_name, item in selected:
            product_id = str(item.get("id") or item.get("asin"))
            output_path = image_dir / f"{product_id}.jpg"
            if output_path.exists() or download_product_image(client, args.api_url, product_id, output_path):
                downloaded.append((group_name, item))

    labels = build_labels(downloaded, image_dir_name=args.image_dir)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {}
    for group_name, _ in downloaded:
        counts[group_name] = counts.get(group_name, 0) + 1

    print(f"Selected {len(selected)} products.")
    print(f"Downloaded {len(downloaded)} images.")
    print(f"Wrote {labels_path}.")
    print(f"Group counts: {counts}")


if __name__ == "__main__":
    main()
