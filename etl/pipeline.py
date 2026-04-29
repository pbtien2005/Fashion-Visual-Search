import json
import os

from dotenv import load_dotenv

from etl.r2_uploader import get_r2_client, upload_image_to_r2
from etl.scraper import fetch_amazon_products

load_dotenv()


def _load_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def run_etl_pipeline(search_keyword: str, max_items: int = 50, output_file: str = "data/fashion_metadata.json"):
    print(f"=== START ETL PIPELINE FOR KEYWORD: '{search_keyword}' ===")

    history_file = "data/scraped_history.json"
    existing_metadata = _load_json_list(output_file)
    metadata_by_id = {item.get("id"): item for item in existing_metadata if item.get("id")}
    seen_ids = set(metadata_by_id)

    if os.path.exists(history_file):
        seen_ids.update(_load_json_list(history_file))
        print(f"Loaded {len(seen_ids)} known IDs for duplicate filtering.")

    items = fetch_amazon_products(search_keyword=search_keyword, max_items=max_items)
    if not items:
        print("No items returned from scraper.")
        return

    r2_client = get_r2_client()
    bucket_name = os.getenv("R2_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("Missing R2_BUCKET_NAME in .env")

    processed_count = 0

    for index, item in enumerate(items):
        item_id = item.get("asin")
        if not item_id:
            continue

        if item_id in seen_ids:
            print(f"[{index + 1}/{len(items)}] Skip {item_id} (already seen).")
            continue

        image_url = item.get("thumbnailImage")
        if not image_url:
            print(f"[{index + 1}/{len(items)}] Skip {item_id} (missing thumbnail).")
            continue

        print(f"[{index + 1}/{len(items)}] Processing new product: {item_id}")
        object_name = f"images/{item_id}.jpg"
        r2_path = upload_image_to_r2(r2_client, image_url, bucket_name, object_name)

        if not r2_path:
            continue

        metadata_by_id[item_id] = {
            "id": item_id,
            "title": item.get("title"),
            "brand": item.get("brand"),
            "price": item.get("price"),
            "category": item.get("category"),
            "original_url": item.get("url"),
            "r2_image_url": r2_path,
        }
        seen_ids.add(item_id)
        processed_count += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(metadata_by_id.values()), f, ensure_ascii=False, indent=4)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(sorted(seen_ids), f, ensure_ascii=False, indent=2)

    print(f"Done. Added {processed_count} new products. Metadata total: {len(metadata_by_id)}.")


if __name__ == "__main__":
    run_etl_pipeline(search_keyword="men graphic t-shirt", max_items=15)
