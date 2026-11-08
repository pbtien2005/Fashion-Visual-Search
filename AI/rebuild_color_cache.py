"""Rebuild only HSV147 colors from the full local/remote catalog, no embeddings.

Writes a new sidecar cache. Production and this CLI share catalog ROI preparation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-remote", action="store_true", help="Fetch catalog R2/HTTP images")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Explicit whole-original-image fallback; log/record this for every item")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new cache path")
    if args.config:
        import os
        os.environ["HSV_CONFIG_PATH"] = str(args.config.resolve())
    from PIL import Image
    from AI.color_cache import cache_from_extracted_items
    from AI.embedding_model import get_embedding_config
    from AI.production_color import production_config, ROI_POLICY
    from AI.vectorizer import catalog_color_inputs, fetch_image_for_ai

    config, embedding_config = production_config(), get_embedding_config()
    items = json.loads(args.catalog.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list):
        parser.error("Catalog must be a JSON list; use AI.color_cache for a prepared ROI manifest")
    processed = []
    source_failures = []
    for index, item in enumerate(items):
        product_id = str(item.get("id") or item.get("asin") or "")
        record = {"id": product_id, "color_descriptor": None}
        try:
            local = item.get("image_path") or item.get("image")
            if local:
                with Image.open(args.catalog.parent / local) as image:
                    image = image.convert("RGB")
            elif args.allow_remote:
                url = item.get("r2_image_url") or item.get("thumbnailImage") or item.get("image_url")
                image = fetch_image_for_ai(url) if url else None
                if image is None:
                    raise ValueError("missing_or_unavailable_image")
            else:
                raise ValueError("No local image; --allow-remote required for network/R2")
            _, color = catalog_color_inputs(item, image, detect=not args.skip_detection)
            record["color_descriptor"] = color.to_dict() if color else None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            source_failures.append({"record_index": index, "product_id": product_id, "error": str(exc)})
        processed.append(record)
    cache = cache_from_extracted_items(processed, config, embedding_config.model_name,
                                       embedding_config.version, ROI_POLICY)
    cache["source_failures"] = source_failures
    cache["source_catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(cache, stream, indent=2, allow_nan=False)
    print(json.dumps({"coverage": cache["coverage"], "failed": len(cache["failures"])}))
    if cache["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
