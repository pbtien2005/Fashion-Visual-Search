"""Separate, append-free catalog descriptor cache; never writes catalog/vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from AI.hsv_descriptor import ColorDescriptor, HSVConfig, extract_hsv_descriptor, validate_descriptor


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roi_record(record, root):
    """Manifest images are already source RGB ROIs; mask must share coordinates."""
    path = Path(root) / record["image"]
    with Image.open(path) as image:
        roi = np.asarray(image.convert("RGB")).copy()
    mask = None
    hashes = {"image": file_hash(path)}
    if record.get("mask"):
        mask_path = Path(root) / record["mask"]
        try:
            with Image.open(mask_path) as image:
                mask = np.asarray(image).copy()
            hashes["mask"] = file_hash(mask_path)
        except (OSError, ValueError) as exc:
            mask = np.array([], dtype=np.uint8)
            hashes["mask_error"] = str(exc)
    return roi, mask, hashes


def descriptor_cache_key(product_id, model_name, model_version, roi_policy, color_version,
                         source_hashes=None):
    identity = [str(product_id), model_name, model_version, roi_policy, color_version,
                source_hashes or {}]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def read_cached_descriptor(cache, key, model_name, model_version, roi_policy, color_version):
    """Fail closed on version/model/policy mismatch; never reuse a different descriptor."""
    entry = cache["entries"][key]
    descriptor = ColorDescriptor(**entry["descriptor"])
    if (cache["color_version"] != color_version or descriptor.version != color_version
            or descriptor.roi_policy != roi_policy or entry["embedding_model"] != model_name
            or entry["embedding_version"] != model_version):
        raise ValueError("Incompatible color cache identity")
    expected_key = descriptor_cache_key(entry["product_id"], model_name, model_version,
                                        roi_policy, color_version, entry["source_hashes"])
    if key != expected_key:
        raise ValueError("Color cache key mismatch")
    validate_descriptor(descriptor)
    return descriptor


def catalog_descriptor_for_query(cache, product_id, query_color, model_name, model_version):
    """M3 handoff helper: validate query/config/policy and resolve one catalog item.

    Raises KeyError for missing IDs, ValueError for incompatible/ambiguous entries.
    Scoring and missing-descriptor ranking policy remain M4's responsibility.
    """
    validate_descriptor(query_color)
    if cache.get("schema") != "m3-color-cache-v1" or cache.get("config") != query_color.config:
        raise ValueError("Query/catalog descriptor config mismatch")
    matches = [key for key, entry in cache["entries"].items()
               if entry["product_id"] == str(product_id)]
    if not matches:
        raise KeyError(f"Missing catalog color descriptor: {product_id}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous catalog color descriptor: {product_id}")
    return read_cached_descriptor(cache, matches[0], model_name, model_version,
                                  query_color.roi_policy, query_color.version)


def build_color_cache(records, root, config=None, model_name="patrickjohncyh/fashion-clip",
                      model_version=2, roi_policy="source_roi_v1"):
    config = config or HSVConfig()
    entries, failures, statuses = {}, [], {}
    seen = set()
    for record in records:
        product_id = str(record["id"])
        if product_id in seen:
            raise ValueError(f"Duplicate product ID: {product_id}")
        seen.add(product_id)
        try:
            roi, mask, hashes = load_roi_record(record, root)
            descriptor = extract_hsv_descriptor(roi, mask, config,
                         {"roi_policy": roi_policy, "roi_source": record.get("roi_source", "manifest_roi")})
            statuses[descriptor.mask_status] = statuses.get(descriptor.mask_status, 0) + 1
            if not descriptor.valid_pixels:
                raise ValueError("No valid color pixels")
            key = descriptor_cache_key(product_id, model_name, model_version, roi_policy,
                                       config.version, hashes)
            entries[key] = {"product_id": product_id, "embedding_model": model_name,
                            "embedding_version": model_version, "source_hashes": hashes,
                            "descriptor": descriptor.to_dict()}
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"product_id": product_id, "error": str(exc)})
    return {"schema": "m3-color-cache-v1", "color_version": config.version,
            "config": config.spec(), "entries": entries, "failures": failures,
            "coverage": {"total": len(records), "extracted": len(entries),
                         "fraction": len(entries) / len(records) if records else 0,
                         "mask_status_counts": statuses}}


def cache_from_extracted_items(items, config, model_name, model_version, roi_policy):
    """Build a separate cache from production HSV outputs, without re-encoding."""
    entries, failures, statuses, seen = {}, [], {}, set()
    for index, item in enumerate(items):
        product_id = str(item.get("id") or item.get("asin") or "")
        if not product_id or product_id in seen:
            raise ValueError(f"Missing or duplicate product ID at record {index}: {product_id}")
        seen.add(product_id)
        try:
            color = ColorDescriptor(**item["color_descriptor"])
            validate_descriptor(color)
            if (color.config != config.spec() or color.version != config.version
                    or color.roi_policy != roi_policy or not color.valid_pixels):
                raise ValueError("descriptor_config_policy_or_pixels_mismatch")
            hashes = color.metadata.get("source_hashes", {})
            key = descriptor_cache_key(product_id, model_name, model_version, roi_policy, color.version, hashes)
            entries[key] = {"product_id": product_id, "embedding_model": model_name,
                            "embedding_version": model_version, "source_hashes": hashes,
                            "descriptor": color.to_dict()}
            statuses[color.mask_status] = statuses.get(color.mask_status, 0) + 1
        except (KeyError, ValueError, TypeError) as exc:
            failures.append({"product_id": product_id, "error": str(exc)})
    return {"schema": "m3-color-cache-v1", "config": config.spec(), "color_version": config.version,
            "entries": entries, "failures": failures,
            "coverage": {"total": len(items), "extracted": len(entries),
                         "fraction": len(entries) / len(items) if items else 0,
                         "mask_status_counts": statuses}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-name", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--model-version", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = HSVConfig(**json.loads(args.config.read_text())) if args.config else HSVConfig()
    cache = build_color_cache(manifest["catalog"], args.manifest.parent, config,
                              args.model_name, args.model_version,
                              manifest.get("roi_policy", "source_roi_v1"))
    # Exclusive create: an existing cache, source catalog or vector file cannot be overwritten.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(cache, stream, indent=2, allow_nan=False)
    print(json.dumps(cache["coverage"]))


if __name__ == "__main__":
    main()
