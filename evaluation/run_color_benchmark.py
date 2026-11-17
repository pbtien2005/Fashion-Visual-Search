"""Offline M3 color-only benchmark. Does not modify M4 weights or use fake embeddings."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from AI.color_cache import file_hash, load_roi_record
from AI.color_utils import color_similarity, extract_dominant_color, name_color
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor, histogram_similarity

ROOT = Path(__file__).resolve().parents[1]


def synthetic_records():
    """Known rectangular foreground masks, no inferred clothing segmentation."""
    colors = {"white": (245, 245, 245), "gray": (125, 125, 125), "black": (15, 15, 15),
              "dark_blue": (8, 15, 50), "low_saturation": (150, 140, 140),
              "red": (195, 25, 25), "blue": (25, 70, 195), "multicolor": (0, 0, 0)}
    catalog, queries = [], []
    for key, rgb in colors.items():
        roi = np.full((64, 64, 3), rgb, dtype=np.uint8)
        if key == "multicolor":
            roi[:, :32] = colors["red"]
            roi[:, 32:] = colors["blue"]
        mask = np.ones((64, 64), dtype=bool)
        catalog.append({"id": key, "roi": roi, "mask_array": mask,
                        "roi_source": "synthetic_ground_truth"})
        for background in ((255, 255, 255), (30, 130, 50), (180, 100, 75)):
            canvas = np.full((112, 112, 3), background, dtype=np.uint8)
            canvas[24:88, 24:88] = roi
            foreground = np.zeros((112, 112), dtype=bool)
            foreground[24:88, 24:88] = True
            if key == "white":
                canvas[24:28, 24:40] = (210, 150, 110)  # Small skin region; mask preserves it.
            queries.append({"id": f"{key}_{background[0]}", "roi": canvas,
                            "mask_array": foreground, "relevant_ids": [key],
                            "roi_source": "synthetic_ground_truth"})
    return catalog, queries


def photo_records():
    """Actual local photos, controlled injected border and exact placement mask.

    These are transformed instances of catalog images, NOT independent queries.
    Masks exclude only injected borders, not background within original photos.
    """
    catalog, queries, hashes = [], [], {}
    for path in sorted((ROOT / "evaluation/queries/starter_catalog").glob("*.jpg")):
        with Image.open(path) as image:
            roi = np.asarray(image.convert("RGB")).copy()
        height, width = roi.shape[:2]
        border = max(1, min(height, width) // 3)
        canvas = np.full((height + 2 * border, width + 2 * border, 3),
                         (40, 120, 60), dtype=np.uint8)
        canvas[border:border+height, border:border+width] = roi
        mask = np.zeros(canvas.shape[:2], dtype=bool)
        mask[border:border+height, border:border+width] = True
        catalog.append({"id": path.stem, "roi": roi, "mask_array": np.ones(roi.shape[:2], bool),
                        "roi_source": "whole_catalog_photo"})
        queries.append({"id": path.stem, "roi": canvas, "mask_array": mask,
                        "relevant_ids": [path.stem], "roi_source": "photo_with_injected_border"})
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    if not catalog:
        raise ValueError("No starter catalog photos found")
    return catalog, queries, hashes


def materialize(records, root):
    output = []
    for record in records:
        try:
            roi, mask, hashes = load_roi_record(record, root)
            output.append({**record, "roi": roi, "mask_array": mask, "hashes": hashes})
        except Exception as exc:
            output.append({**record, "load_error": str(exc)})
    return output


def extract(record, config, masked, rgb):
    if "load_error" in record:
        raise ValueError(record["load_error"])
    roi = record["roi"]
    mask = record.get("mask_array") if masked else None
    metadata = {"roi_source": record.get("roi_source", "manifest_roi"),
                "roi_policy": record.get("roi_policy", "source_roi_v1")}
    descriptor = extract_hsv_descriptor(roi, mask, config, metadata)
    if not descriptor.valid_pixels:
        raise ValueError(f"No valid pixels: {descriptor.mask_status}")
    if rgb:
        # Baseline RGB extractor operates on the same selected pixels in mask arm.
        if masked and descriptor.mask_status == "valid":
            roi = roi[mask != 0].reshape(-1, 1, 3)
        dominant = extract_dominant_color(Image.fromarray(roi))
        return (dominant, name_color(dominant)), descriptor
    return descriptor, descriptor


def run_variant(catalog, queries, config, masked, rgb=False, metric="hellinger", top_k=5):
    descriptors, failures, statuses, extraction_ms = {}, [], Counter(), []
    def prepare(record, role):
        start = time.perf_counter()
        try:
            value, audit = extract(record, config, masked, rgb)
            statuses[f"{role}:{audit.mask_status}"] += 1
            return value, audit
        finally:
            extraction_ms.append((time.perf_counter() - start) * 1000)
    for record in catalog:
        try:
            descriptors[record["id"]] = prepare(record, "catalog")[0]
        except Exception as exc:
            failures.append({"role": "catalog", "id": record["id"], "error": str(exc)})
    results = []
    for record in queries:
        row = {"id": record["id"], "hit_at_1": 0, "hit_at_k": 0,
               "recall_at_k": 0.0, "reciprocal_rank": 0.0, "ranking": []}
        try:
            value, audit = prepare(record, "query")
            relevant = set(record["relevant_ids"])
            scored = []
            for product_id, product in descriptors.items():
                score = (color_similarity(value[0], product[0], value[1], product[1]) if rgb
                         else histogram_similarity(value, product, metric))
                scored.append((score, product_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            ids = [item[1] for item in scored]
            row.update(hit_at_1=int(bool(ids) and ids[0] in relevant),
                       hit_at_k=int(bool(set(ids[:top_k]) & relevant)),
                       recall_at_k=len(set(ids[:top_k]) & relevant) / len(relevant),
                       reciprocal_rank=next((1 / rank for rank, pid in enumerate(ids, 1)
                                             if pid in relevant), 0),
                       ranking=ids[:top_k], mask_status=audit.mask_status,
                       fallback_reason=audit.fallback_reason)
        except Exception as exc:
            row["error"] = str(exc)
            failures.append({"role": "query", "id": record["id"], "error": str(exc)})
        results.append(row)  # Failed queries remain in every denominator.
    metrics = {key: float(np.mean([row[key] for row in results])) for key in
               ("hit_at_1", "hit_at_k", "recall_at_k", "reciprocal_rank")}
    return {"config": config.spec(), "version": config.version,
            "masked": masked, "metric": "legacy_rgb" if rgb else metric, "metrics": metrics,
            "catalog_coverage": len(descriptors) / len(catalog),
            "query_coverage": sum("error" not in row for row in results) / len(queries),
            "mask_status_counts": dict(statuses), "failures": failures,
            "extraction_ms_p50": float(np.median(extraction_ms)),
            "extraction_ms_p95": float(np.percentile(extraction_ms, 95)), "per_query": results}


def paired_delta(left, right, queries, seed):
    # Cluster by source garment so repeated synthetic backgrounds aren't independent samples.
    groups = {}
    for i, query in enumerate(queries):
        groups.setdefault(tuple(sorted(query["relevant_ids"])), []).append(i)
    differences = np.array([np.mean([right["per_query"][i]["reciprocal_rank"] -
                                    left["per_query"][i]["reciprocal_rank"] for i in indexes])
                            for indexes in groups.values()])
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
    return {"mrr_delta_cluster_mean": float(differences.mean()),
            "bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "clusters": len(groups), "seed": seed, "resamples": 2000}


def validate_records(catalog, queries):
    if not catalog or not queries:
        raise ValueError("Catalog and queries must both be nonempty")
    for records in (catalog, queries):
        ids = [record["id"] for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs")
    for query in queries:
        if not query.get("relevant_ids"):
            raise ValueError("Every query needs independently specified relevant_ids")
        if not set(query["relevant_ids"]).issubset({record["id"] for record in catalog}):
            raise ValueError("Relevant IDs must exist in the catalog manifest (failed images still count)")


def benchmark(catalog, queries, base=None, seed=20261107, top_k=5):
    validate_records(catalog, queries)
    base = base or HSVConfig()
    variants = {}
    for masked in (False, True):
        suffix = "mask" if masked else "roi"
        variants[f"rgb_{suffix}"] = run_variant(catalog, queries, base, masked, rgb=True, top_k=top_k)
        for achromatic in (False, True):
            key = f"hs{'a' if achromatic else ''}_{suffix}"
            config = replace(base, achromatic=achromatic)
            variants[key] = run_variant(catalog, queries, config, masked, top_k=top_k)
    variants["hsa_mask_intersection"] = run_variant(catalog, queries, replace(base, achromatic=True),
                                                   True, metric="intersection", top_k=top_k)
    pairs = [("hs_roi", "hsa_roi"), ("hs_mask", "hsa_mask"),
             ("hs_roi", "hs_mask"), ("hsa_roi", "hsa_mask"), ("rgb_roi", "hsa_roi")]
    return {"top_k": top_k, "catalog_count": len(catalog), "query_count": len(queries),
            "variants": variants, "paired_ablation": {
                f"{a} -> {b}": paired_delta(variants[a], variants[b], queries, seed) for a, b in pairs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["synthetic", "photos", "manifest"], default="synthetic")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    hashes = {}
    if args.suite == "synthetic":
        catalog, queries = synthetic_records()
    elif args.suite == "photos":
        catalog, queries, hashes = photo_records()
    else:
        if not args.manifest:
            parser.error("--manifest required")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        catalog = materialize(manifest["catalog"], args.manifest.parent)
        queries = materialize(manifest["queries"], args.manifest.parent)
        hashes["manifest"] = file_hash(args.manifest)
        for role, records in (("catalog", catalog), ("query", queries)):
            for record in records:
                hashes[f"{role}:{record['id']}"] = record.get("hashes", {"error": record.get("load_error")})
    config = HSVConfig(**json.loads(args.config.read_text())) if args.config else HSVConfig()
    report = benchmark(catalog, queries, config, args.seed, args.top_k)
    report.update(suite=args.suite, input_hashes=hashes,
                  environment={"python": platform.python_version(), "numpy": np.__version__,
                               "opencv": cv2.__version__, "pillow": Image.__version__},
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                  source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in
                      [ROOT / "AI/hsv_descriptor.py", ROOT / "AI/color_utils.py", ROOT / "AI/features.py",
                       ROOT / "AI/color_cache.py", ROOT / "AI/embedding_model.py", Path(__file__).resolve()]},
                  limitations="Color-only diagnostic; no FashionCLIP/M4 integration or independent user-query improvement claim. "
                               "Photo suite uses transformed same-image queries and placement masks; synthetic suite uses known fixture masks.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"# M3 {args.suite} benchmark", "", report["limitations"], "",
             f"Catalog={len(catalog)}, queries={len(queries)}, top-k={args.top_k}, seed={args.seed}.", "",
             "| Variant | Hit@1 | Hit@k | Recall@k | MRR | Catalog/query coverage | p50/p95 extraction ms |",
             "|---|---:|---:|---:|---:|---|---|"]
    for name, run in report["variants"].items():
        m = run["metrics"]
        lines.append(f"| {name} | {m['hit_at_1']:.4f} | {m['hit_at_k']:.4f} | {m['recall_at_k']:.4f} | "
                     f"{m['reciprocal_rank']:.4f} | {run['catalog_coverage']:.3f}/{run['query_coverage']:.3f} | "
                     f"{run['extraction_ms_p50']:.2f}/{run['extraction_ms_p95']:.2f} |")
    lines += ["", "Paired cluster bootstrap MRR differences (95% CI; diagnostic only):", ""]
    for name, delta in report["paired_ablation"].items():
        lines.append(f"- {name}: {delta['mrr_delta_cluster_mean']:.4f}, CI {delta['bootstrap_95_ci']}")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
