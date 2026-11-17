"""Frozen-ROI M2 ablation using existing M3 HSV and reviewed references."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from AI.hsv_descriptor import HSVConfig, histogram_similarity
from AI.production_color import ColorStore, descriptor_display, extract_source_color
from evaluation.m2_annotations import ROOT, file_hash, read_json, reference_for, workspace_path
from evaluation.m2_metrics import mask_iou, place_roi_mask, summarize_rows
from evaluation.run_color_benchmark import extract as benchmark_extract
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.segmentation import MaskConfig, segment_roi
from object_detection.utils import binary_mask_to_base64_png, rgb_to_base64_png


MODES = ("off", "raw", "open_close")
CRITERIA = {"min_validation_references": 2, "min_each_reference_mask_iou": .80,
            "min_mean_color_similarity_gain": .02, "max_each_color_similarity_loss": .01,
            "require_all_reference_masks_delivered": True}
SOURCE_FILES = ("object_detection/detector.py", "object_detection/image_preprocessing.py",
                "object_detection/segmentation.py", "object_detection/utils.py", "AI/features.py",
                "AI/hsv_descriptor.py", "AI/production_color.py", "AI/color_cache.py",
                "evaluation/run_color_benchmark.py", "evaluation/m2_phase5.py", "evaluation/run_m2_phase5.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def input_identity(manifest_path, report_path):
    return {"manifest_sha256": file_hash(Path(manifest_path)), "phase4_report_sha256": file_hash(Path(report_path)),
            "model_sha256": file_hash(ROOT / "object_detection/best.pt"),
            "HSV_profile_sha256": file_hash(ROOT / "docs/m3_contract/hsv147.json"),
            "HSV_version": HSVConfig().version, "selection_criteria": CRITERIA,
            "source_sha256": {path: file_hash(ROOT / path) for path in SOURCE_FILES}}


def verify_lock(lock, fingerprint):
    if lock.get("input_fingerprint") != fingerprint or lock.get("mask_modes") != list(MODES):
        raise ValueError("Locked input/config/profile changed; do not reuse this experiment")
    if "lock_sha256" in lock:
        body = {k: v for k, v in lock.items() if k != "lock_sha256"}
        if digest(body) != lock["lock_sha256"]:
            raise ValueError("Lock content changed")


def assess_validation(rows):
    """Use validation references only, including failed masks in every check."""
    rows = [r for r in rows if r["split"] == "validation" and r["label_status"] == "human_reviewed"
            and r["mask_label_status"] == "human_reviewed"]
    baseline = {r["query_id"]: r for r in rows if r["mask_mode"] == "off"}
    candidates = {}
    for mode in MODES[1:]:
        selected = {r["query_id"]: r for r in rows if r["mask_mode"] == mode}
        reasons = []
        if len(baseline) < CRITERIA["min_validation_references"] or set(selected) != set(baseline):
            reasons.append("insufficient_or_unpaired_validation_references")
        ordered = [selected[key] for key in sorted(set(selected) & set(baseline))]
        if not ordered or not all(r["matched"] and r["mask_delivered"] for r in ordered):
            reasons.append("not_all_reference_masks_delivered")
        ious = [r.get("mask_iou_end_to_end") or 0.0 for r in ordered]
        if not ious or min(ious) < CRITERIA["min_each_reference_mask_iou"]:
            reasons.append("reference_mask_iou_below_engineering_guard")
        gains = []
        for row in ordered:
            left = baseline[row["query_id"]].get("color_reference_similarity")
            right = row.get("color_reference_similarity")
            if left is None or right is None or not all(math.isfinite(x) and 0 <= x <= 1 for x in (left, right)):
                reasons.append("color_measurement_missing_or_invalid")
            else:
                gains.append(right - left)
        mean_gain = float(np.mean(gains)) if gains else None
        if mean_gain is None or mean_gain < CRITERIA["min_mean_color_similarity_gain"]:
            reasons.append("color_gain_below_engineering_guard")
        if gains and min(gains) < -CRITERIA["max_each_color_similarity_loss"]:
            reasons.append("per_reference_color_regression")
        candidates[mode] = {"eligible": not reasons, "reasons": reasons,
                            "mean_color_similarity_gain": mean_gain,
                            "mean_mask_iou": float(np.mean(ious)) if ious else None}
    eligible = [mode for mode in MODES[1:] if candidates[mode]["eligible"]]
    selected = max(eligible, key=lambda mode: (candidates[mode]["mean_color_similarity_gain"],
                                              candidates[mode]["mean_mask_iou"], mode == "raw")) if eligible else "off"
    return {"deployment_mode": selected, "validation_reference_count": len(baseline),
            "criteria": CRITERIA, "candidates": candidates,
            "policy_scope": "Engineering guards for default activation; not a course requirement or statistical significance test."}


def safe_hsv_env():
    """Read only color/embedding configuration keys; never expose unrelated env secrets."""
    allowed = {"HSV_COLOR_CACHE_FILE", "HSV_COLOR_CACHE_PATH", "HSV_CONFIG_PATH", "HSV_MISSING_POLICY",
               "EMBEDDING_MODEL", "EMBEDDING_VERSION"}
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, raw = line.partition("=")
            key = key.strip()
            if separator and key in allowed:
                values[key] = raw.strip().strip("\"'")
    for key in allowed:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def audit_cache(env_values=None):
    values = safe_hsv_env() if env_values is None else env_values
    # Compose mounts the host FILE at a fixed container PATH. They are not interchangeable.
    raw_path = values.get("HSV_COLOR_CACHE_FILE") or "data/color_cache_hsv147.json"
    path = Path(raw_path)
    path = path if path.is_absolute() else ROOT / path
    production = {"host_source": raw_path, "compose_target": "/app/color-cache/catalog.json",
                  "status": "missing", "candidate_id_coverage": "not_verified_no_live_catalog_snapshot"}
    if os.name != "nt" and len(raw_path) > 2 and raw_path[1] == ":":
        production["status"] = "windows_host_path_requires_host_verification"
    if path.is_file():
        try:
            store = ColorStore(read_json(path), HSVConfig())
            model_name = values.get("EMBEDDING_MODEL", "patrickjohncyh/fashion-clip")
            version = int(values.get("EMBEDDING_VERSION", "2"))
            for pid in store.index:
                store.get(pid, model_name, version)
            production.update(status="schema_and_entries_valid", entries=len(store.index), sha256=file_hash(path))
        except (ValueError, KeyError, TypeError) as exc:
            production.update(status="invalid", reason=str(exc))
    sample_path = ROOT / "docs/m3_results/catalog_color_cache.json"
    sample = read_json(sample_path) if sample_path.is_file() else None
    sample_info = {"status": "not_present"}
    if sample is not None:
        sample_info = {"entries": len(sample.get("entries", {})), "sha256": file_hash(sample_path),
                       "roi_policies": sorted({e["descriptor"]["roi_policy"] for e in sample["entries"].values()}),
                       "status": "diagnostic_sample_only"}
    return {"production_cache": production, "diagnostic_cache": sample_info,
            "sample_cache_is_production": False, "product_retrieval": "not_measured",
            "reason": "No verified live candidate/catalog/relevance snapshot; never substitute the 28-photo diagnostic cache.",
            "missing_color_policy": values.get("HSV_MISSING_POLICY", "zero"),
            "reference_color_metric": "Real HSV147 Hellinger similarity to reviewed clothing pixels, not product retrieval."}


def frozen_color_rows(manifest, phase4_report, split, *, root=ROOT):
    """Reuse genuine phase4 detections; never rerun YOLO or fabricate missing targets."""
    index = {(r["query_id"], r["mask_mode"]): r for r in phase4_report["rows"]}
    config = HSVConfig()
    output = []
    for query in manifest["queries"]:
        if query["split"] != split:
            continue
        prepared = prepare_image(workspace_path(query["image"], root).read_bytes(), PreprocessConfig(**manifest["preprocess_config"]))
        source = prepared.source_rgb
        reference = reference_for(query, prepared.geometry_meta, root)
        reference_color = benchmark_extract({"roi": source, "mask_array": reference,
                                             "roi_source": "reviewed_clothing_reference"}, config, True, False)[0] if reference is not None else None
        control = index[query["query_id"], "off"]
        for mode in MODES:
            row = deepcopy(index[query["query_id"], mode])
            if (row["predicted_bbox"], row["garment_id"], row["matched"]) != (
                control["predicted_bbox"], control["garment_id"], control["matched"]
            ):
                raise ValueError("Ablation variants do not share the same detections")
            row.update(color_reference_similarity=0.0 if reference is not None else None,
                       color_conditional_similarity=None, color_ms=None)
            if row["matched"]:
                x1, y1, x2, y2 = row["predicted_bbox"]
                roi = source[y1:y2, x1:x2]
                mode_config = MaskConfig(**phase4_report["mask_configs"][mode])
                segmentation = segment_roi(roi, mode_config)
                # Exclude nondeterministic latency fields when checking cached segmentation metadata.
                for key in ("roi_status", "fallback_reason", "mask_config_id"):
                    if row["roi_metadata"][key] != segmentation.metadata[key]:
                        raise ValueError("Segmentation differs from the frozen phase4 run")
                wire = {"source_crop_png_base64": rgb_to_base64_png(roi),
                        "mask_png_base64": binary_mask_to_base64_png(segmentation.mask) if segmentation.mask is not None else None,
                        "query_id": row["query_id"], "garment_id": row["garment_id"],
                        "roi_metadata": dict(row["roi_metadata"], **segmentation.metadata)}
                start = time.perf_counter()
                color = extract_source_color(detection=wire, item_id=row["garment_id"])
                row["color_ms"] = (time.perf_counter() - start) * 1000
                if color is None or color.version != config.version:
                    raise ValueError("Runtime M3 descriptor missing or profile changed")
                expected = int(np.count_nonzero(segmentation.mask)) if segmentation.mask is not None else roi.shape[0] * roi.shape[1]
                if color.valid_pixels != expected:
                    raise ValueError("M3 pixel count does not match the chosen mask")
                row.update(HSV_valid_pixels=color.valid_pixels, HSV_mask_status=color.mask_status,
                           HSV_version=color.version, roi_source=color.roi_source,
                           display_color=descriptor_display(color)[1], source_hashes=color.metadata["source_hashes"])
                if reference is not None:
                    similarity = histogram_similarity(color, reference_color, "hellinger")
                    row.update(color_reference_similarity=similarity, color_conditional_similarity=similarity)
                    reproduced = mask_iou(place_roi_mask(segmentation.mask, row["predicted_bbox"],
                                                        (source.shape[1], source.shape[0])), reference) if segmentation.mask is not None else 0.0
                    if not math.isclose(reproduced, row["mask_iou_end_to_end"], abs_tol=1e-12):
                        raise ValueError("Frozen reference mask IoU no longer reproduces")
            output.append(row)
    return output


def summarize_color(rows):
    result = {}
    for mode in MODES:
        group = [r for r in rows if r["mask_mode"] == mode]
        metrics = summarize_rows(group)["reviewed"]
        for key in ("color_reference_similarity", "color_conditional_similarity"):
            values = [r[key] for r in group if r.get(key) is not None]
            metrics[key + "_n"] = len(values)
            metrics[key + "_mean"] = float(np.mean(values)) if values else None
        values = [r["color_ms"] for r in group if r.get("color_ms") is not None]
        metrics["M3_color_ms_mean"] = float(np.mean(values)) if values else None
        result[mode] = metrics
    return result
