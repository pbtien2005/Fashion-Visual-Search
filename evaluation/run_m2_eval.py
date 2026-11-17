"""Offline target/mask evaluation with M1 identity and actual M3 HSV transport.

This runner never promotes imported/unreviewed labels to verified ground truth.
Segmentation modes share one detection pass. It does not tune configs or rank products.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from AI.features import color_input_from_detection
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor
from AI.production_color import descriptor_display
from evaluation.m2_annotations import ROOT, file_hash, load_manifest, reference_for, workspace_path, write_json
from evaluation.m2_metrics import bbox_to_working, mask_iou, place_roi_mask, score_target, summarize_rows
from object_detection.detector import ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.roi import clip_bbox
from object_detection.segmentation import MaskConfig, segment_roi
from object_detection.utils import crops_to_response


def evaluate(manifest, detector, *, root=ROOT, modes=("off", "raw", "open_close"), output=None, warmup=True):
    config = PreprocessConfig(**manifest["preprocess_config"])
    detector.mask_config = MaskConfig(mode="off")
    configs = {mode: replace(detector.mask_config, mode=mode) for mode in modes}
    rows, timings = [], []
    warmup_ms = None
    if warmup:
        start = time.perf_counter()
        detector._ensure_model().predict(np.zeros((64, 64, 3), np.uint8), conf=detector.conf, verbose=False)
        warmup_ms = (time.perf_counter() - start) * 1000
    for query in manifest["queries"]:
        bases = {mode: {"query_id": query["query_id"], "target_id": query["target_id"], "split": query["split"],
                        "label_status": query["label_status"], "mask_label_status": query["mask_status"],
                        "mask_mode": mode, "mask_config_id": configs[mode].config_id,
                        "class_name": query["class_name"], "image_sha256": query["sha256"],
                        "source": query["source_url"], "matched": False, "mask_delivered": False,
                        "mask_iou_conditional": None,
                        "gt_bbox_mask_iou": 0.0 if query["mask_status"] == "human_reviewed" else None,
                        "mask_iou_end_to_end": 0.0 if query["mask_status"] == "human_reviewed" else None}
                 for mode in modes}
        try:
            start = time.perf_counter()
            prepared = prepare_image(workspace_path(query["image"], root).read_bytes(), config)
            prepare_ms = (time.perf_counter() - start) * 1000
            geometry = prepared.geometry_meta
            width, height = geometry["width"], geometry["height"]
            target = {"class_name": query["class_name"],
                      "bbox": bbox_to_working(query["bbox_normalized"], width, height)}
            reference = reference_for(query, geometry, root)
            start = time.perf_counter()
            batch = detector.detect_prepared(
                prepared.source_rgb, prepared.detection_rgb, query_id=query["query_id"],
                geometry_meta=dict(geometry, mode="m1_prepared", schema_version=prepared.schema_version,
                                   working_width=width, working_height=height,
                                   scale_x=width / geometry["oriented_width"], scale_y=height / geometry["oriented_height"]),
                quality_meta=prepared.quality_meta,
            )
            detect_ms = (time.perf_counter() - start) * 1000
            for mode in modes:
                row = bases[mode]
                try:
                    mode_crops = []
                    for crop in batch.crops:
                        segmentation = segment_roi(crop["source_rgb"], configs[mode])
                        item = dict(crop, mask=segmentation.mask, roi_metadata=dict(crop["roi_metadata"], **segmentation.metadata))
                        mode_crops.append(item)
                    start = time.perf_counter()
                    response = crops_to_response(mode_crops, query_id=batch.query_id, diagnostics=batch.diagnostics)
                    payload = response.model_dump()
                    payload_bytes = len(response.model_dump_json().encode("utf-8"))
                    serialize_ms = (time.perf_counter() - start) * 1000
                    row.update(score_target(target, mode_crops, (width, height), reference))
                    row.update(target_bbox=target["bbox"], working_size=[width, height], geometry_meta=geometry,
                               quality_meta=prepared.quality_meta, detection_count=len(mode_crops), diagnostics=batch.diagnostics,
                               prepare_ms=prepare_ms, detect_crop_off_ms=detect_ms, serialize_ms=serialize_ms,
                               payload_bytes=payload_bytes,
                               segmentation_ms=sum(c["roi_metadata"]["segmentation"].get("latency_ms", 0) for c in mode_crops))
                    if row["matched"]:
                        item = mode_crops[row["prediction_index"]]
                        wire = payload["crops"][row["prediction_index"]]
                        roi, mask, metadata = color_input_from_detection(wire)
                        np.testing.assert_array_equal(roi, item["source_rgb"])
                        descriptor = extract_hsv_descriptor(roi, mask, metadata=metadata)
                        expected_pixels = int(np.count_nonzero(mask)) if mask is not None else roi.shape[0] * roi.shape[1]
                        if descriptor.valid_pixels != expected_pixels:
                            raise ValueError("M3 did not consume the selected source pixels")
                        row.update(HSV_mask_status=descriptor.mask_status, HSV_valid_pixels=descriptor.valid_pixels,
                                   HSV_version=descriptor.version, roi_source=descriptor.roi_source,
                                   display_color=descriptor_display(descriptor)[1],
                                   roi_metadata=item["roi_metadata"])
                    if reference is not None:
                        gt_bbox = clip_bbox(target["bbox"], width, height)
                        x1, y1, x2, y2 = gt_bbox
                        oracle = segment_roi(prepared.source_rgb[y1:y2, x1:x2], configs[mode])
                        row["gt_bbox_mask_iou"] = mask_iou(place_roi_mask(oracle.mask, gt_bbox, (width, height)), reference) if oracle.mask is not None else 0.0
                    if output is not None and mode == modes[0]:
                        save_overlay(output, query, prepared.source_rgb, target, mode_crops)
                except Exception as exc:
                    # Keep target/reference denominator. A transport/error row cannot count as a success.
                    row.update(matched=False, mask_delivered=False, mask_iou_conditional=None,
                               mask_iou_end_to_end=0.0 if reference is not None else None,
                               error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
            timings.append({"query_id": query["query_id"], "prepare_ms": prepare_ms, "detect_crop_off_ms": detect_ms})
        except Exception as exc:
            for row in bases.values():
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    return {
        "schema_version": "m2.eval.v1", "dataset_kind": manifest["dataset_kind"],
        "independent_test": manifest.get("independent_test", False), "preprocess_config": asdict(config),
        "mask_configs_provisional": True, "mask_configs": {mode: asdict(cfg) for mode, cfg in configs.items()},
        "HSV_version": HSVConfig().version, "confidence": detector.conf, "matching_iou_threshold": .5,
        "warmup_ms": warmup_ms, "timings_once_per_image": timings,
        "scope": "offline_M1_YOLO_M2_transport_M3_HSV; no_encoder_DB_retrieval_or_config_selection",
        "summary": {mode: summarize_rows([row for row in rows if row["mask_mode"] == mode]) for mode in modes},
        "rows": rows,
    }


def save_overlay(output, query, source, target, predictions):
    image = Image.fromarray(source)
    draw = ImageDraw.Draw(image)
    draw.rectangle(target["bbox"], outline="#00ff60", width=3)
    for prediction in predictions:
        draw.rectangle(prediction["bbox"], outline="#ff9900", width=2)
    image.thumbnail((600, 600))
    folder = Path(output) / "overlays"
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder / (query["query_id"] + ".jpg"), quality=85)


def save_report(report, output):
    write_json(output / "results.json", report)
    fields = sorted({key for row in report["rows"] for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "targets.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/m2/annotations.m1.pending.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="object_detection/best.pt")
    parser.add_argument("--modes", nargs="+", choices=("off", "raw", "opening", "closing", "open_close"), default=["off", "raw", "open_close"])
    parser.add_argument("--split", choices=("all", "validation", "test", "regression"), default="all")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new directory to preserve results")
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    if args.split != "all":
        manifest["queries"] = [row for row in manifest["queries"] if row["split"] == args.split]
    if not manifest["queries"]:
        raise SystemExit("No targets in selected split")
    output.mkdir(parents=True)
    model_path = Path(args.model)
    model_hash = file_hash(model_path)
    detector = ClothingDetector(model_path=str(model_path.resolve()), mask_config=MaskConfig(mode="off"))
    report = evaluate(manifest, detector, modes=tuple(dict.fromkeys(args.modes)), output=output)
    if file_hash(model_path) != model_hash:
        raise RuntimeError("Model weights changed during evaluation")
    report.update(manifest_sha256=file_hash(manifest_path), model_sha256=model_hash,
                  base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if (ROOT / ".git").exists() and __import__("shutil").which("git") else "see_source_snapshot",
                  platform=platform.platform(), split=args.split)
    save_report(report, output)
    errors = sum(bool(row.get("error")) for row in report["rows"])
    print(json.dumps({"targets": len(manifest["queries"]), "mode_rows": len(report["rows"]), "errors": errors,
                      "reviewed_targets": sum(row["label_status"] == "human_reviewed" for row in manifest["queries"]),
                      "output": str(output)}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
