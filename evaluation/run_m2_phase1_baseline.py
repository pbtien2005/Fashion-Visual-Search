"""Measure the existing detector on old regression images, without retrieval or GT metrics."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from object_detection.detector import CLASS_NAMES, ClothingDetector
from object_detection.utils import crops_to_response


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def draw_preview(image_bytes, crops, query_id):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Draw on the same OpenCV-decoded canvas as the legacy detector (no extra EXIF transform).
    scale = min(304 / image.width, 220 / image.height)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(image)
    for crop in crops:
        bbox = [round(value * scale) for value in crop["bbox"]]
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], max(0, bbox[1] - 12)), f'{crop["class_name"]} {crop["confidence"]:.2f}', fill="red")
    tile = Image.new("RGB", (320, 260), "white")
    tile.paste(image, ((320 - image.width) // 2, 28))
    ImageDraw.Draw(tile).text((8, 8), query_id, fill="black")
    return tile


def run(args):
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("independent_test") or any(row["split"] != "regression" for row in manifest["queries"]):
        raise ValueError("Phase 1 runner accepts regression data only; do not use it to tune test data")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    detector = ClothingDetector(model_path=args.model, conf=0.5, expand_ratio=0.15)
    model_path = Path(detector.model_path)
    before_hash = sha256(model_path)
    started = time.perf_counter()
    model = detector._ensure_model()
    load_ms = (time.perf_counter() - started) * 1000
    if model.task != "detect" or model.names != CLASS_NAMES:
        raise ValueError(f"Unexpected task/classes: {model.task}, {model.names}")
    model.overrides.update(device="cpu", imgsz=640, iou=0.7, max_det=300, half=False)
    started = time.perf_counter()
    detector._process(np.zeros((640, 640, 3), dtype=np.uint8))
    warmup_ms = (time.perf_counter() - started) * 1000
    rows, tiles = [], []
    counts = Counter()
    for query in manifest["queries"]:
        row = {"query_id": query["query_id"], "image": query["image"], "error": None}
        image_path = ROOT / query["image"]
        try:
            if sha256(image_path) != query["sha256"]:
                raise ValueError("Image hash differs from locked regression manifest")
            image_bytes = image_path.read_bytes()
            started = time.perf_counter()
            crops = detector.detect_from_bytes(image_bytes)
            detect_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            response = crops_to_response(crops).model_dump()
            payload_bytes = len(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            serialize_ms = (time.perf_counter() - started) * 1000
            row.update(
                detection_crop_ms=detect_ms, serialization_ms=serialize_ms,
                total_m2_ms=detect_ms + serialize_ms, payload_bytes=payload_bytes,
                delivered_crop_count=len(crops),
                crops=[{
                    "class_id": crop["class_id"], "class_name": crop["class_name"],
                    "confidence": crop["confidence"], "bbox": crop["bbox"],
                    "expanded_bbox": crop["expanded_bbox"], "variant_count": len(crop["variants"]),
                } for crop in crops],
            )
            counts.update(crop["class_name"] for crop in crops)
            tiles.append(draw_preview(image_bytes, crops, query["query_id"]))
            print(f'{query["query_id"]}: {len(crops)} crops, {detect_ms + serialize_ms:.1f} ms', flush=True)
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", delivered_crop_count=0, crops=[])
            print(f'{query["query_id"]}: ERROR {row["error"]}', flush=True)
        rows.append(row)

    successful = [row for row in rows if row["error"] is None]
    latencies = [row["total_m2_ms"] for row in successful]
    with_crops = sum(row["delivered_crop_count"] > 0 for row in rows)
    predictor = model.predictor.args
    checkpoint = model.ckpt
    report = {
        "run_name": output.name, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "regression_not_independent_test", "schema_version": "m2.baseline.v1",
        "coverage_instrumented": sys.gettrace() is not None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            if (ROOT / ".git").exists() and shutil.which("git") else "recorded_in_M2_ENVIRONMENT.md",
        "model_sha256": before_hash, "model_filename": model_path.name,
        "model_unchanged": before_hash == sha256(model_path),
        "manifest_sha256": sha256(manifest_path), "model_task": model.task, "class_names": model.names,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "packages": {name: importlib.metadata.version(name) for name in [
                "ultralytics", "torchvision", "opencv-python", "opencv-python-headless", "numpy", "pillow", "fastapi", "pydantic",
            ]},
        },
        "config": {key: getattr(predictor, key) for key in ["conf", "iou", "imgsz", "device", "half", "rect", "max_det", "agnostic_nms"]},
        "crop_config": {"expand_ratio": detector.expand_ratio, "wide_expand_ratio": detector.wide_expand_ratio, "output_size": 224, "minimum_variant_side": 50},
        "timing": {"model_load_ms": load_ms, "blank_warmup_ms": warmup_ms, "warmup_calls": 1,
            "mean_m2_ms": statistics.mean(latencies) if latencies else None,
            "median_m2_ms": statistics.median(latencies) if latencies else None,
            "p95_m2_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "scope": "decode + YOLO + legacy crop + serialization; excludes HTTP, M1, embedding, DB and retrieval"},
        "summary": {"total_images": len(rows), "errors": len(rows) - len(successful), "images_with_delivered_crops": with_crops,
            "delivered_crop_image_rate": with_crops / len(rows) if rows else None,
            "delivered_crops_by_class": dict(counts), "total_delivered_crops": sum(counts.values())},
        "unmeasured": ["target_detection_recall", "bbox_mAP", "mask_IoU", "color_accuracy", "retrieval_metrics"],
        "limitations": "No human target boxes/masks; delivered crops are predictions, not correct detections. Old regression data, small sample. CPU Docker latency is not GPU or end-to-end latency.",
        "checkpoint_training_metadata_not_remeasured": {
            "ultralytics_version": checkpoint.get("version"), "date": checkpoint.get("date"),
            "train_metrics": checkpoint.get("train_metrics"),
            "train_args": {key: checkpoint.get("train_args", {}).get(key) for key in ["epochs", "imgsz", "batch", "optimizer", "lr0", "seed", "close_mosaic"]},
            "history_rows": len(checkpoint.get("train_results", {}).get("epoch", [])),
        },
        "queries": rows,
    }
    write_json(output / "baseline.json", report)
    csv_fields = ["query_id", "delivered_crop_count", "detection_crop_ms", "serialization_ms", "total_m2_ms", "payload_bytes", "error"]
    with (output / "baseline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if tiles:
        sheet = Image.new("RGB", (1280, ((len(tiles) + 3) // 4) * 260), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 4) * 320, (index // 4) * 260))
        sheet.save(output / "detections.jpg", quality=90)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 1 if report["summary"]["errors"] or not report["model_unchanged"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--manifest", default="evaluation/m2/regression_manifest.json")
    parser.add_argument("--output", required=True, help="New output directory, relative to repository root")
    raise SystemExit(run(parser.parse_args()))
