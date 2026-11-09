"""Verify source ROI transport and compare legacy JPEGs against the frozen phase 1 image."""

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np

from object_detection.detector import ClothingDetector
from object_detection.utils import crops_to_response


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_source_payload(crop, source_rgb):
    meta = crop["roi_metadata"]
    x1, y1, x2, y2 = meta["bbox_xyxy"]
    raw = base64.b64decode(crop["source_crop_png_base64"], validate=True)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb, source_rgb[y1:y2, x1:x2]), "Source ROI pixels changed during transport"
    assert rgb.shape[:2] == (meta["roi_height"], meta["roi_width"])
    assert crop["mask_png_base64"] is None
    assert meta["roi_status"] == "unmasked" and meta["fallback_reason"] == "mask_disabled"


def prepare_response_fixture():
    fixtures = ROOT / "tests/fixtures/m2"
    source = cv2.cvtColor(cv2.imread(str(fixtures / "source_rgb.png")), cv2.COLOR_BGR2RGB)
    detection = cv2.cvtColor(cv2.imread(str(fixtures / "detection_rgb.png")), cv2.COLOR_BGR2RGB)
    box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[32, 16, 96, 80]])
    detector = ClothingDetector()
    detector._model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[box])])
    batch = detector.detect_prepared(
        source, detection, query_id="phase2_fixture",
        geometry_meta={"mode": "m1_fixture", "working_width": 128, "working_height": 96},
        quality_meta={"preprocessing_mode": "synthetic_plus40_not_M1_gamma"},
    )
    payload = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics).model_dump()
    assert_source_payload(payload["crops"][0], source)
    write_json(fixtures / "phase2_response.json", payload)
    return payload


def load_frozen_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regression_check(output):
    report_path = output / "regression.json"
    if report_path.exists():
        raise FileExistsError("Keep the previous report; choose a new --output directory")
    # /app is the code baked into fashion-m2:phase1, while /workspace is the new code.
    frozen = Path("/app/detector.py")
    if not frozen.is_file() or frozen.read_bytes() == (ROOT / "object_detection/detector.py").read_bytes():
        raise RuntimeError("Run regression with the unchanged fashion-m2:phase1 image and bind the repo to /workspace")
    sys.path.append("/app")
    old_detector_module = load_frozen_module("phase1_frozen_detector", frozen)
    old_utils = load_frozen_module("phase1_frozen_utils", Path("/app/utils.py"))
    manifest_path = ROOT / "evaluation/m2/regression_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = ClothingDetector(model_path=str(ROOT / "object_detection/best.pt"))
    model_hash = hashlib.sha256(Path(current.model_path).read_bytes()).hexdigest()
    model = current._ensure_model()
    model.overrides.update(device="cpu", imgsz=640, iou=0.7, max_det=300, half=False)
    old = old_detector_module.ClothingDetector(model_path=current.model_path)
    old._model = model
    rows = []
    for query in manifest["queries"]:
        image_bytes = (ROOT / query["image"]).read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == query["sha256"]
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        previous_crops = old._process(bgr.copy())
        batch = current.detect_batch_from_bytes(image_bytes, query_id=query["query_id"])
        assert len(previous_crops) == len(batch.crops), "Delivered crop count changed"
        buckets = defaultdict(list)
        for crop in previous_crops:
            buckets[(crop["class_id"], tuple(crop["original_bbox"]), crop["confidence"])].append(crop)
        payload = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics).model_dump()
        checked_variants, outward_bboxes = 0, 0
        for crop, wire_crop in zip(batch.crops, payload["crops"]):
            key = (crop["class_id"], tuple(crop["original_bbox"]), crop["confidence"])
            previous = buckets[key].pop()
            for field in ["image", "expanded_bbox", "padded_bbox", "crop_width", "crop_height",
                          "source_crop_width", "source_crop_height", "crop_strategy"]:
                if field == "image":
                    assert np.array_equal(previous[field], crop[field]), "Legacy primary JPEG input changed"
                else:
                    assert previous[field] == crop[field], f"Legacy metadata changed: {field}"
            assert len(previous["variants"]) == len(crop["variants"])
            for before, after, wire_variant in zip(previous["variants"], crop["variants"], wire_crop["crop_variants"]):
                assert before["name"] == after["name"] and before["bbox"] == after["bbox"]
                assert np.array_equal(before["image"], after["image"]), "Legacy variant pixels changed"
                assert old_utils.ndarray_to_base64_jpeg(before["image"]) == wire_variant["image_base64"]
                checked_variants += 1
            assert_source_payload(wire_crop, source)
            outward_bboxes += crop["bbox"] != previous["bbox"]
        assert all(not pending for pending in buckets.values())
        old_bytes = len(old_utils.crops_to_response(previous_crops).model_dump_json().encode("utf-8"))
        new_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        rows.append({
            "query_id": query["query_id"], "delivered_crops": len(batch.crops),
            "legacy_variants_pixel_and_jpeg_equal": checked_variants,
            "source_rois_exact": len(batch.crops), "bbox_outward_rounding_changes": outward_bboxes,
            "legacy_payload_bytes": old_bytes, "phase2_payload_bytes": new_bytes,
            "diagnostics": batch.diagnostics,
        })
        print(f'{query["query_id"]}: {len(batch.crops)} crops; legacy JPEGs and source PNGs verified', flush=True)
    report = {
        "status": "passed", "data_role": "old_regression_no_target_ground_truth", "model_sha256": model_hash,
        "model_unchanged": model_hash == hashlib.sha256(Path(current.model_path).read_bytes()).hexdigest(),
        "reference": "frozen /app code from fashion-m2:phase1 image; same model, conf=0.5/imgsz=640/iou=0.7/CPU",
        "images": len(rows), "total_crops": sum(row["delivered_crops"] for row in rows),
        "legacy_variants_pixel_and_jpeg_equal": sum(row["legacy_variants_pixel_and_jpeg_equal"] for row in rows),
        "source_rois_exact": sum(row["source_rois_exact"] for row in rows),
        "bbox_outward_rounding_changes": sum(row["bbox_outward_rounding_changes"] for row in rows),
        "legacy_payload_bytes": sum(row["legacy_payload_bytes"] for row in rows),
        "phase2_payload_bytes": sum(row["phase2_payload_bytes"] for row in rows),
        "limitations": "No recall/mAP/IoU/retrieval metrics; no M1 production module or M3 decoder involved. PNG payload cost is measured, not assumed negligible.",
        "queries": rows,
    }
    write_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}), flush=True)


def http_check(api_url, output):
    manifest = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))
    query = manifest["queries"][0]
    image_bytes = (ROOT / query["image"]).read_bytes()
    source = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with httpx.Client(base_url=api_url.rstrip("/"), timeout=120, trust_env=False) as client:
        response = client.post("/detect/upload", params={"query_id": "phase2_http"}, files={"file": ("outfit.jpg", image_bytes, "image/jpeg")})
        response.raise_for_status()
        payload = response.json()
        assert payload["schema_version"] == "m2.roi.v1" and payload["query_id"] == "phase2_http"
        old_schemas = load_frozen_module("phase1_frozen_http_schemas", Path("/app/schemas.py"))
        assert "source_crop_png_base64" not in old_schemas.CropResult.model_fields, "Use the frozen phase 1 image for this compatibility probe"
        legacy_view = old_schemas.DetectionResponse.model_validate(payload)
        assert legacy_view.total_crops == payload["total_crops"]
        assert legacy_view.crops[0].image_base64 == payload["crops"][0]["image_base64"]
        for index, crop in enumerate(payload["crops"]):
            assert crop["garment_id"] == f"phase2_http_g{index:03d}"
            assert_source_payload(crop, source)
        assert payload["crops"]
        bad_image = client.post("/detect/upload", files={"file": ("bad.jpg", b"bad", "image/jpeg")})
        bad_id = client.post("/detect/upload", params={"query_id": "../escape"}, files={"file": ("outfit.jpg", image_bytes, "image/jpeg")})
        assert bad_image.status_code == 400 and bad_id.status_code == 422
        health = client.get("/health")
        health.raise_for_status()
    write_json(output / "http_smoke.json", {
        "status": "passed", "mode": "real_HTTP_real_YOLO_CPU", "health": health.json(),
        "upload_status": response.status_code, "schema_version": payload["schema_version"],
        "query_id": payload["query_id"], "total_crops": payload["total_crops"],
        "exact_source_png_round_trips": len(payload["crops"]), "mask": "null_mask_disabled",
        "invalid_image_status": bad_image.status_code, "invalid_id_status": bad_id.status_code,
        "frozen_legacy_schema_accepts_new_response": True,
        "diagnostics": payload["diagnostics"],
    })
    print(f'HTTP passed: {len(payload["crops"])} source PNGs exact; mask disabled.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fixture", "regression", "http"])
    parser.add_argument("--output", default="docs/m2/phase2")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.mode == "fixture":
        prepare_response_fixture()
        print("Created tests/fixtures/m2/phase2_response.json")
    elif args.mode == "regression":
        regression_check(ROOT / args.output)
    else:
        http_check(args.api_url, ROOT / args.output)
