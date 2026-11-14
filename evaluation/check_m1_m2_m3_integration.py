"""Audit merged data provenance and real YOLO HTTP -> M3 color transport offline."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import httpx
from fastapi.testclient import TestClient

from AI.features import color_input_from_detection
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor
from evaluation.run_m1_ablation import load_labels
from object_detection.api import app
from object_detection.detector import ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.segmentation import MaskConfig


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_data():
    regression = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))
    seen = {row["sha256"] for row in regression["queries"]}
    split_hashes = {}
    splits = {}
    for split in ("validation", "test"):
        path = ROOT / f"evaluation/m1_{split}.csv"
        labels = load_labels(path)
        split_hashes[split] = {row["sha256"] for row in labels}
        splits[split] = {
            "csv_sha256": sha256(path), "count": len(labels),
            "overlap_m2_development": [row["query_id"] for row in labels if row["sha256"] in seen],
            "not_in_m2_regression": [row["query_id"] for row in labels if row["sha256"] not in seen],
            "mask_reference_count": 0,
            "review_status": "imported_M1_labels_require_M2_review_not_independent_M2_test",
        }
    overlap = sorted(split_hashes["validation"] & split_hashes["test"])
    assert not overlap, "M1 validation/test contain identical image bytes"
    return {"splits": splits, "validation_test_hash_overlap": overlap,
            "limitation": "M1 report discloses AI-assisted initial labels; do not invent human cross-review or masks."}


def real_http_check():
    query = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))["queries"][0]
    image_path = ROOT / query["image"]
    data = image_path.read_bytes()
    detector = ClothingDetector(model_path=str(ROOT / "object_detection/best.pt"))
    model_hash = sha256(Path(detector.model_path))
    model = detector._ensure_model()
    checks = []
    # Real YOLO inference, real ASGI route/serializer, actual M3 decoder/HSV.
    # No FashionCLIP/DB; gamma is a transport probe, not a quality comparison.
    for mode, mask_mode in (("identity", "off"), ("identity", "raw"), ("gamma", "raw")):
        detector.preprocess_config = PreprocessConfig(mode=mode, brightness_threshold=0.30, gamma=0.7)
        detector.mask_config = MaskConfig(mode=mask_mode)
        prepared = prepare_image(data, detector.preprocess_config)
        with patch("object_detection.api.detector", detector), patch.object(
            model, "predict", Mock(wraps=model.predict)
        ) as predict:
            response = TestClient(app).post("/detect/upload?query_id=merge_http", files={
                "file": (image_path.name, data, "image/jpeg")
            })
            assert response.status_code == 200, response.text
            assert predict.call_count == 1
        payload = response.json()
        assert payload["crops"], "Choose a regression image with actual detections"
        records = []
        for crop in payload["crops"]:
            roi, mask, meta = color_input_from_detection(crop)
            x1, y1, x2, y2 = crop["bbox"]
            np.testing.assert_array_equal(roi, prepared.source_rgb[y1:y2, x1:x2])
            descriptor = extract_hsv_descriptor(roi, mask, metadata=meta)
            pixels = int(np.count_nonzero(mask)) if mask is not None else roi.shape[0] * roi.shape[1]
            assert descriptor.valid_pixels == pixels
            assert crop["roi_metadata"]["geometry_mode"] == "m1_prepared"
            assert crop["garment_id"].startswith("merge_http_g")
            records.append({"garment_id": crop["garment_id"], "bbox": crop["bbox"],
                            "roi_shape": list(roi.shape), "source_png_exact": True,
                            "roi_status": crop["roi_metadata"]["roi_status"],
                            "fallback_reason": crop["roi_metadata"]["fallback_reason"],
                            "HSV_mask_status": descriptor.mask_status, "HSV_valid_pixels": pixels})
        checks.append({"preprocess_config": asdict(detector.preprocess_config), "mask_mode": mask_mode,
                       "geometry": prepared.geometry_meta, "gamma_applied": prepared.quality_meta["gamma_applied"],
                       "predict_calls_per_upload": 1, "crops": records})
    with patch("object_detection.api.detector", detector):
        invalid = TestClient(app).post("/detect/upload", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
        assert invalid.status_code == 400
    assert sha256(Path(detector.model_path)) == model_hash
    return {"status": "passed", "query_id": query["query_id"], "image_sha256": sha256(image_path),
            "model_sha256": model_hash, "HSV_version": HSVConfig().version,
            "checks": checks, "invalid_upload_after_model_load": 400,
            "scope": "real_YOLO_ASGI_HTTP_serializer_M3_HSV_no_encoder_DB_or_external_TCP",
            "quality_metrics": "not_measured"}


def tcp_check(api_url):
    query = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))["queries"][0]
    path = ROOT / query["image"]
    prepared = prepare_image(path.read_bytes(), PreprocessConfig())
    with httpx.Client(base_url=api_url, timeout=120, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        response = client.post("/detect/upload", params={"query_id": "tcp_merge"}, files={
            "file": (path.name, path.read_bytes(), "image/jpeg")
        })
        response.raise_for_status()
        payload = response.json()
        assert payload["crops"]
        records = []
        for crop in payload["crops"]:
            roi, mask, meta = color_input_from_detection(crop)
            x1, y1, x2, y2 = crop["bbox"]
            np.testing.assert_array_equal(roi, prepared.source_rgb[y1:y2, x1:x2])
            assert crop["roi_metadata"]["geometry_mode"] == "m1_prepared"
            assert crop["roi_metadata"]["quality_meta"]["mode"] == "identity"
            assert crop["roi_metadata"]["mask_method"] == "off"
            assert mask is None
            descriptor = extract_hsv_descriptor(roi, mask, metadata=meta)
            assert descriptor.valid_pixels == roi.shape[0] * roi.shape[1]
            records.append({"garment_id": crop["garment_id"], "source_png_exact": True,
                            "HSV_valid_pixels": descriptor.valid_pixels})
        invalid = client.post("/detect/upload", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
        assert invalid.status_code == 400
    return {"status": "passed", "health": health.json(), "http_status": response.status_code,
            "default_preprocess_mode": "identity", "default_mask_mode": "off", "crops": records,
            "invalid_upload_after_model_load": invalid.status_code, "scope": "live_TCP_detection_service_to_M3_no_encoder_DB"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/m2/m1_merge_review_20261114")
    parser.add_argument("--api-url", help="Only verify the running TCP service with identity/off defaults")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    reports = (("tcp_http", tcp_check(args.api_url)),) if args.api_url else (
        ("data_audit", audit_data()), ("real_http", real_http_check())
    )
    for name, report in reports:
        (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: passed")
