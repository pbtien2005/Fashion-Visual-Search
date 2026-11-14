"""Phase 3 synthetic QA and old-image mask smoke checks, not a segmentation benchmark."""

import argparse
import base64
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw

from AI.hsv_descriptor import HSVConfig
from AI.production_color import extract_source_color
from object_detection.detector import ClothingDetector
from object_detection.segmentation import MaskConfig, segment_roi
from object_detection.utils import binary_mask_to_base64_png, crops_to_response


ROOT = Path(__file__).resolve().parents[1]
MODES = ("raw", "opening", "closing", "open_close")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def overlay(roi, mask):
    image = roi.copy()
    if mask is not None:
        selected = mask != 0
        image[selected] = (image[selected].astype(float) * .55 + np.array([0, 255, 0]) * .45).astype(np.uint8)
    return Image.fromarray(image)


def sheet(rows, headers, path):
    width, height = 190, 180
    canvas = Image.new("RGB", (len(headers) * width, 30 + len(rows) * height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(headers):
        draw.text((i * width + 7, 8), text, fill="black")
    for j, (label, images) in enumerate(rows):
        top = 30 + j * height
        for i, image in enumerate(images):
            draw.text((i * width + 6, top + 3), label if i == 0 else "", fill="black")
            if image is None:
                draw.text((i * width + 15, top + 70), "None / fallback", fill="gray")
                continue
            preview = image.copy()
            preview.thumbnail((174, 140))
            canvas.paste(preview, (i * width + 7, top + 30))
            draw.rectangle((i * width + 6, top + 29, i * width + 7 + preview.width, top + 30 + preview.height), outline="gray")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def mask_image(mask):
    return Image.fromarray(mask).convert("RGB") if mask is not None else None


def synthetic_check(output):
    gt = np.zeros((80, 80), np.uint8)
    gt[16:64, 16:64] = 255
    bright = np.full((80, 80, 3), 30, np.uint8)
    bright[gt != 0] = 210
    dark = 240 - bright
    noisy = bright.copy()
    noisy[5, 5] = 210
    noisy[30, 30] = 30
    ambiguous = bright.copy()
    ambiguous[:, :40], ambiguous[:, 40:] = 30, 210
    cases = [("bright", bright, gt), ("dark", dark, gt), ("noise_and_hole", noisy, gt),
             ("white_on_white", np.full_like(bright, 255), gt), ("ambiguous", ambiguous, gt)]
    report, rows = [], []
    for name, roi, reference in cases:
        for mode in MODES:
            result = segment_roi(roi, MaskConfig(mode=mode))
            union = np.count_nonzero((result.mask != 0) | (reference != 0)) if result.mask is not None else 0
            iou = float(np.count_nonzero((result.mask != 0) & (reference != 0)) / union) if union else 0.0
            report.append({"case": name, "mode": mode, "synthetic_iou_failure_zero": iou, **result.metadata})
            if mode == "open_close":
                rows.append((name, [Image.fromarray(roi), mask_image(result.raw_mask), mask_image(result.processed_mask),
                                     mask_image(reference), overlay(roi, result.mask)]))
                for suffix, mask in [("raw", result.raw_mask), ("processed", result.processed_mask), ("reference", reference)]:
                    if mask is not None:
                        target = output / "synthetic_masks" / f"{name}_{suffix}.png"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(mask).save(target)
    sheet(rows, ["ROI (synthetic)", "raw Otsu", "after open+close", "known GT", "delivered overlay"], output / "synthetic_contact.png")
    write_json(output / "synthetic.json", {"purpose": "synthetic_functional_QA_not_real_validation",
        "config_provisional": True, "rows": report, "failure_iou_policy": "0 on fallback; no quality claim on real clothing"})


def response_examples(output):
    source = cv2.cvtColor(cv2.imread(str(ROOT / "tests/fixtures/m2/source_rgb.png")), cv2.COLOR_BGR2RGB)
    box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[32, 16, 96, 80]])
    examples = {}
    for name, mode, rgb in [("raw", "raw", source), ("morph", "open_close", source),
                            ("fallback", "raw", np.full_like(source, 245)), ("off", "off", source)]:
        detector = ClothingDetector(mask_config=MaskConfig(mode=mode))
        detector._model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[box])])
        batch = detector.detect_prepared(rgb, rgb, query_id=f"phase3_{name}")
        payload = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics).model_dump()
        color = extract_source_color(detection=payload["crops"][0])
        assert color.valid_pixels == (np.count_nonzero(batch.crops[0]["mask"]) if batch.crops[0]["mask"] is not None else 4096)
        examples[name] = payload
    write_json(ROOT / "tests/fixtures/m2/phase3_responses.json", examples)
    write_json(output / "mask_config_candidate.json", asdict(MaskConfig(mode="open_close")))


def regression_check(output):
    manifest = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))
    detector = ClothingDetector(model_path=str(ROOT / "object_detection/best.pt"), mask_config=MaskConfig())
    before_hash = hashlib.sha256(Path(detector.model_path).read_bytes()).hexdigest()
    model = detector._ensure_model()
    model.overrides.update(device="cpu", imgsz=640, iou=.7, max_det=300, half=False)
    records, previews = [], {"masked": [], "fallback": []}
    counts = {mode: Counter() for mode in MODES}
    reasons = {mode: Counter() for mode in MODES}
    latencies = {mode: [] for mode in MODES}
    first_masked_query = None
    for query in manifest["queries"]:
        path = ROOT / query["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == query["sha256"]
        batch = detector.detect_batch_from_bytes(path.read_bytes(), query_id=query["query_id"])
        wire_crops = crops_to_response(batch.crops).model_dump()["crops"]
        for crop, wire in zip(batch.crops, wire_crops):
            roi = crop["source_rgb"]
            row = {"query_id": query["query_id"], "garment_id": crop["garment_id"], "class_name": crop["class_name"],
                   "bbox": crop["bbox"], "modes": {}}
            for mode in MODES:
                result = segment_roi(roi, MaskConfig(mode=mode))
                payload = dict(wire, roi_metadata={**wire["roi_metadata"], **result.metadata},
                               mask_png_base64=binary_mask_to_base64_png(result.mask) if result.mask is not None else None)
                color = extract_source_color(detection=payload)
                expected_count = int(np.count_nonzero(result.mask)) if result.mask is not None else roi.shape[0] * roi.shape[1]
                assert color and color.valid_pixels == expected_count
                assert color.mask_status == ("valid" if result.mask is not None else "missing")
                counts[mode][result.metadata["roi_status"]] += 1
                if result.metadata["fallback_reason"]:
                    reasons[mode][result.metadata["fallback_reason"]] += 1
                latencies[mode].append(result.metadata["segmentation"]["latency_ms"])
                row["modes"][mode] = {**result.metadata, "m3_mask_status": color.mask_status, "m3_valid_pixels": color.valid_pixels}
                status = result.metadata["roi_status"]
                if mode == "raw" and len(previews[status]) < 3:
                    previews[status].append((f'{query["query_id"]}', [Image.fromarray(roi), mask_image(result.raw_mask),
                                               mask_image(result.processed_mask), overlay(roi, result.mask)]))
                if mode == "raw" and result.mask is not None and first_masked_query is None:
                    first_masked_query = query["query_id"]
            records.append(row)
        print(f'{query["query_id"]}: {len(batch.crops)} crops, all four mask modes reached HSV correctly', flush=True)
    report = {
        "status": "passed", "data_role": "old_regression_development_not_independent_validation",
        "model_sha256": before_hash, "model_unchanged": before_hash == hashlib.sha256(Path(detector.model_path).read_bytes()).hexdigest(),
        "query_count": len(manifest["queries"]), "roi_count": len(records), "HSV_config_version": HSVConfig().version,
        "mode_counts": {mode: dict(counter) for mode, counter in counts.items()},
        "fallback_reasons": {mode: dict(counter) for mode, counter in reasons.items()},
        "segmentation_latency_ms": {mode: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))} for mode, values in latencies.items()},
        "latency_scope": "segmentation helper only; excludes YOLO, decode, PNG, HTTP, HSV and retrieval",
        "http_smoke_query": first_masked_query, "config_provisional": True,
        "limitations": "No real GT masks, so no real IoU/recall or model/method winner. Default stays off. All 4 modes are evaluated on the same source ROIs.",
        "rows": records,
    }
    write_json(output / "regression.json", report)
    sheet(previews["masked"] + previews["fallback"], ["real ROI (no GT)", "raw candidate", "processed candidate", "delivered overlay"], output / "real_contact.jpg")
    print(json.dumps({key: report[key] for key in ["query_count", "roi_count", "mode_counts", "fallback_reasons"]}))


def http_check(output, api_url):
    report = json.loads((output / "regression.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))
    query = next(row for row in manifest["queries"] if row["query_id"] == report["http_smoke_query"])
    data = (ROOT / query["image"]).read_bytes()
    with httpx.Client(base_url=api_url, timeout=120, trust_env=False) as client:
        response = client.post("/detect/upload", params={"query_id": "phase3_http"}, files={"file": ("outfit.jpg", data, "image/jpeg")})
        response.raise_for_status()
    payload = response.json()
    masked, fallback = 0, 0
    for crop in payload["crops"]:
        color = extract_source_color(detection=crop)
        if crop["mask_png_base64"] is not None:
            mask = cv2.imdecode(np.frombuffer(base64.b64decode(crop["mask_png_base64"]), np.uint8), 0)
            assert mask.shape == (crop["roi_metadata"]["roi_height"], crop["roi_metadata"]["roi_width"])
            assert set(np.unique(mask)).issubset({0, 255})
            assert color.mask_status == "valid" and color.valid_pixels == int(np.count_nonzero(mask))
            masked += 1
        else:
            assert color.mask_status == "missing" and crop["roi_metadata"]["roi_status"] == "fallback"
            fallback += 1
    assert masked > 0, "Choose a regression query with an accepted mask for the enabled-mode HTTP smoke"
    write_json(output / "http_enabled_mask.json", {"status": "passed", "query_id": query["query_id"],
        "mode": "raw", "HTTP": response.status_code, "masked_crops": masked, "fallback_crops": fallback,
        "real_yolo_and_http": True, "real_m3_HSV_pixel_selection": True, "encoder_or_DB": False})
    print(f"Real HTTP mask smoke passed: {masked} masked and {fallback} fallback crops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["examples", "regression", "http"])
    parser.add_argument("--output", default="docs/m2/phase3")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "examples":
        synthetic_check(output)
        response_examples(output)
    elif args.mode == "regression":
        regression_check(output)
    else:
        http_check(output, args.api_url)
