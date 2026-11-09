"""Create deterministic contract examples and a regression-only image manifest."""

import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from object_detection.detector import ClothingDetector
from object_detection.utils import crops_to_response


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/m2"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def png_base64(image):
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def prepare_fixtures():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    mask_image = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask_image).polygon(
        [(0, 0), (63, 0), (63, 23), (47, 23), (47, 63), (16, 63), (16, 23), (0, 23)],
        fill=255,
    )
    source = np.full((96, 128, 3), [15, 25, 35], dtype=np.uint8)
    source[:16, :16] = [240, 20, 10]
    source[80:, 112:] = [10, 230, 40]
    source[16:80, 32:96][np.array(mask_image) == 255] = [30, 100, 200]
    detection = np.minimum(source.astype(np.uint16) + 40, 255).astype(np.uint8)
    Image.fromarray(source).save(FIXTURES / "source_rgb.png")
    Image.fromarray(detection).save(FIXTURES / "detection_rgb.png")
    mask_image.save(FIXTURES / "reference_mask.png")
    roi_image = Image.fromarray(source[16:80, 32:96])
    roi_image.save(FIXTURES / "source_roi.png")

    fake_box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[32, 16, 96, 80]])
    detector = ClothingDetector()
    detector._model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[fake_box])])
    current = crops_to_response(detector._process(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))).model_dump()
    # Keep the historical example legacy-only after additive schema changes in later phases.
    legacy = {key: current[key] for key in ("success", "total_crops", "crops", "message")}
    new_crop_fields = {"query_id", "garment_id", "source_crop_png_base64", "mask_png_base64", "roi_metadata"}
    legacy["crops"] = [{key: value for key, value in crop.items() if key not in new_crop_fields} for crop in legacy["crops"]]
    write_json(FIXTURES / "legacy_response.json", legacy)
    candidate = copy.deepcopy(legacy)
    candidate.update(schema_version="m2.roi.v1", query_id="fixture_m2")
    candidate["crops"][0].update(
        query_id="fixture_m2",
        garment_id="fixture_m2_g000",
        source_crop_png_base64=png_base64(roi_image),
        mask_png_base64=None,
        roi_metadata={
            "bbox_xyxy": [32, 16, 96, 80], "working_width": 128, "working_height": 96,
            "roi_width": 64, "roi_height": 64, "coordinate_space": "m1_working_image",
            "color_space": "RGB", "roi_source": "synthetic_bbox_tight",
            "roi_status": "unmasked", "fallback_reason": "mask_disabled",
            "mask_method": "off", "mask_config_id": "fixture_v1",
            "geometry_mode": "synthetic_same_geometry",
        },
    )
    write_json(FIXTURES / "candidate_response_unmasked.json", candidate)
    masked = copy.deepcopy(candidate)
    masked["crops"][0]["mask_png_base64"] = png_base64(mask_image)
    masked["crops"][0]["roi_metadata"].update(
        roi_status="masked", fallback_reason=None, mask_method="synthetic_reference",
    )
    write_json(FIXTURES / "candidate_response_masked.json", masked)
    write_json(FIXTURES / "no_detection_response.json", {
        "success": False, "total_crops": 0, "crops": [], "message": "No clothing items found",
    })
    write_json(FIXTURES / "fixture_manifest.json", {
        "schema_version": "m2.fixture.v1", "origin": "generated_synthetic_not_real_evaluation",
        "source": "source_rgb.png", "detection": "detection_rgb.png",
        "branch_transform": "saturating +40 RGB units; fixture only, not M1 gamma",
        "shape_hwc": [96, 128, 3], "dtype": "uint8", "color_space": "RGB",
        "target_bbox_xyxy": [32, 16, 96, 80], "reference_mask": "reference_mask.png",
        "reference_mask_space": "target_roi", "foreground": 255,
        "bbox_cases": [
            {"id": "tight", "bbox_xyxy": [32, 16, 96, 80], "clipped_xyxy": [32, 16, 96, 80], "valid_after_clip": True},
            {"id": "edge", "bbox_xyxy": [0, 0, 64, 64], "clipped_xyxy": [0, 0, 64, 64], "valid_after_clip": True},
            {"id": "outside", "bbox_xyxy": [-8, -4, 140, 110], "clipped_xyxy": [0, 0, 128, 96], "valid_after_clip": True},
            {"id": "small", "bbox_xyxy": [1, 2, 20, 30], "clipped_xyxy": [1, 2, 20, 30], "valid_after_clip": True, "legacy_minimum_policy": "skip"},
            {"id": "inverted", "bbox_xyxy": [50, 30, 20, 70], "clipped_xyxy": [50, 30, 20, 70], "valid_after_clip": False},
            {"id": "empty", "bbox_xyxy": [8, 8, 8, 20], "clipped_xyxy": [8, 8, 8, 20], "valid_after_clip": False},
        ],
    })
    sheet = Image.new("RGB", (720, 190), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate([
        ("source RGB", Image.fromarray(source)), ("detection +40", Image.fromarray(detection)),
        ("tight ROI", roi_image), ("mask white=foreground", mask_image.convert("RGB")),
    ]):
        draw.text((index * 180 + 8, 8), title, fill="black")
        preview = image.copy()
        preview.thumbnail((164, 144), Image.Resampling.NEAREST)
        sheet.paste(preview, (index * 180 + 8, 36))
        draw.rectangle((index * 180 + 7, 35, index * 180 + 8 + preview.width, 36 + preview.height), outline="gray")
    sheet.save(FIXTURES / "contact_sheet.png")


def prepare_regression_manifest():
    labels = json.loads((ROOT / "evaluation/labels.real.json").read_text(encoding="utf-8"))
    sources = {
        row["query_id"]: row
        for row in json.loads((ROOT / "evaluation/real_outfit_sources.json").read_text(encoding="utf-8"))
    }
    rows = []
    for row in labels:
        image = ROOT / "evaluation" / row["image"]
        source = sources[row["query_id"]]
        rows.append({
            "query_id": row["query_id"], "image": image.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "split": "regression", "previously_used_for_development": True,
            "expected_items_legacy_unverified": row["expected_items"],
            "target_bbox_xyxy": None, "mask_path": None, "label_status": "needs_human_annotation",
            "lighting_group": "unreviewed", "source_url": source["source_url"],
            "author": source["author"], "license": source["license"], "license_url": source["license_url"],
        })
    write_json(ROOT / "evaluation/m2/regression_manifest.json", {
        "schema_version": "m2.manifest.v1", "purpose": "phase1_baseline_regression_only",
        "independent_test": False, "queries": rows,
    })
    write_json(ROOT / "evaluation/m2/annotation_manifest.draft.json", {
        "schema_version": "m2.annotation.v1", "status": "empty_draft_not_evaluation_data",
        "required_counts": {"validation_images": 5, "test_images": 15, "validation_masks": 2, "test_masks": 3},
        "queries": [],
        "record_template": {
            "query_id": "m2_new_001", "image": None, "split": None, "target_id": "m2_new_001_target",
            "bbox_xyxy": None, "class_name": None, "color_names": [], "lighting_group": None,
            "source_url": None, "license": None, "source_group": None, "geometry_meta": None,
            "mask_path": None, "mask_coordinate_space": "m1_working_image",
            "label_status": "pending_human_review", "reviewer": None,
        },
    })
    return len(rows)


if __name__ == "__main__":
    prepare_fixtures()
    count = prepare_regression_manifest()
    print(f"Prepared synthetic fixtures and {count} regression images; independent test labels remain empty.")
