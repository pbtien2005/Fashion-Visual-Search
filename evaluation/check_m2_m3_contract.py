"""Audit the actual M2 PNG fixtures against the merged M3 decoder, without models/DB."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AI.features import color_input_from_detection
from AI.hsv_descriptor import extract_hsv_descriptor
from AI.production_color import extract_source_color


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/m2"


def read_json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def matches(actual, expected):
    return isinstance(actual, np.ndarray) and np.array_equal(actual, expected)


def audit(output, revision):
    source = cv2.cvtColor(cv2.imread(str(FIXTURES / "source_rgb.png")), cv2.COLOR_BGR2RGB)
    expected_roi = source[16:80, 32:96]
    expected_mask = cv2.imread(str(FIXTURES / "reference_mask.png"), cv2.IMREAD_GRAYSCALE)
    crop = read_json("phase2_response.json")["crops"][0]
    roi, mask, metadata = color_input_from_detection(crop)
    canonical = {"exact_source_roi": matches(roi, expected_roi),
                 "observed_shape": list(roi.shape), "expected_shape": list(expected_roi.shape),
                 "roi_source": metadata["roi_source"]}

    fractional = dict(crop, original_bbox=[32, 16, 95, 79])
    roi_fractional, _, fractional_metadata = color_input_from_detection(fractional, source)
    rounding = {"exact_source_roi": matches(roi_fractional, expected_roi),
                "observed_shape": list(roi_fractional.shape), "expected_shape": list(expected_roi.shape),
                "roi_source": fractional_metadata["roi_source"]}

    masked = read_json("candidate_response_masked.json")["crops"][0]
    _, decoded_mask, _ = color_input_from_detection(masked, source)
    actual_color = extract_source_color(source, masked, item_id="contract_audit")
    expected_color = extract_hsv_descriptor(expected_roi, expected_mask)
    mask_case = {
        "mask_decoded_exactly": matches(decoded_mask, expected_mask),
        "actual_mask_status": actual_color.mask_status if actual_color else "color_unavailable",
        "actual_valid_pixels": actual_color.valid_pixels if actual_color else None,
        "expected_valid_pixels": expected_color.valid_pixels,
        "descriptor_matches_masked_reference": bool(actual_color and np.allclose(actual_color.values, expected_color.values)),
    }

    aliases = {key: value for key, value in masked.items()
               if key not in {"source_crop_png_base64", "mask_png_base64"}}
    aliases.update(source_roi_base64=masked["source_crop_png_base64"], mask_base64=masked["mask_png_base64"])
    alias_roi, alias_mask, _ = color_input_from_detection(aliases)
    alias_color = extract_source_color(None, aliases, item_id="alias_control")
    alias_case = {"exact_roi": matches(alias_roi, expected_roi), "exact_mask": matches(alias_mask, expected_mask),
                  "descriptor_matches_reference": bool(alias_color and np.allclose(alias_color.values, expected_color.values))}
    compatible = all([canonical["exact_source_roi"], rounding["exact_source_roi"], mask_case["mask_decoded_exactly"],
                      mask_case["descriptor_matches_masked_reference"], *alias_case.values()])
    report = {
        "status": "compatible" if compatible else "needs_alignment", "revision": revision,
        "execution": "real M3 decoder/HSV extractor; synthetic M2 fixtures; no YOLO, encoder or DB",
        "canonical_source_png_without_original": canonical,
        "canonical_source_png_with_different_legacy_rounding": rounding,
        "canonical_mask_png": mask_case,
        "legacy_alias_control": alias_case,
        "does_not_measure": ["real garment segmentation quality", "retrieval quality", "live API/cache readiness"],
    }
    if output:
        path = ROOT / output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if compatible else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--revision", default="unspecified")
    args = parser.parse_args()
    raise SystemExit(audit(args.output, args.revision))
