import base64
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_detection.schemas import DetectionResponse


FIXTURES = Path(__file__).parent / "fixtures" / "m2"


def read_json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def decode_png(payload, flags):
    raw = base64.b64decode(payload, validate=True)
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)


class ContractFixtureTests(unittest.TestCase):
    def test_prepared_branches_have_same_geometry_and_distinct_rgb(self):
        source = cv2.imread(str(FIXTURES / "source_rgb.png"))
        detection = cv2.imread(str(FIXTURES / "detection_rgb.png"))
        self.assertEqual(source.shape, (96, 128, 3))
        self.assertEqual(detection.shape, source.shape)
        self.assertEqual(source.dtype, np.uint8)
        self.assertEqual(source[4, 4, ::-1].tolist(), [240, 20, 10])
        self.assertEqual(source[20, 40, ::-1].tolist(), [30, 100, 200])
        self.assertEqual(detection[20, 40, ::-1].tolist(), [70, 140, 240])
        self.assertFalse(np.array_equal(source, detection))

    def test_roi_png_is_exact_source_slice_without_padding(self):
        response = read_json("candidate_response_unmasked.json")
        crop = response["crops"][0]
        roi = decode_png(crop["source_crop_png_base64"], cv2.IMREAD_COLOR)
        source = cv2.imread(str(FIXTURES / "source_rgb.png"))
        self.assertTrue(np.array_equal(roi, source[16:80, 32:96]))
        self.assertEqual(roi.shape, (64, 64, 3))
        self.assertIsNone(crop["mask_png_base64"])
        self.assertEqual(crop["roi_metadata"]["fallback_reason"], "mask_disabled")

    def test_reference_mask_foreground_is_aligned_with_blue_fabric(self):
        crop = read_json("candidate_response_masked.json")["crops"][0]
        roi = decode_png(crop["source_crop_png_base64"], cv2.IMREAD_COLOR)
        mask = decode_png(crop["mask_png_base64"], cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, roi.shape[:2])
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        expected_bgr = np.array([200, 100, 30], dtype=np.uint8)
        self.assertTrue(np.all(roi[mask == 255] == expected_bgr))
        self.assertTrue(np.array_equal(mask, cv2.imread(str(FIXTURES / "reference_mask.png"), 0)))

    def test_current_schema_accepts_legacy_and_additive_candidate_payloads(self):
        legacy = read_json("legacy_response.json")
        legacy_model = DetectionResponse.model_validate(legacy)
        self.assertEqual(legacy_model.total_crops, 1)
        for name in ("candidate_response_unmasked.json", "candidate_response_masked.json"):
            candidate = DetectionResponse.model_validate(read_json(name))
            for field in legacy["crops"][0]:
                self.assertEqual(getattr(candidate.crops[0], field), getattr(legacy_model.crops[0], field))
            self.assertEqual(candidate.schema_version, "m2.roi.v1")
            self.assertTrue(candidate.crops[0].source_crop_png_base64)
        self.assertIsNone(legacy_model.crops[0].source_crop_png_base64)

    def test_legacy_encoder_still_receives_three_jpeg_variants(self):
        crop = read_json("legacy_response.json")["crops"][0]
        self.assertEqual([v["name"] for v in crop["crop_variants"]], ["tight", "padded", "wide"])
        for variant in crop["crop_variants"]:
            raw = base64.b64decode(variant["image_base64"], validate=True)
            self.assertTrue(raw.startswith(b"\xff\xd8"))
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (224, 224, 3))

    def test_bbox_contract_includes_small_invalid_and_out_of_bounds_cases(self):
        cases = {case["id"]: case for case in read_json("fixture_manifest.json")["bbox_cases"]}
        self.assertEqual(cases["outside"]["clipped_xyxy"], [0, 0, 128, 96])
        self.assertEqual(cases["small"]["legacy_minimum_policy"], "skip")
        self.assertFalse(cases["inverted"]["valid_after_clip"])
        self.assertFalse(cases["empty"]["valid_after_clip"])
        self.assertEqual(read_json("no_detection_response.json")["crops"], [])


if __name__ == "__main__":
    unittest.main()
