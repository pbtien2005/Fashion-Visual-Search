import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from AI.production_color import extract_source_color
from object_detection.api import app, detector as api_detector
from object_detection.detector import ClothingDetector
from object_detection.segmentation import MaskConfig, load_mask_config
from object_detection.utils import binary_mask_to_base64_png, crops_to_response


FIXTURES = Path(__file__).parent / "fixtures/m2"


def fake_model():
    box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[32, 16, 96, 80]])
    return SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=[box])])


class DetectionSegmentationTests(unittest.TestCase):
    def image(self):
        return cv2.cvtColor(cv2.imread(str(FIXTURES / "source_rgb.png")), cv2.COLOR_BGR2RGB)

    def detect(self, image, mode):
        detector = ClothingDetector(mask_config=MaskConfig(mode=mode))
        detector._model = fake_model()
        return detector.detect_prepared(image, image, query_id="masked_query")

    def test_otsu_mask_round_trip_reaches_real_m3_selection(self):
        batch = self.detect(self.image(), "raw")
        crop = batch.crops[0]
        self.assertIsNotNone(crop["raw_mask"])
        self.assertIsNotNone(crop["processed_mask"])
        payload = crops_to_response(batch.crops, query_id=batch.query_id).model_dump()["crops"][0]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload["mask_png_base64"]), np.uint8), 0)
        np.testing.assert_array_equal(decoded, crop["mask"])
        color = extract_source_color(detection=payload)
        self.assertEqual(color.mask_status, "valid")
        self.assertEqual(color.valid_pixels, int(np.count_nonzero(crop["mask"])))
        self.assertEqual(color.valid_pixels, 2816)
        self.assertEqual(color.metadata["m2_roi_metadata"]["roi_status"], "masked")
        self.assertGreaterEqual(payload["roi_metadata"]["segmentation"]["latency_ms"], 0)

    def test_fallback_null_uses_full_source_roi_and_keeps_reason_in_m3(self):
        image = np.full((96, 128, 3), 245, np.uint8)
        batch = self.detect(image, "raw")
        payload = crops_to_response(batch.crops).model_dump()["crops"][0]
        self.assertIsNone(payload["mask_png_base64"])
        self.assertEqual(payload["roi_metadata"]["fallback_reason"], "low_contrast")
        color = extract_source_color(detection=payload)
        self.assertEqual(color.valid_pixels, 4096)
        self.assertEqual(color.mask_status, "missing")
        self.assertEqual(color.metadata["m2_roi_metadata"]["fallback_reason"], "low_contrast")

    def test_mask_switch_does_not_change_encoder_crop_or_id(self):
        off = self.detect(self.image(), "off").crops[0]
        raw = self.detect(self.image(), "raw").crops[0]
        self.assertEqual(off["garment_id"], raw["garment_id"])
        for a, b in zip(off["variants"], raw["variants"]):
            np.testing.assert_array_equal(a["image"], b["image"])

    def test_bad_or_misaligned_masks_are_not_serialized_as_valid(self):
        crop = self.detect(self.image(), "raw").crops[0]
        with self.assertRaises(ValueError):
            crops_to_response([dict(crop, mask=np.zeros((2, 2), np.uint8))])
        with self.assertRaises(ValueError):
            binary_mask_to_base64_png(np.full((3, 3), 128, np.uint8))
        with self.assertRaises(ValueError):
            binary_mask_to_base64_png(np.zeros((3, 3, 3), np.uint8))

    def test_real_http_handler_with_fake_yolo_transmits_computed_mask(self):
        with patch.object(api_detector, "mask_config", MaskConfig(mode="raw")):
            with patch.object(api_detector, "_model", fake_model()):
                with patch.object(type(api_detector), "model_available", new_callable=unittest.mock.PropertyMock, return_value=True):
                    response = TestClient(app).post("/detect/upload", params={"query_id": "api_mask"},
                        files={"file": ("source.png", (FIXTURES / "source_rgb.png").read_bytes(), "image/png")})
        self.assertEqual(response.status_code, 200)
        crop = response.json()["crops"][0]
        self.assertTrue(crop["mask_png_base64"])
        self.assertEqual(extract_source_color(detection=crop).mask_status, "valid")

    def test_environment_mode_is_explicit_and_off_by_default(self):
        with patch.dict(os.environ, {"M2_MASK_MODE": "off", "M2_MASK_CONFIG_PATH": ""}):
            self.assertEqual(load_mask_config().mode, "off")
        with patch.dict(os.environ, {"M2_MASK_MODE": "raw", "M2_MASK_CONFIG_PATH": ""}):
            self.assertEqual(load_mask_config().mode, "raw")
        with patch.dict(os.environ, {"M2_MASK_MODE": "unknown", "M2_MASK_CONFIG_PATH": ""}):
            with self.assertRaises(ValueError):
                load_mask_config()

    def test_explicit_config_file_is_loaded_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.json"
            path.write_text(json.dumps({"mode": "closing", "min_contrast": 20}), encoding="utf-8")
            with patch.dict(os.environ, {"M2_MASK_CONFIG_PATH": str(path)}, clear=True):
                config = load_mask_config()
            self.assertEqual(config.mode, "closing")
            self.assertEqual(config.min_contrast, 20)
            self.assertNotEqual(config.config_id, MaskConfig().config_id)


if __name__ == "__main__":
    unittest.main()
