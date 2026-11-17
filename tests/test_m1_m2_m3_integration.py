"""Regression checks for the merged upload -> M1 -> M2 -> M3 color path."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from AI.features import color_input_from_detection
from AI.hsv_descriptor import extract_hsv_descriptor
from object_detection.api import app
from object_detection.detector import ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.segmentation import MaskConfig
from object_detection.utils import crops_to_response


def upload_bytes():
    pixels = np.full((160, 240, 3), 20, dtype=np.uint8)
    pixels[40:120, 60:180] = (90, 35, 20)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    return stream.getvalue()


def make_detector():
    detector = ClothingDetector(
        preprocess_config=PreprocessConfig(mode="gamma", max_side=120, gamma=0.5),
        mask_config=MaskConfig(mode="raw"),
    )
    box = SimpleNamespace(xyxy=[[0, 0, 80, 120]], cls=[0], conf=[0.9])
    detector._model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
    return detector


class M1M2M3IntegrationTests(unittest.TestCase):
    def test_exif_resize_gamma_preserve_source_mask_and_metadata_through_m3(self):
        detector = make_detector()
        encoded = upload_bytes()
        prepared = prepare_image(encoded, detector.preprocess_config)
        batch = detector.detect_batch_from_bytes(encoded, query_id="merged")
        detector._model.predict.assert_called_once()
        np.testing.assert_array_equal(detector._model.predict.call_args.args[0], prepared.detection_rgb[:, :, ::-1])
        payload = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics).model_dump()
        crop = payload["crops"][0]
        roi, mask, metadata = color_input_from_detection(crop, np.zeros((160, 240, 3), np.uint8))
        np.testing.assert_array_equal(roi, prepared.source_rgb)
        self.assertFalse(np.array_equal(roi, prepared.detection_rgb))
        self.assertEqual(roi.shape, (120, 80, 3))
        self.assertEqual(crop["garment_id"], "merged_g000")
        geometry = crop["roi_metadata"]["geometry_meta"]
        self.assertEqual(crop["roi_metadata"]["geometry_mode"], "m1_prepared")
        self.assertEqual((geometry["working_width"], geometry["working_height"]), (80, 120))
        self.assertEqual(geometry["exif_orientation"], 6)
        self.assertEqual(geometry["scale_x"], 0.5)
        self.assertEqual(geometry["scale_y"], 0.5)
        self.assertTrue(crop["roi_metadata"]["quality_meta"]["gamma_applied"])
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, roi.shape[:2])
        descriptor = extract_hsv_descriptor(roi, mask, metadata=metadata)
        self.assertEqual(descriptor.mask_status, "valid")
        self.assertEqual(descriptor.valid_pixels, int(np.count_nonzero(mask)))

    def test_upload_runs_preparation_and_yolo_once_and_keeps_query_id(self):
        detector = make_detector()
        with patch("object_detection.api.detector", detector), patch.object(
            type(detector), "model_available", True
        ), patch("object_detection.detector.prepare_image", wraps=prepare_image) as prepare:
            response = TestClient(app).post("/detect/upload?query_id=once", files={
                "file": ("oriented.jpg", upload_bytes(), "image/jpeg")
            })
        self.assertEqual(response.status_code, 200, response.text)
        prepare.assert_called_once()
        detector._model.predict.assert_called_once()
        self.assertEqual(response.json()["crops"][0]["garment_id"], "once_g000")

    def test_legacy_dual_bgr_entry_point_still_supports_m1_ablation(self):
        detector = make_detector()
        source = np.full((120, 80, 3), (20, 35, 90), dtype=np.uint8)
        detection = np.full_like(source, 128)
        crop = detector._process(source, detection)[0]
        np.testing.assert_array_equal(detector._model.predict.call_args.args[0], detection)
        np.testing.assert_array_equal(crop["source_rgb"], cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        detector._model.predict.reset_mock()
        with self.assertRaises(ValueError):
            detector._process(source, detection[:50])
        detector._model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
