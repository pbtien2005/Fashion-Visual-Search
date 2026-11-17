import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from object_detection.api import app, detector
from object_detection.roi import DetectionBatch


class DetectionBootstrapTests(unittest.TestCase):
    def test_health_observes_model_file_without_loading_yolo(self):
        with patch.object(detector, "_model", None):
            response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_available"], detector.model_available)
        self.assertFalse(response.json()["model_loaded"])

    def test_upload_legacy_response_stays_compatible_with_fixture(self):
        from pathlib import Path

        image_bytes = (Path(__file__).parent / "fixtures/m2/source_rgb.png").read_bytes()
        crop = {
            "class_id": 0, "class_name": "short_sleeve_top", "class_name_vi": "Áo tay ngắn",
            "confidence": 0.9, "bbox": [32, 16, 96, 80],
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        with patch.object(type(detector), "model_available", new_callable=unittest.mock.PropertyMock, return_value=True):
            batch = DetectionBatch(query_id="bootstrap", crops=[crop], diagnostics={})
            with patch.object(detector, "detect_batch_from_bytes", return_value=batch):
                response = TestClient(app).post("/detect/upload", files={"file": ("fixture.png", image_bytes, "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["crops"][0]["bbox"], [32, 16, 96, 80])

    def test_missing_weights_return_controlled_503(self):
        with patch.object(type(detector), "model_available", new_callable=unittest.mock.PropertyMock, return_value=False):
            response = TestClient(app).post("/detect/upload", files={"file": ("fixture.png", b"fixture", "image/png")})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
