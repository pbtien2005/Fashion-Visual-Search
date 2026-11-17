import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from fastapi.testclient import TestClient

from object_detection.api import app, detector


IMAGE = (Path(__file__).parent / "fixtures/m2/source_rgb.png").read_bytes()


class DetectionROIAPITests(unittest.TestCase):
    def upload(self, **kwargs):
        return TestClient(app).post("/detect/upload", files={"file": ("fixture.png", IMAGE, "image/png")}, **kwargs)

    def test_query_id_source_png_and_null_mask_are_returned_over_http(self):
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[32, 16, 96, 80]])
        model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
        with patch.object(type(detector), "model_available", new_callable=PropertyMock, return_value=True):
            with patch.object(detector, "_model", model):
                response = self.upload(params={"query_id": "http_fixture"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_id"], "http_fixture")
        self.assertEqual(payload["schema_version"], "m2.roi.v1")
        crop = payload["crops"][0]
        self.assertEqual(crop["garment_id"], "http_fixture_g000")
        self.assertTrue(crop["source_crop_png_base64"])
        self.assertIsNone(crop["mask_png_base64"])
        self.assertEqual(crop["roi_metadata"]["geometry_mode"], "m1_prepared")

    def test_no_detection_keeps_legacy_behavior_and_new_request_metadata(self):
        with patch.object(type(detector), "model_available", new_callable=PropertyMock, return_value=True):
            with patch.object(detector, "_model", SimpleNamespace(predict=lambda *a, **kw: [])):
                response = self.upload()
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(len(payload["query_id"]), 32)
        self.assertEqual(payload["crops"], [])
        self.assertEqual(payload["diagnostics"]["raw_box_count"], 0)

    def test_invalid_image_returns_400_and_invalid_id_returns_422(self):
        with patch.object(type(detector), "model_available", new_callable=PropertyMock, return_value=True):
            response = TestClient(app).post("/detect/upload", files={"file": ("bad.jpg", b"bad", "image/jpeg")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.upload(params={"query_id": "../escape"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
