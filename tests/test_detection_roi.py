import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from object_detection.detector import ClothingDetector
from object_detection.utils import crops_to_response


FIXTURES = Path(__file__).parent / "fixtures/m2"


def fake_box(bbox, class_id=0, confidence=0.9):
    return SimpleNamespace(xyxy=[bbox], cls=[class_id], conf=[confidence])


def detector_with(boxes):
    detector = ClothingDetector()
    detector._model = SimpleNamespace(
        names={0: "short_sleeve_top"},
        predict=Mock(return_value=[SimpleNamespace(boxes=boxes)]),
    )
    return detector


def decode_rgb(payload):
    encoded = np.frombuffer(base64.b64decode(payload, validate=True), np.uint8)
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class DetectionROITests(unittest.TestCase):
    def setUp(self):
        self.source = cv2.cvtColor(cv2.imread(str(FIXTURES / "source_rgb.png")), cv2.COLOR_BGR2RGB)
        self.detection = cv2.cvtColor(cv2.imread(str(FIXTURES / "detection_rgb.png")), cv2.COLOR_BGR2RGB)

    def detect(self, detector, **kwargs):
        return detector.detect_prepared(self.source, self.detection, query_id="query_2", **kwargs)

    def test_yolo_sees_detection_bgr_but_roi_and_legacy_crop_use_source(self):
        detector = detector_with([fake_box([32, 16, 96, 80])])
        source_before, detection_before = self.source.copy(), self.detection.copy()
        batch = self.detect(detector)
        seen = detector._model.predict.call_args.args[0]
        self.assertTrue(np.array_equal(seen, self.detection[:, :, ::-1]))
        crop = batch.crops[0]
        self.assertTrue(np.array_equal(crop["source_rgb"], self.source[16:80, 32:96]))
        self.assertIsNone(crop["mask"])
        self.assertEqual(crop["roi_metadata"]["roi_status"], "unmasked")
        self.assertEqual(crop["roi_metadata"]["fallback_reason"], "mask_disabled")
        self.assertTrue(np.array_equal(source_before, self.source))
        self.assertTrue(np.array_equal(detection_before, self.detection))
        crop["source_rgb"][:] = 0
        self.assertTrue(np.array_equal(source_before, self.source), "ROI must not share writable source memory")

    def test_clip_outside_box_to_source_canvas(self):
        batch = self.detect(detector_with([fake_box([-8, -4, 140, 110])]))
        crop = batch.crops[0]
        self.assertEqual(crop["bbox"], [0, 0, 128, 96])
        self.assertEqual(crop["source_rgb"].shape, (96, 128, 3))
        self.assertEqual(crop["roi_metadata"]["bbox_xyxy"], crop["bbox"])

    def test_fractional_roi_rounds_outward_without_changing_legacy_jpeg_policy(self):
        detector = detector_with([fake_box([32.8, 16.2, 95.1, 79.3])])
        crop = self.detect(detector).crops[0]
        self.assertEqual(crop["bbox"], [32, 16, 96, 80])
        self.assertEqual(crop["original_bbox"], [32, 16, 95, 79])
        old_crop = detector._make_crop_variant(
            cv2.cvtColor(self.source, cv2.COLOR_RGB2BGR), [32, 16, 95, 79], 128, 96, "padded", 0.15,
        )
        self.assertTrue(np.array_equal(crop["image"], old_crop["image"]))

    def test_edge_boxes_use_exclusive_endpoints(self):
        crop = self.detect(detector_with([fake_box([64, 32, 128, 96])])).crops[0]
        self.assertEqual(crop["source_rgb"].shape, (64, 64, 3))
        self.assertTrue(np.array_equal(crop["source_rgb"], self.source[32:96, 64:128]))

    def test_invalid_and_small_boxes_are_counted_without_becoming_garments(self):
        boxes = [fake_box(bbox) for bbox in [
            [50, 30, 20, 70], [8, 8, 8, 20], [150, 0, 200, 90],
            [float("nan"), 0, 80, 90], [1, 2, 20, 30], [32, 16, 96, 80],
        ]]
        batch = self.detect(detector_with(boxes))
        self.assertEqual(len(batch.crops), 1)
        self.assertEqual(batch.diagnostics["raw_box_count"], 6)
        self.assertEqual(batch.diagnostics["invalid_bbox_count"], 4)
        self.assertEqual(batch.diagnostics["no_valid_variant_count"], 1)
        self.assertEqual(batch.diagnostics["skipped_variant_count"], 3)

    def test_legacy_variant_minimum_stays_at_50_pixels(self):
        batch = self.detect(detector_with([fake_box([32, 24, 72, 64])]))
        self.assertEqual([v["name"] for v in batch.crops[0]["variants"]], ["padded", "wide"])
        self.assertEqual(batch.crops[0]["source_rgb"].shape, (40, 40, 3))
        self.assertEqual(batch.diagnostics["skipped_variant_count"], 1)

    def test_ids_use_deterministic_order_for_same_request(self):
        boxes = [fake_box([64, 32, 128, 96], 5), fake_box([32, 16, 96, 80], 0)]
        first = self.detect(detector_with(boxes))
        second = self.detect(detector_with(list(reversed(boxes))))
        identify = lambda batch: [(c["garment_id"], c["class_id"], c["bbox"]) for c in batch.crops]
        self.assertEqual(identify(first), identify(second))
        self.assertEqual(first.crops[0]["garment_id"], "query_2_g000")
        self.assertEqual(first.crops[1]["garment_id"], "query_2_g001")

    def test_no_detection_still_preserves_query_and_diagnostics(self):
        batch = self.detect(detector_with([]))
        response = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics)
        self.assertEqual(response.query_id, "query_2")
        self.assertEqual(response.total_crops, 0)
        self.assertEqual(response.diagnostics.delivered_crop_count, 0)

    def test_prepared_input_rejects_bad_geometry_type_and_empty_images_before_yolo(self):
        detector = detector_with([])
        bad_inputs = [
            (self.source, self.detection[:80]), (self.source.astype(float), self.detection),
            (self.source[:, :, 0], self.detection), (np.zeros((0, 128, 3), np.uint8), self.detection),
            ([], self.detection),
        ]
        for source, detection in bad_inputs:
            with self.subTest(shape=getattr(source, "shape", None)):
                with self.assertRaises(ValueError):
                    detector.detect_prepared(source, detection)
        detector._model.predict.assert_not_called()

    def test_metadata_is_copied_and_geometry_mismatch_is_rejected(self):
        geometry = {"mode": "m1_fixture", "working_width": 128, "working_height": 96, "scale": [1, 1]}
        quality = {"gamma_enabled": True}
        crop = self.detect(detector_with([fake_box([32, 16, 96, 80])]), geometry_meta=geometry, quality_meta=quality).crops[0]
        geometry["scale"][0] = 2
        self.assertEqual(crop["roi_metadata"]["geometry_meta"]["scale"], [1, 1])
        self.assertTrue(crop["roi_metadata"]["quality_meta"]["gamma_enabled"])
        with self.assertRaises(ValueError):
            self.detect(detector_with([]), geometry_meta={"working_width": 99})

    def test_invalid_query_id_and_non_json_metadata_are_rejected(self):
        for query in ["", "../escape", "x" * 129, "with space"]:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    detector_with([]).detect_prepared(self.source, self.detection, query_id=query)
        with self.assertRaises(ValueError):
            self.detect(detector_with([]), quality_meta={"invalid": float("nan")})

    def test_adapter_from_bytes_records_identity_and_returns_rgb_source(self):
        detector = detector_with([fake_box([32, 16, 96, 80])])
        batch = detector.detect_batch_from_bytes((FIXTURES / "source_rgb.png").read_bytes(), query_id="upload")
        self.assertEqual(batch.crops[0]["roi_metadata"]["geometry_mode"], "legacy_identity")
        self.assertTrue(np.array_equal(batch.crops[0]["source_rgb"], self.source[16:80, 32:96]))
        with self.assertRaises(ValueError):
            detector.detect_batch_from_bytes(b"not an image")
        with self.assertRaises(ValueError):
            detector.detect_batch_from_bytes(b"")

    def test_png_round_trip_and_jpeg_legacy_fields_survive_serialization(self):
        batch = self.detect(detector_with([fake_box([32, 16, 96, 80])]))
        response = crops_to_response(batch.crops, query_id=batch.query_id, diagnostics=batch.diagnostics)
        payload = json.loads(response.model_dump_json())
        crop = payload["crops"][0]
        self.assertEqual(payload["schema_version"], "m2.roi.v1")
        self.assertEqual(crop["query_id"], payload["query_id"])
        self.assertTrue(np.array_equal(decode_rgb(crop["source_crop_png_base64"]), self.source[16:80, 32:96]))
        self.assertIsNone(crop["mask_png_base64"])
        self.assertEqual(crop["roi_metadata"]["roi_width"], 64)
        self.assertEqual(len(crop["crop_variants"]), 3)
        self.assertTrue(base64.b64decode(crop["image_base64"]).startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
