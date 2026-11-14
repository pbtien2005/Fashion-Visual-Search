from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from evaluation.m2_phase5 import assess_validation, verify_lock, audit_cache, digest, frozen_color_rows, summarize_color


def validation_rows(mode, iou=.9, similarity=.96):
    return [dict(query_id=f"val_{i}", split="validation", mask_mode=mode,
                 label_status="human_reviewed", mask_label_status="human_reviewed",
                 matched=True, mask_delivered=mode != "off", mask_iou_end_to_end=iou if mode != "off" else 0,
                 color_reference_similarity=similarity) for i in range(2)]


class Phase5Tests(unittest.TestCase):
    def test_validation_decision_ignores_test_rows_and_rejects_color_regression_or_leaky_masks(self):
        rows = validation_rows("off", similarity=.85) + validation_rows("raw") + validation_rows("open_close", iou=.91)
        rows += [dict(row, split="test", mask_iou_end_to_end=0) for row in validation_rows("open_close")]
        decision = assess_validation(rows)
        self.assertEqual(decision["deployment_mode"], "open_close")
        self.assertEqual(decision["validation_reference_count"], 2)
        rows = validation_rows("off", similarity=.95) + validation_rows("raw", similarity=.90) + validation_rows("open_close", iou=.72)
        self.assertEqual(assess_validation(rows)["deployment_mode"], "off")

    def test_missing_reference_or_failed_delivery_does_not_count_as_improvement(self):
        rows = validation_rows("off", similarity=.8) + validation_rows("raw")
        rows[-1].update(mask_delivered=False, color_reference_similarity=None)
        self.assertEqual(assess_validation(rows)["deployment_mode"], "off")
        self.assertEqual(assess_validation([])["deployment_mode"], "off")

    def test_lock_detects_input_or_profile_change(self):
        lock = dict(input_fingerprint="same", mask_modes=["off", "raw", "open_close"])
        verify_lock(lock, "same")
        with self.assertRaises(ValueError):
            verify_lock(lock, "changed")
        lock["lock_sha256"] = digest(lock)
        verify_lock(lock, "same")
        lock["unexpected_edit"] = True
        with self.assertRaises(ValueError):
            verify_lock(lock, "same")

    def test_production_cache_missing_is_explicit_and_sample_is_never_promoted(self):
        report = audit_cache(env_values={"HSV_COLOR_CACHE_FILE": "tmp/nonexistent_phase5_cache.json"})
        self.assertEqual(report["production_cache"]["status"], "missing")
        self.assertEqual(report["product_retrieval"], "not_measured")
        self.assertFalse(report["sample_cache_is_production"])

    def test_frozen_real_m3_color_transport_and_mask_iou_without_more_inference(self):
        from evaluation.run_m2_eval import evaluate
        from object_detection.detector import ClothingDetector
        from object_detection.image_preprocessing import PreprocessConfig
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((100, 100, 3), (20, 20, 20), np.uint8)
            source[25:75, 25:75] = (180, 80, 40)
            Image.fromarray(source).save(root / "source.png")
            mask = np.zeros((100, 100), np.uint8)
            mask[25:75, 25:75] = 255
            Image.fromarray(mask).save(root / "reference.png")
            query = dict(query_id="fixture", target_id="fixture_target", image="source.png", sha256="fixture",
                         class_name="short_sleeve_top", bbox_normalized=[0, 0, 1, 1], split="validation",
                         source_url="synthetic_fixture", label_status="human_reviewed", mask_status="human_reviewed",
                         mask_path="reference.png", mask_coordinate_space="m1_working_image")
            manifest = dict(dataset_kind="synthetic_fixture_only", preprocess_config=asdict(PreprocessConfig()), queries=[query])
            box = SimpleNamespace(xyxy=[[0, 0, 100, 100]], cls=[0], conf=[.9])
            detector = ClothingDetector()
            detector._model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
            base = evaluate(manifest, detector, root=root, warmup=False)
            calls = detector._model.predict.call_count
            rows = frozen_color_rows(manifest, base, "validation", root=root)
            self.assertEqual(detector._model.predict.call_count, calls)
            self.assertEqual(rows[1]["color_reference_similarity"], 1)
            self.assertLess(rows[0]["color_reference_similarity"], rows[1]["color_reference_similarity"])
            self.assertEqual(rows[1]["HSV_valid_pixels"], 2500)
            self.assertEqual(summarize_color(rows)["raw"]["color_reference_similarity_n"], 1)
            changed = deepcopy(base)
            changed["rows"][1]["predicted_bbox"] = [1, 0, 100, 100]
            with self.assertRaises(ValueError):
                frozen_color_rows(manifest, changed, "validation", root=root)


if __name__ == "__main__":
    unittest.main()
