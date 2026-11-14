import unittest

import numpy as np

from evaluation.m2_metrics import (
    bbox_to_working, box_iou, mask_iou, normalize_reference_mask, place_roi_mask,
    score_target, summarize_rows,
)


class M2MetricTests(unittest.TestCase):
    def test_iou_zero_one_partial_and_invalid(self):
        self.assertEqual(box_iou([0, 0, 4, 4], [0, 0, 4, 4]), 1)
        self.assertEqual(box_iou([0, 0, 4, 4], [4, 4, 8, 8]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 4, 4], [2, 0, 6, 4]), 1 / 3)
        for box in ([0, 0, 0, 4], [0, 0, float("nan"), 4]):
            with self.assertRaises(ValueError):
                box_iou(box, [0, 0, 4, 4])

    def test_bbox_uses_working_axes_and_rejects_outside_normalized_labels(self):
        np.testing.assert_allclose(bbox_to_working([.2, .1, .8, .9], 80, 51), [16, 5.1, 64, 45.9])
        with self.assertRaises(ValueError):
            bbox_to_working([-.1, 0, 1, 1], 80, 51)

    def test_match_requires_class_correct_target_and_deterministic_best_iou(self):
        target = {"class_name": "trousers", "bbox": [4, 0, 8, 4]}
        predictions = [
            {"garment_id": "wrong_class", "class_name": "skirt", "bbox": target["bbox"]},
            {"garment_id": "wrong_item", "class_name": "trousers", "bbox": [0, 0, 4, 4]},
            {"garment_id": "z", "class_name": "trousers", "bbox": target["bbox"]},
            {"garment_id": "a", "class_name": "trousers", "bbox": target["bbox"]},
        ]
        for values in (predictions, list(reversed(predictions))):
            row = score_target(target, values, (8, 4))
            self.assertTrue(row["matched"])
            self.assertEqual(row["garment_id"], "a")
        self.assertFalse(score_target(target, predictions[:2], (8, 4))["matched"])
        self.assertFalse(score_target(target, [], (8, 4))["matched"])

    def test_mask_uses_canvas_and_does_not_hide_bbox_offset(self):
        target = {"class_name": "skirt", "bbox": [0, 0, 4, 4]}
        reference = np.zeros((4, 8), np.uint8)
        reference[:, :4] = 255
        prediction = {"class_name": "skirt", "bbox": [1, 0, 5, 4], "mask": np.ones((4, 4), np.uint8)}
        row = score_target(target, [prediction], (8, 4), reference)
        self.assertTrue(row["matched"])
        self.assertAlmostEqual(row["mask_iou_conditional"], 3 / 5)
        self.assertAlmostEqual(row["mask_iou_end_to_end"], 3 / 5)
        np.testing.assert_array_equal(place_roi_mask(prediction["mask"], prediction["bbox"], (8, 4))[:, 1:5], True)
        for bbox, mask in (([0, 0, 5, 4], prediction["mask"]), ([-1, 0, 3, 4], prediction["mask"])):
            with self.assertRaises(ValueError):
                place_roi_mask(mask, bbox, (8, 4))

    def test_miss_fallback_and_off_keep_reference_denominator(self):
        target = {"class_name": "skirt", "bbox": [0, 0, 4, 4]}
        reference = np.ones((4, 4), np.uint8)
        for predictions in ([], [{"class_name": "skirt", "bbox": target["bbox"], "mask": None}]):
            row = score_target(target, predictions, (4, 4), reference)
            self.assertIsNone(row["mask_iou_conditional"])
            self.assertEqual(row["mask_iou_end_to_end"], 0)
        self.assertIsNone(score_target(target, [], (4, 4))["mask_iou_end_to_end"])

    def test_binary_mask_shape_values_empty_union(self):
        self.assertEqual(mask_iou(np.zeros((2, 2)), np.zeros((2, 2))), 0)
        with self.assertRaises(ValueError):
            mask_iou(np.ones((2, 2)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            mask_iou(np.full((2, 2), 128), np.ones((2, 2)))

    def test_reference_exif_once_and_nearest_resize(self):
        geometry = dict(raw_width=6, raw_height=4, oriented_width=4, oriented_height=6,
                        width=2, height=3, exif_orientation=6)
        raw = np.zeros((4, 6), np.uint8)
        raw[:2, :2] = 255
        actual = normalize_reference_mask(raw, "raw_image", geometry)
        expected = np.array([[0, 255], [0, 0], [0, 0]], dtype=bool)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(normalize_reference_mask(expected, "m1_working_image", geometry), expected)
        with self.assertRaises(ValueError):
            normalize_reference_mask(raw, "m1_working_image", geometry)
        with self.assertRaises(ValueError):
            normalize_reference_mask(raw, "unknown", geometry)

    def test_summary_separates_pending_and_retains_errors_and_zero_denominators(self):
        rows = [dict(label_status="human_reviewed", matched=True, mask_delivered=True,
                     mask_iou_conditional=.5, mask_iou_end_to_end=.5, roi_status="masked"),
                dict(label_status="human_reviewed", matched=False, mask_delivered=False,
                     mask_iou_conditional=None, mask_iou_end_to_end=0, error="inference_failed"),
                dict(label_status="pending_human_review", matched=True, mask_delivered=True,
                     mask_iou_conditional=1, mask_iou_end_to_end=1)]
        report = summarize_rows(rows)
        self.assertEqual(report["reviewed"]["target_count"], 2)
        self.assertEqual(report["reviewed"]["target_detection_recall"], .5)
        self.assertEqual(report["reviewed"]["mask_iou_end_to_end_mean"], .25)
        self.assertEqual(report["reviewed"]["mask_iou_conditional_mean"], .5)
        self.assertEqual(report["reviewed"]["errors"], 1)
        self.assertEqual(report["unreviewed_diagnostic"]["target_count"], 1)
        self.assertIsNone(summarize_rows([])["reviewed"]["target_detection_recall"])


if __name__ == "__main__":
    unittest.main()
