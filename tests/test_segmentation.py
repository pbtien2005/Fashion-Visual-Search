from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from object_detection.segmentation import MaskConfig, segment_roi


def garment(foreground=210, background=30):
    image = np.full((80, 80, 3), background, dtype=np.uint8)
    image[16:64, 16:64] = foreground
    return image


class SegmentationTests(unittest.TestCase):
    def config(self, mode="raw", **kwargs):
        return MaskConfig(mode=mode, **kwargs)

    def test_off_returns_null_and_does_not_run_otsu(self):
        with patch("cv2.threshold", side_effect=AssertionError("Otsu should not run")):
            result = segment_roi(garment(), MaskConfig())
        self.assertIsNone(result.mask)
        self.assertEqual(result.metadata["fallback_reason"], "mask_disabled")
        self.assertEqual(result.metadata["roi_status"], "unmasked")

    def test_bright_and_dark_foreground_are_selected_by_center_and_border(self):
        for foreground, background, polarity in [(210, 30, "bright"), (30, 210, "dark")]:
            with self.subTest(polarity=polarity):
                image = garment(foreground, background)
                before = image.copy()
                result = segment_roi(image, self.config())
                self.assertEqual(result.metadata["roi_status"], "masked")
                self.assertEqual(result.metadata["segmentation"]["polarity"], polarity)
                self.assertEqual(int(np.count_nonzero(result.mask)), 48 * 48)
                self.assertEqual(set(np.unique(result.mask)), {0, 255})
                np.testing.assert_array_equal(image, before)

    def test_white_on_white_and_small_roi_fall_back(self):
        for image, reason in [(np.full((80, 80, 3), 255, np.uint8), "low_contrast"),
                              (np.zeros((4, 4, 3), np.uint8), "roi_too_small")]:
            result = segment_roi(image, self.config())
            self.assertIsNone(result.mask)
            self.assertEqual(result.metadata["fallback_reason"], reason)

    def test_side_by_side_regions_are_ambiguous_not_forced_to_foreground(self):
        image = np.full((80, 80, 3), 30, np.uint8)
        image[:, 40:] = 210
        result = segment_roi(image, self.config())
        self.assertIsNone(result.mask)
        self.assertEqual(result.metadata["fallback_reason"], "ambiguous_foreground")

    def test_otsu_on_low_light_still_works_when_contrast_is_present(self):
        result = segment_roi(garment(55, 5), self.config())
        self.assertIsNotNone(result.mask)
        self.assertEqual(int(np.count_nonzero(result.mask)), 48 * 48)

    def test_raw_opening_closing_and_open_close_are_available(self):
        image = garment()
        image[5, 5] = 210
        image[30, 30] = 30
        outputs = {mode: segment_roi(image, self.config(mode)) for mode in ["raw", "opening", "closing", "open_close"]}
        for result in outputs.values():
            self.assertIsNotNone(result.mask)
            self.assertIsNotNone(result.raw_mask)
            self.assertIsNotNone(result.processed_mask)
        self.assertEqual(outputs["raw"].mask[5, 5], 255)
        self.assertEqual(outputs["opening"].mask[5, 5], 0)
        self.assertEqual(outputs["closing"].mask[30, 30], 255)

    def test_area_gate_and_large_morphology_changes_fall_back(self):
        result = segment_roi(garment(), self.config(min_area_fraction=0.5))
        self.assertIsNone(result.mask)
        self.assertEqual(result.metadata["fallback_reason"], "area_out_of_range")
        noisy = garment()
        noisy[5, 5] = 210
        result = segment_roi(noisy, self.config("opening", max_change_fraction=0))
        self.assertIsNone(result.mask)
        self.assertEqual(result.metadata["fallback_reason"], "morphology_unstable")

    def test_no_largest_component_or_hole_fill_is_implicitly_applied(self):
        image = garment()
        image[30:34, 30:34] = 30
        image[7:11, 7:11] = 210
        result = segment_roi(image, self.config())
        self.assertEqual(result.mask[31, 31], 0)
        self.assertEqual(result.mask[8, 8], 255)

    def test_opencv_failure_returns_controlled_mask_error(self):
        with patch("cv2.threshold", side_effect=cv2.error("fixture error")):
            result = segment_roi(garment(), self.config())
        self.assertIsNone(result.mask)
        self.assertEqual(result.metadata["fallback_reason"], "mask_processing_error")

    def test_empty_or_full_processed_mask_is_never_delivered(self):
        for value, reason in [(0, "empty_mask"), (255, "area_out_of_range")]:
            with self.subTest(value=value):
                with patch("object_detection.segmentation._morphology", return_value=np.full((80, 80), value, np.uint8)):
                    result = segment_roi(garment(), self.config("closing"))
                self.assertIsNone(result.mask)
                self.assertEqual(result.metadata["fallback_reason"], reason)

    def test_config_is_versioned_and_invalid_inputs_fail_explicitly(self):
        self.assertNotEqual(self.config().config_id, self.config("closing").config_id)
        self.assertNotEqual(self.config().config_id, replace(self.config(), min_contrast=20).config_id)
        for config in [dict(mode="sam"), dict(kernel_size=5), dict(min_area_fraction=1.1), dict(min_side=0),
                       dict(max_change_fraction=-1), dict(min_contrast=0), dict(min_area_fraction=.97)]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    MaskConfig(**config)
        with self.assertRaises(ValueError):
            segment_roi(np.zeros((80, 80), np.uint8), self.config())


if __name__ == "__main__":
    unittest.main()
