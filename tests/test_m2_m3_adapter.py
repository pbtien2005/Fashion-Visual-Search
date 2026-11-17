import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from AI.features import color_input_from_detection, detection_identity
from AI.hsv_descriptor import extract_hsv_descriptor


FIXTURES = Path(__file__).parent / "fixtures/m2"


def png(array, format="PNG"):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format=format)
    return base64.b64encode(stream.getvalue()).decode()


class M2M3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.crop = json.loads((FIXTURES / "candidate_response_masked.json").read_text(encoding="utf-8"))["crops"][0]
        self.expected_roi = np.asarray(Image.open(FIXTURES / "source_roi.png").convert("RGB"))
        self.expected_mask = np.asarray(Image.open(FIXTURES / "reference_mask.png"))

    def test_canonical_png_beats_upload_legacy_bbox_and_aliases(self):
        crop = dict(self.crop, original_bbox=[0, 0, 1, 1], source_roi_base64=png(np.zeros((2, 2, 3), np.uint8)),
                    mask_base64=png(np.zeros((2, 2), np.uint8)))
        roi, mask, metadata = color_input_from_detection(crop, np.zeros((3, 3, 3), np.uint8))
        np.testing.assert_array_equal(roi, self.expected_roi)
        np.testing.assert_array_equal(mask, self.expected_mask)
        self.assertEqual(metadata["roi_source"], "source_crop_png_base64")
        self.assertEqual(extract_hsv_descriptor(roi, mask).mask_status, "valid")

    def test_null_canonical_mask_does_not_reuse_stale_alias_mask(self):
        crop = dict(self.crop, mask_png_base64=None, mask_base64=png(self.expected_mask))
        _, mask, _ = color_input_from_detection(crop)
        self.assertIsNone(mask)

    def test_bad_canonical_source_is_not_hidden_by_valid_alias(self):
        for value in ("", "invalid", png(self.expected_roi, "JPEG")):
            with self.subTest(value=value[:10]):
                with self.assertRaises(ValueError):
                    color_input_from_detection(dict(self.crop, source_crop_png_base64=value,
                                                   source_roi_base64=png(self.expected_roi)))

    def test_m2_metadata_and_ids_are_preserved_without_overriding_m3_policy(self):
        crop = deepcopy(self.crop)
        crop["roi_metadata"].update(roi_policy="wrong", segmentation={"candidate": True})
        _, _, metadata = color_input_from_detection(crop)
        self.assertEqual(metadata["roi_policy"], "source_roi_v1")
        self.assertEqual(metadata["garment_id"], crop["garment_id"])
        self.assertEqual(metadata["m2_roi_metadata"]["mask_config_id"], crop["roi_metadata"]["mask_config_id"])
        metadata["m2_roi_metadata"]["segmentation"]["candidate"] = False
        self.assertTrue(crop["roi_metadata"]["segmentation"]["candidate"])

    def test_malformed_nonbinary_rgb_and_wrong_shape_masks_follow_hsv_policy(self):
        masks = ["bad", png(np.full((64, 64), 128, np.uint8)), png(self.expected_roi), png(np.zeros((3, 3), np.uint8))]
        for payload in masks:
            with self.subTest(payload=payload[:10]):
                roi, mask, _ = color_input_from_detection(dict(self.crop, mask_png_base64=payload))
                descriptor = extract_hsv_descriptor(roi, mask)
                self.assertEqual(descriptor.mask_status, "invalid")
                self.assertEqual(descriptor.valid_pixels, 4096)

    def test_inconsistent_roi_metadata_is_rejected(self):
        crop = deepcopy(self.crop)
        crop["roi_metadata"]["roi_width"] = 63
        with self.assertRaises(ValueError):
            color_input_from_detection(crop)

    def test_working_bbox_is_not_cut_from_an_unverified_original_frame(self):
        crop = {"bbox": [0, 0, 20, 20], "roi_metadata": {"geometry_mode": "m1_resized"}}
        with self.assertRaises(ValueError):
            color_input_from_detection(crop, np.zeros((40, 40, 3), np.uint8))

    def test_legacy_alias_payload_and_identity_fallback_remain_supported(self):
        aliases = dict(source_roi_base64=self.crop["source_crop_png_base64"], mask_base64=self.crop["mask_png_base64"])
        roi, mask, _ = color_input_from_detection(aliases)
        np.testing.assert_array_equal(roi, self.expected_roi)
        np.testing.assert_array_equal(mask, self.expected_mask)
        self.assertEqual(detection_identity(dict(garment_id="q_g0", id="old", class_name="shirt")), "q_g0")
        self.assertEqual(detection_identity(dict(id="old")), "old")
        self.assertEqual(detection_identity({}), "query")


if __name__ == "__main__":
    unittest.main()
