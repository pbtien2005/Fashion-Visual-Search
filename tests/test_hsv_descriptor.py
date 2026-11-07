import base64
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from AI.color_cache import build_color_cache, descriptor_cache_key, read_cached_descriptor
from AI.features import color_input_from_detection, extract_detection_features, extract_features
from AI.hsv_descriptor import (HSVConfig, extract_hsv_descriptor, histogram_similarity,
                               validate_descriptor)
from evaluation.run_color_benchmark import benchmark, synthetic_records, validate_records


def solid(rgb, size=(20, 20)):
    return np.full((*size, 3), rgb, np.uint8)


def png64(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class HistogramTests(unittest.TestCase):
    def test_dimensions_l1_and_metadata(self):
        for achromatic, dimension in ((False, 144), (True, 147)):
            config = HSVConfig(achromatic=achromatic)
            descriptor = extract_hsv_descriptor(solid((255, 0, 0)), config=config)
            self.assertEqual(len(descriptor.values), dimension)
            self.assertAlmostEqual(sum(descriptor.values), 1)
            self.assertEqual(descriptor.config["h_range"], [0, 180])
            self.assertEqual(descriptor.config["s_range"], [0, 256])
            self.assertEqual(descriptor.mask_status, "missing")
            self.assertEqual(descriptor.fallback_reason, "mask_missing")

    def test_white_on_white_mask_preserves_garment(self):
        image = solid((255, 255, 255))
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[5:15, 5:15] = 255
        descriptor = extract_hsv_descriptor(image, mask)
        self.assertEqual(descriptor.values[-1], 1)
        self.assertEqual(descriptor.valid_pixels, 100)
        self.assertEqual(descriptor.mask_fraction, .25)

    def test_white_with_small_skin_region(self):
        image = solid((245, 245, 245))
        image[:2, :2] = (210, 150, 110)
        descriptor = extract_hsv_descriptor(image)
        self.assertAlmostEqual(descriptor.values[-1], .99)
        self.assertAlmostEqual(sum(descriptor.values[:-3]), .01)

    def test_gray_black_low_saturation(self):
        for rgb, group in (((125, 125, 125), -2), ((15, 15, 15), -3), ((150, 140, 140), -2)):
            with self.subTest(rgb=rgb):
                descriptor = extract_hsv_descriptor(solid(rgb))
                self.assertEqual(descriptor.values[group], 1)
                self.assertEqual(sum(descriptor.values[:-3]), 0)

    def test_dark_saturated_fabric_retains_hue(self):
        descriptor = extract_hsv_descriptor(solid((3, 8, 30)))
        self.assertEqual(sum(descriptor.values[-3:]), 0)
        self.assertEqual(sum(descriptor.values[:-3]), 1)

    def test_threshold_boundaries_mutually_exclusive(self):
        config = HSVConfig()
        hsv = np.array([[[0, 32, 40], [0, 32, 41], [0, 32, 220],
                         [0, 33, 30], [0, 33, 221]]], np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # Compare against actual round-tripped uint8 HSV, including rounding.
        converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0]
        descriptor = extract_hsv_descriptor(rgb, config=config)
        self.assertAlmostEqual(sum(descriptor.values), 1)
        expected = ((converted[:, 1] <= 32) & (converted[:, 2] <= 40)).sum() / 5
        self.assertAlmostEqual(descriptor.values[-3], expected)

    def test_multicolor_mass_no_double_counting(self):
        image = solid((255, 0, 0))
        image[:, :10] = (0, 0, 255)
        descriptor = extract_hsv_descriptor(image)
        self.assertEqual(sorted(x for x in descriptor.values if x), [.5, .5])

    def test_rgb_bgr_conversion_is_explicit(self):
        red = extract_hsv_descriptor(solid((255, 0, 0)))
        blue = extract_hsv_descriptor(solid((0, 0, 255)))
        self.assertEqual(histogram_similarity(red, blue), 0)
        bgr_red = solid((0, 0, 255))
        converted = cv2.cvtColor(bgr_red, cv2.COLOR_BGR2RGB)
        self.assertAlmostEqual(histogram_similarity(red, extract_hsv_descriptor(converted)), 1)

    def test_mask_excludes_background(self):
        image = solid((0, 0, 255))
        image[5:15, 5:15] = (255, 0, 0)
        mask = np.zeros((20, 20), bool)
        mask[5:15, 5:15] = True
        descriptor = extract_hsv_descriptor(image, mask)
        red = extract_hsv_descriptor(solid((255, 0, 0)))
        self.assertAlmostEqual(histogram_similarity(descriptor, red), 1)

    def test_masks_missing_empty_invalid_and_insufficient(self):
        image = solid((255, 255, 255))
        small = np.zeros((20, 20), bool)
        small[0, 0] = True
        cases = [(None, "missing"), (np.zeros((20, 20), bool), "empty"),
                 (np.zeros((3, 3), bool), "invalid"), (np.ones((20, 20), float), "invalid"),
                 (np.full((20, 20), 2, np.uint8), "invalid"), (small, "insufficient")]
        for mask, status in cases:
            with self.subTest(status=status):
                cfg = HSVConfig(min_pixels=2)
                descriptor = extract_hsv_descriptor(image, mask, cfg)
                self.assertEqual(descriptor.mask_status, status)
                self.assertEqual(descriptor.valid_pixels, 400)
                empty = extract_hsv_descriptor(image, mask, replace(cfg, mask_policy="empty"))
                self.assertEqual(sum(empty.values), 0)
                with self.assertRaises(ValueError):
                    extract_hsv_descriptor(image, mask, replace(cfg, mask_policy="error"))

    def test_mask_area_upper_bound(self):
        descriptor = extract_hsv_descriptor(solid((255, 0, 0)), np.ones((20, 20), bool),
                                            HSVConfig(max_mask_fraction=.9))
        self.assertEqual(descriptor.mask_status, "insufficient")

    def test_input_validation(self):
        for image in (None, [], np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8),
                      np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), float)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                extract_hsv_descriptor(image)

    def test_config_validation(self):
        for kwargs in ({"hue_bins": 0}, {"saturation_bins": 1.2}, {"mask_policy": "guess"},
                       {"black_v_max": 230}, {"achromatic_s_max": -1},
                       {"min_mask_fraction": float("nan")}, {"min_pixels": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HSVConfig(**kwargs)

    def test_hellinger_formula_and_intersection(self):
        red = extract_hsv_descriptor(solid((255, 0, 0)))
        mixed = solid((255, 0, 0))
        mixed[:, :10] = (0, 0, 255)
        other = extract_hsv_descriptor(mixed)
        self.assertAlmostEqual(histogram_similarity(red, other), 1 - np.sqrt(1 - np.sqrt(.5)))
        self.assertEqual(histogram_similarity(red, other, "intersection"), .5)
        self.assertAlmostEqual(histogram_similarity(red, other), histogram_similarity(other, red))

    def test_empty_similarity_is_zero(self):
        empty = extract_hsv_descriptor(solid((0, 0, 0)), config=HSVConfig(mask_policy="empty"))
        self.assertEqual(histogram_similarity(empty, empty), 0)

    def test_reject_invalid_descriptors(self):
        original = extract_hsv_descriptor(solid((255, 0, 0)))
        for value in (float("nan"), float("inf"), -1, 2):
            invalid = deepcopy(original)
            invalid.values[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                histogram_similarity(original, invalid)
        invalid = deepcopy(original)
        invalid.values.pop()
        with self.assertRaises(ValueError):
            histogram_similarity(original, invalid)
        invalid = deepcopy(original)
        invalid.config["normalization"] = "L2"
        with self.assertRaises(ValueError):
            validate_descriptor(invalid)

    def test_version_includes_thresholds_bins_and_policy(self):
        config = HSVConfig()
        original = extract_hsv_descriptor(solid((0, 0, 0)), config=config)
        for cfg in (replace(config, achromatic=False), replace(config, hue_bins=12),
                    replace(config, black_v_max=39), replace(config, mask_policy="empty")):
            self.assertNotEqual(config.version, cfg.version)
            other = extract_hsv_descriptor(solid((0, 0, 0)), config=cfg)
            with self.assertRaises(ValueError):
                histogram_similarity(original, other)


class IntegrationTests(unittest.TestCase):
    def test_adapter_prefers_source_roi_to_legacy_crop(self):
        payload = {"source_roi_base64": png64(solid((255, 0, 0), (7, 11))),
                   "image_base64": png64(solid((0, 0, 255))),
                   "mask_base64": png64(np.ones((7, 11), np.uint8) * 255)}
        roi, mask, metadata = color_input_from_detection(payload)
        self.assertEqual(roi.shape, (7, 11, 3))
        self.assertEqual(mask.shape, (7, 11))
        self.assertEqual(metadata["roi_source"], "source_roi_base64")

    def test_adapter_reconstructs_source_bbox(self):
        original = solid((255, 0, 0), (30, 40))
        roi, mask, metadata = color_input_from_detection({"bbox": [2, 3, 12, 16]}, original)
        self.assertEqual(roi.shape, (13, 10, 3))
        self.assertEqual(metadata["roi_source"], "original_image_bbox")
        with self.assertRaises(ValueError):
            color_input_from_detection({"bbox": [-1, 0, 8, 8]}, original)

    def test_adapter_legacy_and_invalid_mask_are_recorded(self):
        roi, mask, metadata = color_input_from_detection(
            {"image_base64": png64(solid((255, 255, 255))), "mask_base64": "bad"})
        descriptor = extract_hsv_descriptor(roi, mask, metadata=metadata)
        self.assertEqual(descriptor.roi_source, "legacy_processed_crop")
        self.assertEqual(descriptor.metadata["roi_fallback_reason"], "source_roi_unavailable")
        self.assertEqual(descriptor.mask_status, "invalid")

    def test_unified_result_keeps_encoder_input_separate(self):
        vector = [1.] + [0.] * 511
        image = Image.new("RGB", (224, 224))
        with patch("AI.embedding_model.encode_image", return_value=vector) as encoder:
            result = extract_features("shirt-1", image, solid((255, 255, 255)))
        self.assertIs(encoder.call_args.args[0], image)
        self.assertEqual(result.color.values[-1], 1)
        self.assertEqual(len(result.embedding), 512)
        self.assertEqual(result.embedding_model, "patrickjohncyh/fashion-clip")
        json.dumps(result.to_dict(), allow_nan=False)

    def test_encoder_failure_propagates(self):
        with patch("AI.embedding_model.encode_image", side_effect=RuntimeError("encoder failed")):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                extract_features("a", Image.new("RGB", (224, 224)), solid((1, 2, 3)))

    def test_detection_entrypoint_uses_original_for_color_and_crop_for_encoder(self):
        detection = {"bbox": [1, 1, 11, 11], "class_name": "short_sleeve_top",
                     "image_base64": png64(solid((0, 0, 255), (224, 224)))}
        with patch("AI.embedding_model.encode_image", return_value=[1.] + [0.] * 511) as encoder:
            result = extract_detection_features("garment", detection, solid((255, 255, 255)))
        self.assertEqual(encoder.call_args.args[0].size, (224, 224))
        self.assertEqual(encoder.call_args.args[0].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result.color.values[-1], 1)
        self.assertEqual(result.color.valid_pixels, 100)
        self.assertEqual(result.color.roi_source, "original_image_bbox")

    def test_embedding_validation(self):
        from AI.embedding_model import validate_embedding_dimension
        validate_embedding_dimension([1.] + [0.] * 511)
        for vector in ([0.] * 511, [0.] * 512, [1.] * 512,
                       [float("nan")] + [0.] * 511, [float("inf")] + [0.] * 511):
            with self.subTest(first=vector[0]), self.assertRaises(ValueError):
                validate_embedding_dimension(vector)

    def test_cache_coverage_versions_and_no_catalog_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "output") as folder:
            root = Path(folder)
            Image.fromarray(solid((255, 255, 255))).save(root / "a.png")
            records = [{"id": "a", "image": "a.png", "mask": "absent.png"},
                       {"id": "b", "image": "missing.png"}]
            original = deepcopy(records)
            cache = build_color_cache(records, root)
            self.assertEqual(records, original)
            self.assertEqual(cache["coverage"]["fraction"], .5)
            self.assertEqual(len(cache["failures"]), 1)
            entry = next(iter(cache["entries"].values()))
            self.assertEqual(entry["descriptor"]["mask_status"], "invalid")
            self.assertNotIn("embedding", entry)
            key = next(iter(cache["entries"]))
            descriptor = read_cached_descriptor(cache, key, "patrickjohncyh/fashion-clip", 2,
                                                "source_roi_v1", cache["color_version"])
            self.assertEqual(descriptor.values[-1], 1)
            with self.assertRaises(ValueError):
                read_cached_descriptor(cache, key, "patrickjohncyh/fashion-clip", 3,
                                       "source_roi_v1", cache["color_version"])
            strict = build_color_cache(records, root, HSVConfig(mask_policy="error"))
            self.assertEqual(len(strict["entries"]), 0)

    def test_cache_key_separates_all_contract_fields(self):
        args = ["p", "model", 2, "roi", "hsv", {"image": "hash"}]
        key = descriptor_cache_key(*args)
        for i, change in enumerate(("q", "other", 3, "crop", "other-hsv", {"image": "changed"})):
            changed = args.copy()
            changed[i] = change
            self.assertNotEqual(key, descriptor_cache_key(*changed))

    def test_benchmark_factorial_and_failure_denominator(self):
        catalog, queries = synthetic_records()
        queries = queries[:2] + [{"id": "missing", "relevant_ids": ["white"], "load_error": "missing image"}]
        report = benchmark(catalog, queries)
        self.assertEqual(len(report["variants"]), 7)
        self.assertEqual(len(report["paired_ablation"]), 5)
        for variant in report["variants"].values():
            self.assertEqual(len(variant["per_query"]), 3)
            self.assertAlmostEqual(variant["query_coverage"], 2/3)
            self.assertEqual(variant["per_query"][-1]["reciprocal_rank"], 0)
            self.assertEqual(variant["failures"][-1]["id"], "missing")

    def test_benchmark_requires_labels_and_unique_ids(self):
        catalog, queries = synthetic_records()
        with self.assertRaises(ValueError):
            validate_records(catalog + [catalog[0]], queries)
        with self.assertRaises(ValueError):
            validate_records(catalog, [{"id": "unlabeled"}])

    def test_actual_encoder_path_normalizes_and_validates_outputs(self):
        import torch
        from AI.embedding_model import encode_image

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def __call__(self, **kwargs):
                return Inputs()

        class Model:
            def __init__(self, values):
                self.values = values

            def get_image_features(self, **kwargs):
                return torch.tensor([self.values], dtype=torch.float32)

        for values, valid in (([2.] + [0.] * 511, True), ([0.] * 512, False),
                              ([float("nan")] * 512, False), ([1.] * 511, False)):
            with self.subTest(valid=valid), patch("AI.embedding_model.load_embedding_model",
                                                return_value=(Model(values), Processor())):
                if valid:
                    self.assertEqual(encode_image(Image.new("RGB", (224, 224)))[0], 1)
                else:
                    with self.assertRaises(ValueError):
                        encode_image(Image.new("RGB", (224, 224)))


if __name__ == "__main__":
    unittest.main()
