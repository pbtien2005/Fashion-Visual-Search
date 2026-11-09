"""Contract tests, independent of production M2/M4 and pretrained model downloads."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from AI.color_cache import build_color_cache, catalog_descriptor_for_query
from AI.features import FeatureBundle, color_input_from_detection, extract_features
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor

ROOT = Path(__file__).resolve().parents[1]


class HandoffTests(unittest.TestCase):
    def bundle(self):
        roi = np.full((16, 24, 3), 245, np.uint8)
        with patch("AI.embedding_model.encode_image", return_value=[1.] + [0.] * 511):
            return extract_features("p1", Image.fromarray(roi), roi)

    def test_bundle_json_roundtrip_and_alias(self):
        bundle = self.bundle()
        restored = FeatureBundle.from_dict(json.loads(json.dumps(bundle.to_dict(), allow_nan=False)))
        self.assertEqual(bundle.to_dict(), restored.to_dict())
        self.assertEqual(restored.schema_version, "m3-features-v1")

    def test_bundle_rejects_invalid_serialized_contract(self):
        payload = self.bundle().to_dict()
        changes = [("schema_version", "other"), ("garment_id", ""),
                   ("embedding_version", 0), ("metadata", []),
                   ("embedding", [0.] * 512)]
        for key, value in changes:
            with self.subTest(key=key):
                modified = deepcopy(payload)
                modified[key] = value
                with self.assertRaises(ValueError):
                    FeatureBundle.from_dict(modified)
        payload["color"]["version"] = "other"
        with self.assertRaises(ValueError):
            FeatureBundle.from_dict(payload)
        for key, value in (("mask_status", "guessed"), ("valid_pixels", -1), ("total_pixels", 0)):
            modified = self.bundle().to_dict()
            modified["color"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                FeatureBundle.from_dict(modified)

    def test_invalid_image_payload_is_explicit_error(self):
        for payload in (None, "", "not-base64"):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Invalid RGB image"):
                color_input_from_detection({"image_base64": payload})

    def test_shared_profile_query_catalog_and_all_mismatches(self):
        config = HSVConfig(**json.loads((ROOT / "docs/m3_contract/hsv147.json").read_text()))
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            roi = np.full((16, 24, 3), 245, np.uint8)
            mask = np.zeros((16, 24), np.uint8)
            mask[2:14, 3:21] = 255
            Image.fromarray(roi).save(Path(folder) / "roi.png")
            Image.fromarray(mask).save(Path(folder) / "mask.png")
            records = [{"id": "p1", "image": "roi.png", "mask": "mask.png"}]
            cache = build_color_cache(records, folder, config)
            query = extract_hsv_descriptor(roi, mask, config)
            product = catalog_descriptor_for_query(cache, "p1", query, "patrickjohncyh/fashion-clip", 2)
            self.assertEqual(query.values, product.values)
            self.assertEqual(query.version, product.version)
            changed_config = extract_hsv_descriptor(roi, mask, replace(config, black_v_max=39))
            for candidate, model, version in ((changed_config, "patrickjohncyh/fashion-clip", 2),
                (replace(query, roi_policy="other"), "patrickjohncyh/fashion-clip", 2),
                (query, "other", 2), (query, "patrickjohncyh/fashion-clip", 3)):
                with self.subTest(model=model, version=version), self.assertRaises(ValueError):
                    catalog_descriptor_for_query(cache, "p1", candidate, model, version)
            with self.assertRaises(KeyError):
                catalog_descriptor_for_query(cache, "missing", query, "patrickjohncyh/fashion-clip", 2)
            ambiguous = deepcopy(cache)
            ambiguous["entries"]["extra"] = deepcopy(next(iter(cache["entries"].values())))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                catalog_descriptor_for_query(ambiguous, "p1", query, "patrickjohncyh/fashion-clip", 2)

    def test_binary_mask_payload_roundtrip_coordinates_and_fallback(self):
        roi = np.full((7, 11, 3), 255, np.uint8)
        mask = np.zeros((7, 11), np.uint8)
        mask[2:5, 3:9] = 255
        def encode(array):
            buffer = BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        image, decoded, metadata = color_input_from_detection(
            {"source_roi_base64": encode(roi), "mask_base64": encode(mask)})
        np.testing.assert_array_equal(decoded, mask)
        descriptor = extract_hsv_descriptor(image, decoded, metadata=metadata)
        self.assertEqual(descriptor.valid_pixels, 18)
        self.assertEqual(descriptor.mask_status, "valid")
        wrong_shape = extract_hsv_descriptor(image, np.ones((3, 3), np.uint8))
        self.assertEqual(wrong_shape.mask_status, "invalid")
        self.assertEqual(wrong_shape.fallback_reason, "mask_invalid")

    def test_mask_jpeg_is_rejected_by_png_contract(self):
        image_buffer, mask_buffer = BytesIO(), BytesIO()
        Image.new("RGB", (11, 7), "white").save(image_buffer, format="PNG")
        Image.new("L", (11, 7), 255).save(mask_buffer, format="JPEG")
        roi, mask, metadata = color_input_from_detection({
            "source_roi_base64": base64.b64encode(image_buffer.getvalue()).decode(),
            "mask_base64": base64.b64encode(mask_buffer.getvalue()).decode()})
        descriptor = extract_hsv_descriptor(roi, mask, metadata=metadata)
        self.assertEqual(descriptor.mask_status, "invalid")
        self.assertEqual(descriptor.fallback_reason, "mask_invalid")

    def test_batch_cache_cli_coverage_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            image = folder / "roi.png"
            Image.new("RGB", (11, 7), "white").save(image)
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"catalog": [
                {"id": "p1", "image": "roi.png"}, {"id": "p2", "image": "missing.png"}]}))
            output = folder / "new_cache.json"
            command = [sys.executable, "-B", "-m", "AI.color_cache", "--manifest", str(manifest),
                       "--config", str(ROOT / "docs/m3_contract/hsv147.json"), "--output", str(output)]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            original = output.read_bytes()
            cache = json.loads(original)
            self.assertEqual(cache["coverage"]["extracted"], 1)
            self.assertEqual(len(cache["failures"]), 1)
            retry = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(len(json.loads(manifest.read_text())["catalog"]), 2)


if __name__ == "__main__":
    unittest.main()
