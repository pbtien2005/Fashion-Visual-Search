"""Production routing/HTTP tests: legacy color functions must never be called."""
import ast
import asyncio
import base64
from contextlib import ExitStack
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image
import httpx

from AI.color_cache import cache_from_extracted_items
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor
from AI.production_color import (ColorStore, ROI_POLICY, descriptor_display,
    extract_source_color, product_color, production_config)
from AI.vectorizer import process_single_item, get_product_crop_detection_url
from api.main import app
from api.search_service import get_rerank_weights, rerank_products, search_match_for_crop

ROOT = Path(__file__).resolve().parents[1]


def image(rgb, size=(30, 40)):
    return np.full((*size, 3), rgb, dtype=np.uint8)


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(array):
    return base64.b64encode(png(array)).decode()


def descriptor(rgb):
    return extract_hsv_descriptor(image(rgb))


def cache_for(colors):
    return cache_from_extracted_items([
        {"id": key, "color_descriptor": color.to_dict() if color else None} for key, color in colors.items()
    ], HSVConfig(), "patrickjohncyh/fashion-clip", 2, ROI_POLICY)


class ProductionHSVTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"HSV_MISSING_POLICY": "zero", "HSV_CONFIG_PATH": "",
            "ENABLE_TEXT_STYLE_RERANK": "false", "USE_PRODUCT_CROP_EMBEDDING": "false"})
        env.start()
        self.addCleanup(env.stop)
        guards = ExitStack()
        for name in ("extract_dominant_color", "color_similarity", "name_color"):
            guards.enter_context(patch(f"AI.color_utils.{name}", side_effect=AssertionError("Legacy RGB called")))
        self.addCleanup(guards.close)
        self.vector = [1.] + [0.] * 511
        self.red, self.blue = descriptor((220, 20, 20)), descriptor((20, 20, 220))

    def test_display_neutrals_and_hues_are_derived_from_descriptor(self):
        for rgb, name in (((255, 255, 255), "white"), ((128, 128, 128), "gray"),
                          ((0, 0, 0), "black"), ((255, 0, 0), "red"), ((0, 0, 255), "blue")):
            with self.subTest(name=name):
                swatch, label = descriptor_display(descriptor(rgb))
                self.assertEqual(label, name)
                self.assertEqual(len(swatch), 3)

    def test_source_roi_is_shared_across_encoder_variants_and_keeps_white(self):
        original = image((255, 255, 255))
        crop = {"bbox": [5, 4, 25, 24], "image_base64": b64(image((255, 0, 0))),
                "class_name": "short_sleeve_top", "crop_variants": [
                    {"name": "tight", "image_base64": b64(image((255, 0, 0)))},
                    {"name": "wide", "image_base64": b64(image((0, 0, 255)))}]}
        white = descriptor((255, 255, 255))
        with patch("AI.embedding_model.encode_image", return_value=self.vector) as encoder, \
             patch("api.search_service.search_similar_by_vector", return_value=[
                 {"id": "white", "title": "White shirt", "category": "shirt", "similarity": .95}]), \
             patch("AI.production_color.configured_color_store", return_value=ColorStore(cache_for({"white": white}), HSVConfig())):
            result = search_match_for_crop(crop, original_rgb=original)
        self.assertEqual(encoder.call_count, 2)
        self.assertEqual(result["query_color_name"], "white")
        self.assertEqual(result["query_color_rgb"], [240, 240, 240])
        self.assertAlmostEqual(result["similar_products"][0]["color_score"], 1)

    def test_catalog_runtime_features_preserve_encoder_source_and_use_raw_roi(self):
        original = image((255, 255, 255))
        staged = {"bbox": [5, 4, 25, 24], "image_base64": b64(image((0, 0, 255)))}
        item = {"id": "p1", "r2_image_url": "https://example.invalid/p1.jpg",
                "dominant_color_name": "red", "dominant_color_rgb": [255, 0, 0]}
        with patch("AI.vectorizer.fetch_image_for_ai", return_value=Image.fromarray(original)), \
             patch("AI.vectorizer.detect_product_crop", return_value=staged), \
             patch("AI.embedding_model.encode_image", return_value=self.vector) as encoder:
            result = process_single_item(item)
        self.assertEqual(item["dominant_color_name"], "red")  # Caller not mutated.
        self.assertEqual(encoder.call_args.args[0].size, (40, 30))
        self.assertEqual(result["color_descriptor"]["roi_source"], "original_image_bbox")
        self.assertEqual(result["dominant_color_name"], "white")
        self.assertEqual(len(result["color_descriptor"]["values"]), 147)
        self.assertEqual(result["embedding"], self.vector)
        query = extract_source_color(original, staged)
        cache = cache_from_extracted_items([result], production_config(), "patrickjohncyh/fashion-clip", 2, ROI_POLICY)
        stored = ColorStore(cache, production_config()).get("p1", "patrickjohncyh/fashion-clip", 2)
        self.assertEqual(query.values, stored.values)
        self.assertEqual(query.version, stored.version)

    def test_opt_in_catalog_crop_embedding_stays_opt_in(self):
        original = image((255, 255, 255))
        staged = {"bbox": [0, 0, 20, 20], "image_base64": b64(image((0, 0, 255), (224, 224)))}
        with patch.dict(os.environ, {"USE_PRODUCT_CROP_EMBEDDING": "true"}), \
             patch("AI.vectorizer.fetch_image_for_ai", return_value=Image.fromarray(original)), \
             patch("AI.vectorizer.detect_product_crop", return_value=staged), \
             patch("AI.embedding_model.encode_image", return_value=self.vector) as encoder:
            result = process_single_item({"id": "p1", "r2_image_url": "fixture"})
        self.assertEqual(encoder.call_args.args[0].size, (224, 224))
        self.assertEqual(result["dominant_color_name"], "white")
        with patch.dict(os.environ, {"OBJECT_DETECTION_URL": "http://object-detection:8001", "PRODUCT_CROP_DETECTION_URL": ""}):
            self.assertEqual(get_product_crop_detection_url(), "http://object-detection:8001")

    def test_upload_decoder_matches_m2_coordinates_with_exif_orientation(self):
        import cv2
        from AI.production_color import decode_uploaded_rgb
        source = Image.new("RGB", (20, 30), "red")
        exif = source.getexif()
        exif[274] = 6
        buffer = BytesIO()
        source.save(buffer, format="JPEG", exif=exif)
        payload = buffer.getvalue()
        expected_bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        actual_rgb = decode_uploaded_rgb(payload)
        self.assertEqual(actual_rgb.shape, (20, 30, 3))
        np.testing.assert_array_equal(actual_rgb, cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB))

    def test_reranking_uses_histogram_and_ignores_contradictory_db_rgb(self):
        products = [{"id": "red", "title": "Shirt", "similarity": .9, "category": "shirt",
                     "dominant_color_rgb": [0, 0, 255], "dominant_color_name": "blue"},
                    {"id": "blue", "title": "Shirt", "similarity": .9, "category": "shirt",
                     "dominant_color_rgb": [255, 0, 0], "dominant_color_name": "red"}]
        store = ColorStore(cache_for({"red": self.red, "blue": self.blue}), HSVConfig())
        with patch("AI.production_color.configured_color_store", return_value=store):
            ranked = rerank_products(products, ["shirt"], 2, query_color=self.red)
        self.assertEqual([p["id"] for p in ranked], ["red", "blue"])
        self.assertAlmostEqual(ranked[0]["color_score"], 1)
        self.assertEqual(ranked[1]["color_score"], 0)
        self.assertEqual(ranked[0]["dominant_color_name"], "red")
        self.assertEqual(ranked[1]["color_penalty"], 0)
        self.assertEqual(get_rerank_weights(), (.6, .15, .15, .1))

    def test_missing_cache_color_has_zero_contribution_without_weight_redistribution(self):
        with patch("AI.production_color.configured_color_store", side_effect=FileNotFoundError("not-built")), \
             self.assertLogs("AI.production_color", level="WARNING"):
            result = rerank_products([{"id": "p1", "similarity": .9, "dominant_color_name": "red"}], [], 1,
                                     query_color=self.red)
        self.assertEqual(result[0]["color_score"], 0)
        self.assertIsNone(result[0]["dominant_color_name"])
        self.assertAlmostEqual(result[0]["final_score"], .9 * (.6 / .9))

    def test_missing_policy_error_rejects_cache_and_extraction_failure(self):
        with patch.dict(os.environ, {"HSV_MISSING_POLICY": "error"}):
            with self.assertRaises(ValueError):
                extract_source_color(None, {"image_base64": "bad"})
            with patch("AI.production_color.configured_color_store", side_effect=FileNotFoundError()):
                with self.assertRaises(ValueError):
                    product_color("p", "patrickjohncyh/fashion-clip", 2)

    def test_invalid_color_never_falls_back_to_rgb_and_encoder_remains_valid(self):
        staged = {"source_roi_base64": "bad", "image_base64": b64(image((255, 0, 0)))}
        with patch("AI.vectorizer.fetch_image_for_ai", return_value=Image.fromarray(image((255, 0, 0)))), \
             patch("AI.vectorizer.detect_product_crop", return_value=staged), \
             patch("AI.embedding_model.encode_image", return_value=self.vector), \
             self.assertLogs("AI.production_color", level="WARNING"):
            result = process_single_item({"id": "p", "r2_image_url": "fixture", "dominant_color_name": "red"})
        self.assertIsNone(result["color_descriptor"])
        self.assertIsNone(result["dominant_color_name"])
        self.assertEqual(result["embedding"], self.vector)

    def test_cache_rejects_wrong_config_version_policy_nan_and_model(self):
        for field, value in (("version", "other"), ("roi_policy", "other"),
                             ("values", [float("nan")] * 147), ("valid_pixels", -1)):
            cache = cache_for({"p": self.red})
            entry = next(iter(cache["entries"].values()))
            entry["descriptor"][field] = value
            store = ColorStore(cache, HSVConfig())
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                store.get("p", "patrickjohncyh/fashion-clip", 2)
        with self.assertRaises(ValueError):
            ColorStore(cache_for({"p": self.red}), replace(HSVConfig(), black_v_max=39))
        with self.assertRaises(ValueError):
            ColorStore(cache_for({"p": self.red}), HSVConfig()).get("p", "other-model", 2)

    def test_direct_m4_call_missing_or_invalid_query_obeys_policy(self):
        with patch.dict(os.environ, {"HSV_MISSING_POLICY": "error"}):
            for color in (None, replace(self.red, roi_policy="other"), replace(self.red, valid_pixels=0)):
                with self.subTest(color=color is None), self.assertRaises(ValueError):
                    rerank_products([], [], 1, query_color=color)

    def test_non147_profile_is_rejected_and_disk_cache_reload_is_versioned(self):
        from AI.production_color import configured_color_store
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            profile = folder / "invalid.json"
            profile.write_text(json.dumps({"achromatic": False}))
            with patch.dict(os.environ, {"HSV_CONFIG_PATH": str(profile)}), self.assertRaises(ValueError):
                production_config()
            cache_file = folder / "cache.json"
            cache_file.write_text(json.dumps(cache_for({"p": self.red})), encoding="utf-8")
            with patch.dict(os.environ, {"HSV_COLOR_CACHE_PATH": str(cache_file)}):
                store = configured_color_store()
                self.assertEqual(store.get("p", "patrickjohncyh/fashion-clip", 2).values, self.red.values)
                cache_file.write_text('{"schema":"wrong"}', encoding="utf-8")
                with self.assertRaises(ValueError):
                    configured_color_store()

    def test_rebuild_function_never_calls_encoder(self):
        from AI.rebuild_color_cache import main
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            Image.fromarray(image((255, 255, 255))).save(folder / "p.png")
            catalog = folder / "catalog.json"
            catalog.write_text(json.dumps([{"id": "p", "image": "p.png"}]))
            with patch("sys.argv", ["rebuild", "--catalog", str(catalog), "--output", str(folder / "new.json"), "--skip-detection"]), \
                 patch("AI.embedding_model.encode_image", side_effect=AssertionError("Re-embedding forbidden")):
                main()
            cache = json.loads((folder / "new.json").read_text())
            self.assertEqual(cache["coverage"]["extracted"], 1)

    def test_http_search_and_listing_use_hsv_and_keep_api_schema(self):
        original = image((255, 255, 255))
        crop = {"bbox": [1, 1, 20, 20], "confidence": .95, "class_name": "short_sleeve_top",
                "class_name_vi": "Áo", "image_base64": b64(image((255, 0, 0)))}
        product = {"id": "p1", "title": "White shirt", "category": "shirt", "similarity": .95,
                   "dominant_color_name": "red", "dominant_color_rgb": [255, 0, 0]}
        store = ColorStore(cache_for({"p1": descriptor((255, 255, 255))}), HSVConfig())
        with patch("api.main.DetectionClient.detect_upload", new=AsyncMock(return_value={"success": True, "crops": [crop]})), \
             patch("api.search_service.search_similar_by_vector", return_value=[product]), \
             patch("api.main.list_products", return_value={"items": [product], "total": 1}), \
             patch("AI.production_color.configured_color_store", return_value=store), \
             patch("AI.embedding_model.encode_image", return_value=self.vector):
            async def requests():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post("/api/search/upload", files={"file": ("query.png", png(original), "image/png")})
                    listing = await client.get("/api/products")
                    return response, listing
            response, listing = asyncio.run(requests())
        self.assertEqual(response.status_code, 200, response.text)
        detected = response.json()["detected_items"][0]
        self.assertEqual(detected["query_color_name"], "white")
        self.assertAlmostEqual(detected["similar_products"][0]["color_score"], 1)
        self.assertEqual(detected["similar_products"][0]["dominant_color_name"], "white")
        self.assertIn("color_penalty", detected["similar_products"][0])
        self.assertNotIn("_hsv_display", detected["similar_products"][0])
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["items"][0]["dominant_color_name"], "white")

    def test_production_modules_do_not_import_legacy_color_algorithms(self):
        for relative in ("AI/vectorizer.py", "AI/features.py", "AI/production_color.py",
                         "api/search_service.py", "api/main.py", "db/database.py",
                         "etl/backfill_product_colors.py", "db/backfill_color_names.py"):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            with self.subTest(module=relative):
                self.assertNotIn("AI.color_utils", imports)

    def test_display_backfill_uses_hsv_sidecar_not_db_rgb(self):
        from db.backfill_color_names import backfill_color_names
        connection, rows, update = MagicMock(), MagicMock(), MagicMock()
        rows.fetchall.return_value = [("p",)]
        update.rowcount = 1
        connection.execute.side_effect = [rows, update]
        with patch("db.backfill_color_names.get_db_connection", return_value=connection), \
             patch("db.backfill_color_names.product_color", return_value=self.red):
            backfill_color_names()
        params = connection.execute.call_args_list[1].args[1]
        self.assertEqual(params[0], "red")
        self.assertEqual(len(params[1]), 3)
        connection.commit.assert_called_once()

    def test_rebuild_cli_does_not_encode_or_modify_original_catalog(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            Image.fromarray(image((255, 255, 255))).save(folder / "p.png")
            catalog = folder / "catalog.json"
            original = json.dumps([{"id": "p", "image": "p.png", "embedding": self.vector},
                                   {"id": "missing", "image": "missing.png"}]).encode()
            catalog.write_bytes(original)
            output = folder / "new_cache.json"
            command = [sys.executable, "-B", "-m", "AI.rebuild_color_cache", "--catalog", str(catalog),
                       "--output", str(output), "--skip-detection"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            cache = json.loads(output.read_text())
            self.assertEqual(cache["coverage"]["total"], 2)
            self.assertEqual(cache["coverage"]["extracted"], 1)
            self.assertEqual(catalog.read_bytes(), original)
            entry = next(iter(cache["entries"].values()))
            self.assertNotIn("embedding", entry)
            self.assertEqual(entry["descriptor"]["roi_policy"], ROI_POLICY)
            retry = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)


if __name__ == "__main__":
    unittest.main()
