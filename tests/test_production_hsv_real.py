"""Opt-in real FashionCLIP integration, with staged M2 and DB transport fixtures.

Not a benchmark and not proof of a live YOLO/PostgreSQL deployment.
Set cache/offline flags before starting Python; no encoder or HSV mocks here.
"""
import asyncio
import base64
from contextlib import ExitStack
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.getenv("RUN_REAL_HSV_INTEGRATION") == "1", "Opt-in pretrained weights test")
class RealHSVProductionTests(unittest.TestCase):
    def test_actual_catalog_encoder_cache_and_http_query(self):
        from AI.vectorizer import process_single_item
        from AI.color_cache import cache_from_extracted_items
        from AI.production_color import ROI_POLICY, production_config
        from api.main import app

        path = ROOT / "evaluation/queries/starter_catalog/B0GGYQ4J5Z.jpg"
        with Image.open(path) as source:
            original = source.convert("RGB")
        width, height = original.size
        crop_buffer = BytesIO()
        ImageOps.pad(original, (224, 224), color="white").save(crop_buffer, format="PNG")
        crop = {"class_name": "short_sleeve_top", "class_name_vi": "Áo", "confidence": .99,
                "bbox": [0, 0, width, height],
                "image_base64": base64.b64encode(crop_buffer.getvalue()).decode()}
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"HSV_MISSING_POLICY": "error", "HSV_CONFIG_PATH": "",
                "HSV_COLOR_CACHE_PATH": str(Path(folder) / "cache.json"),
                "ENABLE_TEXT_STYLE_RERANK": "false", "USE_PRODUCT_CROP_EMBEDDING": "false"}))
            for name in ("extract_dominant_color", "color_similarity", "name_color"):
                stack.enter_context(patch(f"AI.color_utils.{name}", side_effect=AssertionError("RGB used")))
            stack.enter_context(patch("AI.vectorizer.fetch_image_for_ai", return_value=original))
            # Staged M2 output only. Production catalog path and both encoders run unchanged.
            record = process_single_item({"id": "fixture", "r2_image_url": "fixture", "detection": crop})
            self.assertEqual(len(record["embedding"]), 512)
            self.assertAlmostEqual(float(np.linalg.norm(record["embedding"])), 1, places=5)
            cache = cache_from_extracted_items([record], production_config(),
                                               record["embedding_model"], record["embedding_version"], ROI_POLICY)
            (Path(folder) / "cache.json").write_text(json.dumps(cache, allow_nan=False), encoding="utf-8")
            stack.enter_context(patch("api.main.DetectionClient.detect_upload",
                new=AsyncMock(return_value={"success": True, "crops": [crop]})))
            def candidates(vector, **kwargs):
                self.assertEqual(len(vector), 512)
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=5)
                return [{"id": "fixture", "title": "Shirt", "category": "shirt",
                         "similarity": float(np.dot(vector, record["embedding"]))}]
            stack.enter_context(patch("api.search_service.search_similar_by_vector", side_effect=candidates))
            async def request():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    return await client.post("/api/search/upload", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
            response = asyncio.run(request())
            self.assertEqual(response.status_code, 200, response.text)
            item = response.json()["detected_items"][0]
            products = item["similar_products"] or item["low_confidence_products"]
            self.assertEqual(products[0]["id"], "fixture")
            self.assertAlmostEqual(products[0]["color_score"], 1, places=6)
            self.assertGreater(products[0]["image_similarity"], .6)
            self.assertEqual(products[0]["color_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
