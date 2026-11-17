import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from AI.hsv_descriptor import extract_hsv_descriptor
from api.search_service import (
    RETRIEVAL_ACCEPT_THRESHOLD,
    _search_single_crop_variant,
    apply_match_threshold,
    search_by_histogram,
    search_match_for_crop,
)


class RetrievalModeTests(unittest.TestCase):
    def setUp(self):
        self.red = extract_hsv_descriptor(np.full((8, 8, 3), (180, 30, 30), np.uint8))
        self.blue = extract_hsv_descriptor(np.full((8, 8, 3), (30, 30, 180), np.uint8))
        self.config = SimpleNamespace(model_name="patrickjohncyh/fashion-clip", version=2)

    def test_histogram_scans_full_catalog_without_vector_retrieval(self):
        catalog = [{"id": "blue"}, {"id": "red"}]
        with patch("api.search_service.validate_production_color"), patch(
            "api.search_service.list_color_search_products", return_value=catalog
        ) as list_all, patch("api.search_service.product_color", side_effect=[self.blue, self.red]), patch(
            "api.search_service.search_similar_by_vector"
        ) as vector_search:
            ranked = search_by_histogram(self.red, self.config, top_k=2)
        self.assertEqual([row["id"] for row in ranked], ["red", "blue"])
        list_all.assert_called_once_with(self.config.model_name, self.config.version)
        vector_search.assert_not_called()

    def test_embedding_only_uses_cosine_and_skips_hsv_rerank(self):
        with patch("api.search_service.image_from_base64", return_value=object()), patch(
            "api.search_service.extract_runtime_features",
            return_value=SimpleNamespace(embedding=[0.0] * 512, color=None, embedding_version=2),
        ), patch("api.search_service.search_similar_by_vector", return_value=[
            {"id": "A", "similarity": 0.72}, {"id": "B", "similarity": 0.61}
        ]), patch("api.search_service.rerank_products") as rerank, patch(
            "api.search_service.product_color"
        ) as color:
            ranked, _, _ = _search_single_crop_variant(
                {"class_name": "shirt"}, {"image_base64": "ignored"}, 2, [], self.config,
                mode="embedding_only",
            )
        self.assertEqual([row["final_score"] for row in ranked], [0.72, 0.61])
        rerank.assert_not_called()
        color.assert_not_called()

    def test_strict_point_six_boundary(self):
        for score, accepted in [(0.5999, False), (RETRIEVAL_ACCEPT_THRESHOLD, False),
                                (0.6001, True)]:
            with self.subTest(score=score):
                result = apply_match_threshold(
                    [{"id": "A", "final_score": score, "similarity": score,
                      "image_similarity": 0.2, "_hsv_display": True}],
                    threshold=RETRIEVAL_ACCEPT_THRESHOLD, strict=True,
                    require_image_similarity=False,
                )
                self.assertEqual(result["is_confident_match"], accepted)
                self.assertEqual(len(result["similar_products"]), int(accepted))

    def test_embedding_response_removes_each_product_at_or_below_threshold(self):
        crop = {"class_name": "short_sleeve_top", "image_base64": "unused"}
        products = [
            {"id": "A", "final_score": 0.71, "similarity": 0.71,
             "image_similarity": 0.71, "_hsv_display": True},
            {"id": "B", "final_score": 0.60, "similarity": 0.60,
             "image_similarity": 0.60, "_hsv_display": True},
        ]
        with patch("api.search_service._search_single_crop_variant",
                   return_value=(products, None, None)):
            result = search_match_for_crop(crop, top_k=2, mode="embedding_only")
        self.assertEqual([item["id"] for item in result["similar_products"]], ["A"])
        self.assertEqual(result["low_confidence_products"], [])


if __name__ == "__main__":
    unittest.main()
