import unittest
from unittest.mock import patch
import numpy as np
from AI.hsv_descriptor import extract_hsv_descriptor

from api.search_service import (
    apply_match_threshold,
    compute_category_penalty,
    compute_color_penalty,
    get_rerank_weights,
    merge_variant_rankings,
    rerank_products,
)


class SearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.gray = extract_hsv_descriptor(np.full((8, 8, 3), 128, np.uint8))
        self.black = extract_hsv_descriptor(np.zeros((8, 8, 3), np.uint8))
        color_patch = patch("api.search_service.product_color", return_value=self.gray)
        color_patch.start()
        self.addCleanup(color_patch.stop)

    def test_threshold_returns_low_confidence_suggestions(self):
        products = [
            {
                "id": "A1",
                "title": "Weak match",
                "final_score": 0.44,
                "similarity": 0.44,
                "image_similarity": 0.63,
            }
        ]

        result = apply_match_threshold(products, threshold=0.45, min_image_similarity=0.60)

        self.assertFalse(result["is_confident_match"])
        self.assertEqual(result["similar_products"], [])
        self.assertEqual(result["low_confidence_products"][0]["id"], "A1")
        self.assertEqual(result["reject_reason"], "final_score_too_low")

    def test_merge_variant_rankings_keeps_best_product_score(self):
        ranked_variants = [
            (
                "tight",
                [
                    {"id": "P1", "final_score": 0.50, "similarity": 0.50},
                    {"id": "P2", "final_score": 0.65, "similarity": 0.65},
                ],
            ),
            (
                "wide",
                [
                    {"id": "P1", "final_score": 0.75, "similarity": 0.75},
                    {"id": "P3", "final_score": 0.55, "similarity": 0.55},
                ],
            ),
        ]

        merged = merge_variant_rankings(ranked_variants, top_k=3)

        self.assertEqual([product["id"] for product in merged], ["P1", "P2", "P3"])
        self.assertEqual(merged[0]["best_crop_variant"], "wide")
        self.assertEqual(merged[0]["variant_scores"]["tight"], 0.50)
        self.assertEqual(merged[0]["variant_scores"]["wide"], 0.75)

    def test_rerank_weight_defaults_prioritize_measured_real_eval_result(self):
        image_weight, category_weight, color_weight, text_style_weight = get_rerank_weights()

        self.assertAlmostEqual(image_weight, 0.60)
        self.assertAlmostEqual(category_weight, 0.15)
        self.assertAlmostEqual(color_weight, 0.15)
        self.assertAlmostEqual(text_style_weight, 0.10)

    def test_rerank_weight_env_values_are_normalized(self):
        with patch.dict(
            "os.environ",
            {
                "RERANK_IMAGE_WEIGHT": "6",
                "RERANK_CATEGORY_WEIGHT": "2",
                "RERANK_COLOR_WEIGHT": "2",
                "RERANK_TEXT_STYLE_WEIGHT": "2",
            },
        ):
            image_weight, category_weight, color_weight, text_style_weight = get_rerank_weights()

        self.assertAlmostEqual(image_weight, 0.5)
        self.assertAlmostEqual(category_weight, 1 / 6)
        self.assertAlmostEqual(color_weight, 1 / 6)
        self.assertAlmostEqual(text_style_weight, 1 / 6)

    def test_rgb_name_penalty_is_not_transferred_to_hsv(self):
        self.assertEqual(
            compute_color_penalty(0.90, query_color_name="black", product_color_name="gray"),
            0.0,
        )
        self.assertEqual(
            compute_color_penalty(0.10, query_color_name="black", product_color_name="black"),
            0.0,
        )
        self.assertEqual(
            compute_color_penalty(0.10, query_color_name="black", product_color_name="white"),
            0.0,
        )

    def test_category_penalty_applies_to_opposite_product_type(self):
        product = {"category": "Clothing > Men > Pants > Jeans", "title": "Blue jeans"}

        self.assertGreater(compute_category_penalty(product, class_name="short_sleeve_top"), 0.0)
        self.assertEqual(compute_category_penalty(product, class_name="trousers"), 0.0)

    def test_rerank_products_exposes_penalty_breakdown(self):
        products = [
            {
                "id": "P1",
                "similarity": 0.80,
                "category": "Clothing > Men > Pants > Jeans",
                "dominant_color_rgb": [245, 245, 245],
                "dominant_color_name": "white",
            }
        ]

        ranked = rerank_products(
            products,
            category_terms=["shirt", "top"],
            top_k=1,
            query_color=self.black,
            query_color_name="black",
            class_name="short_sleeve_top",
        )

        self.assertGreater(ranked[0]["category_penalty"], 0.0)
        self.assertEqual(ranked[0]["color_penalty"], 0.0)
        self.assertIn("penalty", ranked[0]["match_reason"])

    def test_rerank_products_uses_text_style_signal_without_filtering_candidates(self):
        products = [
            {
                "id": "TEE",
                "title": "Plain Cotton T-Shirt",
                "similarity": 0.70,
                "category": "Clothing > Men > Shirts > T-Shirts",
                "dominant_color_rgb": [120, 120, 120],
                "dominant_color_name": "gray",
            },
            {
                "id": "BUTTON",
                "title": "Short Sleeve Button Down Shirt",
                "similarity": 0.70,
                "category": "Clothing > Men > Shirts > Button-Down Shirts",
                "dominant_color_rgb": [120, 120, 120],
                "dominant_color_name": "gray",
            },
        ]

        with patch(
            "api.search_service.compute_text_style_scores",
            return_value={
                "TEE": {
                    "text_score": 0.10,
                    "style_score": 0.20,
                    "text_style_score": 0.14,
                    "query_style_label": "button_down",
                    "product_styles": ["tshirt"],
                },
                "BUTTON": {
                    "text_score": 0.80,
                    "style_score": 1.00,
                    "text_style_score": 0.88,
                    "query_style_label": "button_down",
                    "product_styles": ["button_down"],
                },
            },
        ):
            ranked = rerank_products(
                products,
                category_terms=["shirt", "top"],
                top_k=2,
                query_color=self.gray,
                query_color_name="gray",
                class_name="short_sleeve_top",
                query_vector=[0.0] * 512,
            )

        self.assertEqual([product["id"] for product in ranked], ["BUTTON", "TEE"])
        self.assertEqual(len(ranked), 2)
        self.assertGreater(ranked[0]["text_style_score"], ranked[1]["text_style_score"])
        self.assertEqual(ranked[0]["query_style_label"], "button_down")
        self.assertIn("text/style match", ranked[0]["match_reason"])

    def test_disabled_text_style_rerank_does_not_consume_score_weight(self):
        product = {
            "id": "P1",
            "title": "Gray T-Shirt",
            "similarity": 0.60,
            "category": "Clothing > Men > Shirts > T-Shirts",
            "dominant_color_rgb": [120, 120, 120],
            "dominant_color_name": "gray",
        }

        with patch.dict("os.environ", {"ENABLE_TEXT_STYLE_RERANK": "false"}):
            ranked = rerank_products(
                [product],
                category_terms=["shirt", "top"],
                top_k=1,
                query_color=self.gray,
                query_color_name="gray",
                class_name="short_sleeve_top",
                query_vector=[0.0] * 512,
        )

        self.assertEqual(ranked[0]["text_style_score"], 0.0)
        self.assertGreater(ranked[0]["final_score"], 0.67)


if __name__ == "__main__":
    unittest.main()
