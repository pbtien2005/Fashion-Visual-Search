import os
import unittest
from unittest import mock

from api.category_mapping import UPPER_LAYER_CLASSES, get_category_terms
from api.search_service import compute_category_score

BLAZER = {"title": "Women's Casual Blazer Open Front Jacket", "category": "Clothing > Women > Suits & Sport Coats > Blazers"}
SHIRT = {"title": "Men's Long Sleeve Button Down Shirt", "category": "Clothing > Shirts"}
TROUSERS = {"title": "Men's Slim Fit Jeans", "category": "Clothing > Jeans"}


class CategoryGroupingTests(unittest.TestCase):
    def test_default_is_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CATEGORY_GROUP_UPPER", None)
            self.assertEqual(get_category_terms("vest"), ["vest"])
            self.assertEqual(get_category_terms("long_sleeve_outwear"), ["jacket", "coat", "outwear"])

    def test_flag_shares_terms_across_the_three_classes_only(self):
        with mock.patch.dict(os.environ, {"CATEGORY_GROUP_UPPER": "true"}):
            shared = {tuple(get_category_terms(name)) for name in UPPER_LAYER_CLASSES}
            self.assertEqual(len(shared), 1)
            self.assertIn("blazer", get_category_terms("vest"))
            self.assertEqual(get_category_terms("trousers"), ["pants", "trousers", "jeans", "chino"])
            self.assertEqual(get_category_terms("short_sleeve_top"), ["top", "shirt", "t-shirt", "tee"])

    def test_mislabeled_class_no_longer_zeroes_the_right_product(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CATEGORY_GROUP_UPPER", None)
            off = compute_category_score(BLAZER, get_category_terms("vest"), "vest")
        with mock.patch.dict(os.environ, {"CATEGORY_GROUP_UPPER": "1"}):
            on = compute_category_score(BLAZER, get_category_terms("vest"), "vest")
        self.assertGreater(on, off)

    def test_opposite_categories_still_score_zero_when_grouped(self):
        with mock.patch.dict(os.environ, {"CATEGORY_GROUP_UPPER": "true"}):
            self.assertEqual(compute_category_score(TROUSERS, get_category_terms("long_sleeve_outwear"),
                                                    "long_sleeve_outwear"), 0.0)
            self.assertGreater(compute_category_score(SHIRT, get_category_terms("long_sleeve_outwear"),
                                                      "long_sleeve_outwear"), 0.0)


if __name__ == "__main__":
    unittest.main()
