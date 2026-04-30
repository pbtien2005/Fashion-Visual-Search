import unittest

from PIL import Image, ImageDraw

from AI.color_utils import color_similarity, extract_dominant_color, name_color


class ColorUtilsTests(unittest.TestCase):
    def test_names_common_fashion_neutrals(self):
        self.assertEqual(name_color((134, 126, 92)), "olive")
        self.assertEqual(name_color((190, 169, 117)), "khaki")
        self.assertEqual(name_color((210, 180, 140)), "tan")
        self.assertEqual(name_color((235, 228, 205)), "cream")
        self.assertEqual(name_color((24, 38, 74)), "navy")

    def test_khaki_is_not_similar_to_black(self):
        score = color_similarity(
            (190, 169, 117),
            (20, 20, 20),
            query_name="khaki",
            product_name="black",
        )

        self.assertLessEqual(score, 0.25)

    def test_olive_is_only_partly_similar_to_green(self):
        score = color_similarity(
            (91, 98, 57),
            (60, 140, 80),
            query_name="olive",
            product_name="green",
        )

        self.assertLessEqual(score, 0.70)

    def test_dominant_color_prioritizes_center_over_border_noise(self):
        image = Image.new("RGB", (100, 100), (30, 90, 170))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 69, 69), fill=(190, 45, 45))

        dominant = extract_dominant_color(image)

        self.assertEqual(name_color(dominant), "red")


if __name__ == "__main__":
    unittest.main()
