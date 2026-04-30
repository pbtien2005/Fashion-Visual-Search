import unittest

from AI.vectorizer import choose_best_detection_crop


class VectorizerCropTests(unittest.TestCase):
    def test_choose_best_detection_crop_prefers_confident_large_crop(self):
        crops = [
            {"confidence": 0.95, "bbox": [0, 0, 10, 10], "image_base64": "small"},
            {"confidence": 0.60, "bbox": [0, 0, 40, 40], "image_base64": "large"},
        ]

        selected = choose_best_detection_crop(crops)

        self.assertEqual(selected["image_base64"], "large")

    def test_choose_best_detection_crop_handles_empty_input(self):
        self.assertIsNone(choose_best_detection_crop([]))


if __name__ == "__main__":
    unittest.main()
