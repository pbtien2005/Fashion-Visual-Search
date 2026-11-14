import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.search_service import search_match_for_crop


class RuntimeGarmentIdentityTests(unittest.TestCase):
    def test_each_encoder_variant_keeps_the_same_m2_garment_identity(self):
        path = Path(__file__).parent / "fixtures/m2/phase2_response.json"
        crop = json.loads(path.read_text(encoding="utf-8"))["crops"][0]
        observed = []

        def features(garment_id, image, color):
            observed.append(garment_id)
            return SimpleNamespace(embedding=[1.0] + [0.0] * 511, embedding_version=2, color=color)

        with patch.dict(os.environ, {"HSV_CONFIG_PATH": "", "HSV_MISSING_POLICY": "zero"}):
            with patch("api.search_service.extract_runtime_features", side_effect=features):
                with patch("api.search_service.search_similar_by_vector", return_value=[]):
                    search_match_for_crop(crop)
        self.assertEqual(observed, [crop["garment_id"]] * 3)


if __name__ == "__main__":
    unittest.main()
