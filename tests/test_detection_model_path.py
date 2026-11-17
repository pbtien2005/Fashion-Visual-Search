import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from object_detection.config import resolve_model_path
from object_detection.detector import ClothingDetector


class ModelPathTests(unittest.TestCase):
    def test_explicit_path_wins_over_environment(self):
        with patch.dict(os.environ, {"OBJECT_DETECTION_MODEL_PATH": "env.pt"}):
            self.assertEqual(resolve_model_path("chosen.pt"), "chosen.pt")

    def test_configured_missing_path_is_not_silently_replaced(self):
        with patch.dict(os.environ, {"OBJECT_DETECTION_MODEL_PATH": "missing.pt"}):
            detector = ClothingDetector()
            self.assertEqual(detector.model_path, "missing.pt")
            self.assertFalse(detector.model_available)

    def test_default_finds_bundled_weight_from_another_working_directory(self):
        local_weight = Path(__file__).resolve().parents[1] / "object_detection" / "best.pt"
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", lambda path: path == local_weight):
                original_directory = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    try:
                        os.chdir(directory)
                        self.assertEqual(Path(resolve_model_path()), local_weight)
                    finally:
                        os.chdir(original_directory)

    def test_container_mount_takes_precedence_when_present(self):
        mount = Path("/models/object-detection/best.pt")
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", return_value=True):
                self.assertEqual(Path(resolve_model_path()), mount)

    def test_legacy_repository_location_remains_supported(self):
        legacy = Path(__file__).resolve().parents[1] / "models" / "object-detection" / "best.pt"
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", lambda path: path == legacy):
                self.assertEqual(Path(resolve_model_path()), legacy)

    def test_missing_defaults_return_predictable_local_path(self):
        expected = Path(__file__).resolve().parents[1] / "object_detection" / "best.pt"
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", return_value=False):
                self.assertEqual(Path(resolve_model_path()), expected)

    def test_directory_is_not_an_available_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(ClothingDetector(model_path=directory).model_available)

    def test_missing_model_fails_before_importing_yolo(self):
        detector = ClothingDetector(model_path="does-not-exist.pt")
        with self.assertRaisesRegex(FileNotFoundError, "Missing YOLO model"):
            detector._ensure_model()


if __name__ == "__main__":
    unittest.main()
