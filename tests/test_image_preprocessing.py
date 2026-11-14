import base64
import io
import unittest
from dataclasses import replace
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from object_detection.image_preprocessing import ImageDecodeError, PreprocessConfig, prepare_image


def encode(image, format="PNG", **options):
    buffer = io.BytesIO()
    image.save(buffer, format=format, **options)
    return buffer.getvalue()


class ImagePreprocessingTests(unittest.TestCase):
    def test_exif_rotation_moves_corner_marker_once(self):
        image = Image.new("RGB", (80, 48))
        ImageDraw.Draw(image).rectangle((0, 0, 19, 19), fill="red")
        for orientation, size, marker in (
            (1, (80, 48), (5, 5)),
            (3, (80, 48), (74, 42)),
            (6, (48, 80), (42, 5)),
            (8, (48, 80), (5, 74)),
        ):
            with self.subTest(orientation=orientation):
                exif = image.getexif()
                exif[274] = orientation
                prepared = prepare_image(encode(image, "JPEG", exif=exif))
                meta = prepared.geometry_meta
                self.assertEqual((meta["raw_width"], meta["raw_height"]), (80, 48))
                self.assertEqual((meta["width"], meta["height"]), size)
                self.assertEqual(meta["exif_orientation"], orientation)
                self.assertEqual(meta["exif_transposed"], orientation != 1)
                x, y = marker
                self.assertGreater(prepared.source_rgb[y, x, 0], 230)
                self.assertLess(prepared.source_rgb[y, x, 1], 15)

    def test_modes_and_white_transparency_background(self):
        palette = Image.new("P", (2, 2))
        palette.putpalette([255, 0, 0] + [0] * 765)
        for image, fmt, options, expected, composited in (
            (Image.new("RGB", (2, 2), (255, 0, 0)), "PNG", {}, (255, 0, 0), False),
            (Image.new("RGBA", (2, 2), (255, 0, 0, 0)), "PNG", {}, (255, 255, 255), True),
            (Image.new("LA", (2, 2), (0, 0)), "PNG", {}, (255, 255, 255), True),
            (Image.new("L", (2, 2), 64), "PNG", {}, (64, 64, 64), False),
            (Image.new("1", (2, 2), 1), "PNG", {}, (255, 255, 255), False),
            (palette, "PNG", {}, (255, 0, 0), False),
            (palette, "PNG", {"transparency": 0}, (255, 255, 255), True),
            (Image.new("CMYK", (2, 2), (0, 255, 255, 0)), "TIFF", {}, (255, 0, 0), False),
        ):
            with self.subTest(mode=image.mode, options=options):
                prepared = prepare_image(encode(image, fmt, **options))
                self.assertEqual(prepared.source_rgb.shape, (2, 2, 3))
                self.assertEqual(prepared.source_rgb.dtype, np.uint8)
                self.assertEqual(tuple(prepared.source_rgb[0, 0]), expected)
                self.assertEqual(prepared.quality_meta["alpha_composited"], composited)
                self.assertEqual(prepared.quality_meta["input_mode"], image.mode)

    def test_rotated_tiff_retains_raw_dimensions(self):
        image = Image.new("RGB", (80, 48))
        ImageDraw.Draw(image).rectangle((0, 0, 19, 19), fill="red")
        exif = image.getexif()
        exif[274] = 6
        prepared = prepare_image(encode(image, "TIFF", exif=exif))
        meta = prepared.geometry_meta
        self.assertEqual((meta["raw_width"], meta["raw_height"]), (80, 48))
        self.assertEqual((meta["oriented_width"], meta["oriented_height"]), (48, 80))
        self.assertEqual(tuple(prepared.source_rgb[5, 42]), (255, 0, 0))

    def test_animated_input_uses_first_frame(self):
        red = Image.new("RGB", (8, 8), "red")
        blue = Image.new("RGB", (8, 8), "blue")
        prepared = prepare_image(encode(red, "GIF", save_all=True, append_images=[blue]))
        self.assertEqual(tuple(prepared.source_rgb[0, 0]), (255, 0, 0))

    def test_bad_inputs_raise_decode_error(self):
        jpeg = encode(Image.new("RGB", (64, 64)), "JPEG")
        oversized = encode(Image.new("RGB", (11, 10)))
        unsupported = encode(Image.new("I;16", (2, 2)))
        config = PreprocessConfig(max_pixels=5000)
        for payload in (b"", b"garbage", jpeg[:-32], unsupported):
            with self.subTest(size=len(payload)), self.assertRaises(ImageDecodeError):
                prepare_image(payload, config)
        with self.assertRaisesRegex(ImageDecodeError, "pixel"):
            prepare_image(oversized, replace(config, max_pixels=100))
        with patch.object(Image, "MAX_IMAGE_PIXELS", 50), self.assertRaises(ImageDecodeError):
            prepare_image(oversized)

    def test_decoder_ignores_open_patch_installed_after_model_load(self):
        encoded = encode(Image.new("RGB", (2, 2), "red"))
        with patch.object(Image, "open", side_effect=ModuleNotFoundError(name="pi_heif")) as patched:
            with self.assertRaises(ImageDecodeError):
                prepare_image(b"garbage")
            self.assertEqual(tuple(prepare_image(encoded).source_rgb[0, 0]), (255, 0, 0))
        patched.assert_not_called()

    def test_downscale_preserves_ratio_without_upscaling(self):
        config = PreprocessConfig(max_side=80)
        for size, expected in (((160, 100), (80, 50)), ((1, 200), (1, 80)), ((30, 15), (30, 15))):
            with self.subTest(size=size):
                prepared = prepare_image(encode(Image.new("RGB", size)), config)
                meta = prepared.geometry_meta
                self.assertEqual(prepared.source_rgb.shape, (*reversed(expected), 3))
                self.assertEqual(meta["scale"], expected[0] / size[0])
                self.assertEqual(meta["resized"], size != expected)

    def test_gamma_is_conditional_and_preserves_independent_source(self):
        gamma = PreprocessConfig(mode="gamma", gamma=0.5)
        for config, value, output, applied in (
            (gamma, 64, 128, True),
            (gamma, 200, 200, False),
            (PreprocessConfig(), 64, 64, False),
        ):
            with self.subTest(mode=config.mode, value=value):
                prepared = prepare_image(encode(Image.new("RGB", (8, 8), (value,) * 3)), config)
                source, detection = prepared.source_rgb, prepared.detection_rgb
                self.assertEqual(prepared.schema_version, 1)
                self.assertEqual(source.shape, detection.shape)
                self.assertEqual(detection.dtype, np.uint8)
                self.assertFalse(np.shares_memory(source, detection))
                self.assertEqual(prepared.quality_meta["gamma_applied"], applied)
                self.assertAlmostEqual(prepared.quality_meta["mean_luma"], value / 255)
                np.testing.assert_array_equal(detection, np.full(source.shape, output, dtype=np.uint8))
                detection[:] = 1
                np.testing.assert_array_equal(source, np.full(source.shape, value, dtype=np.uint8))

    def test_gamma_lut_is_monotone_and_preserves_endpoints(self):
        pixels = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        config = PreprocessConfig(mode="gamma", brightness_threshold=1, gamma=0.5)
        corrected = prepare_image(encode(Image.fromarray(pixels)), config).detection_rgb[0, :, 0]
        self.assertEqual((corrected[0], corrected[-1]), (0, 255))
        self.assertTrue(np.all(np.diff(corrected.astype(int)) >= 0))

    def test_rgb_luma_and_alternative_statistic(self):
        pixels = np.zeros((1, 10, 3), dtype=np.uint8)
        pixels[0, :4] = (255, 255, 255)
        encoded = encode(Image.fromarray(pixels))
        config = PreprocessConfig(mode="gamma")
        self.assertFalse(prepare_image(encoded, config).quality_meta["gamma_applied"])
        self.assertTrue(
            prepare_image(encoded, replace(config, brightness_stat="median_luma")).quality_meta["gamma_applied"]
        )
        red = prepare_image(encode(Image.new("RGB", (2, 2), "red")))
        for statistic in ("mean_luma", "median_luma", "p5_luma", "p95_luma"):
            self.assertAlmostEqual(red.quality_meta[statistic], 0.299)
        white = prepare_image(encode(Image.new("RGB", (2, 2), "white")), replace(config, brightness_threshold=1))
        self.assertEqual(white.quality_meta["mean_luma"], 1)
        self.assertFalse(white.quality_meta["gamma_applied"])


@unittest.skipUnless(find_spec("cv2"), "OpenCV is required for detector integration")
class DetectorIntegrationTests(unittest.TestCase):
    def test_identity_matches_legacy_and_gamma_does_not_change_crop_pixels(self):
        import cv2

        from object_detection.detector import ClothingDetector
        from object_detection.utils import ndarray_to_base64_jpeg

        image = Image.new("RGB", (100, 90), "red")
        encoded = encode(image)
        old_image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[10, 10, 80, 80]])
        detector = ClothingDetector(preprocess_config=PreprocessConfig())
        model = SimpleNamespace()

        def predict(pixels, **options):
            model.input = pixels
            return [SimpleNamespace(boxes=[box])]

        model.predict = predict
        detector._model = model
        legacy = detector._process(old_image)[0]
        identity = detector.detect_from_bytes(encoded)[0]
        np.testing.assert_array_equal(model.input, old_image)
        self.assertEqual(tuple(model.input[0, 0]), (0, 0, 255))
        self.assertEqual(legacy["bbox"], identity["bbox"])
        for before, after in zip(legacy["variants"], identity["variants"], strict=True):
            self.assertEqual(before["bbox"], after["bbox"])
            np.testing.assert_array_equal(before["image"], after["image"])
        with Image.open(io.BytesIO(base64.b64decode(ndarray_to_base64_jpeg(identity["image"])))) as crop:
            self.assertGreater(crop.getpixel((100, 100))[0], 240)
            self.assertLess(crop.getpixel((100, 100))[2], 10)

        detector.preprocess_config = PreprocessConfig(mode="gamma", gamma=0.5)
        dark = encode(Image.new("RGB", image.size, (64, 0, 0)))
        gamma_crop = detector.detect_from_bytes(dark)[0]
        self.assertEqual(gamma_crop["bbox"], [10, 10, 80, 80])
        self.assertEqual(tuple(model.input[0, 0]), (0, 0, 128))
        self.assertEqual(tuple(gamma_crop["image"][100, 100]), (0, 0, 64))


@unittest.skipUnless(
    all(find_spec(name) for name in ("fastapi", "httpx", "multipart")),
    "FastAPI, python-multipart and httpx are required for the HTTP test",
)
class UploadTests(unittest.TestCase):
    def test_bad_upload_returns_400_and_missing_model_returns_503(self):
        from fastapi.testclient import TestClient

        from object_detection.api import app, detector

        with TestClient(app) as client:
            with patch.object(type(detector), "model_available", True):
                response = client.post("/detect/upload", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
                self.assertEqual(response.status_code, 400)
                self.assertIn("decode", response.json()["detail"])
                with patch.object(detector, "detect_batch_from_bytes", side_effect=RuntimeError("model error")):
                    response = client.post("/detect/upload", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
                    self.assertEqual(response.status_code, 500)
            with patch.object(type(detector), "model_available", False):
                response = client.post("/detect/upload", files={"file": ("test.jpg", b"garbage", "image/jpeg")})
                self.assertEqual(response.status_code, 503)
