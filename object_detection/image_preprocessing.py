"""Normalize query images and prepare independent RGB arrays for YOLO and crops."""

import io
import os
from dataclasses import dataclass
from struct import error as StructError
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps, TiffImagePlugin
from PIL.Image import open as open_pillow_image  # Keep Pillow's opener before YOLO loads.


class ImageDecodeError(ValueError):
    """The uploaded image cannot be prepared safely."""


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str = "identity"
    max_side: int = 1280
    max_pixels: int = 50_000_000
    brightness_stat: str = "mean_luma"
    brightness_threshold: float = 0.35
    gamma: float = 0.6

    def __post_init__(self):
        if self.mode not in {"identity", "gamma"}:
            raise ValueError("Preprocess mode must be identity or gamma")
        if self.max_side < 1 or self.max_pixels < 1:
            raise ValueError("Preprocess size limits must be positive")
        if self.brightness_stat not in {"mean_luma", "median_luma", "p5_luma", "p95_luma"}:
            raise ValueError("Unsupported brightness statistic")
        if not 0 <= self.brightness_threshold <= 1:
            raise ValueError("Brightness threshold must be in [0, 1]")
        if not 0 < self.gamma < 1:
            raise ValueError("Gamma must be in (0, 1) to brighten dark images")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            mode=os.getenv("PREPROCESS_MODE", defaults.mode),
            max_side=int(os.getenv("PREPROCESS_MAX_SIDE", defaults.max_side)),
            max_pixels=int(os.getenv("PREPROCESS_MAX_PIXELS", defaults.max_pixels)),
            brightness_stat=os.getenv("PREPROCESS_BRIGHTNESS_STAT", defaults.brightness_stat),
            brightness_threshold=float(os.getenv("PREPROCESS_DARK_THRESHOLD", defaults.brightness_threshold)),
            gamma=float(os.getenv("PREPROCESS_GAMMA", defaults.gamma)),
        )


@dataclass
class PreparedImage:
    """Both arrays use normalized coordinates (after EXIF and resize).

    Detection bboxes index either array directly. To map to the oriented
    original, use the oriented-to-final dimension ratio separately on each
    axis. Use the EXIF orientation to map further back to raw coordinates.
    """

    source_rgb: np.ndarray
    detection_rgb: np.ndarray
    geometry_meta: dict
    quality_meta: dict
    schema_version: int = 1


def prepare_image(image_bytes: bytes, config: PreprocessConfig | None = None) -> PreparedImage:
    """Use frame zero, apply EXIF orientation, and downscale to the configured limit.

    Keep RGB, composite RGBA/LA and transparent palette images on white, and
    convert L, 1, opaque P, and CMYK to RGB. Other modes raise ImageDecodeError.
    Gamma uses s = r**gamma for normalized channel values, so gamma < 1
    brightens only the detection array when the configured statistic is low.
    """
    started = perf_counter()
    config = config if config is not None else PreprocessConfig()
    if not image_bytes:
        raise ImageDecodeError("Uploaded image is empty")

    try:
        with open_pillow_image(io.BytesIO(image_bytes)) as decoded:
            raw_width, raw_height = decoded.size
            if isinstance(decoded, TiffImagePlugin.TiffImageFile):
                # Pillow may already expose the oriented size in the TIFF header.
                raw_width = decoded.tag_v2[TiffImagePlugin.IMAGEWIDTH]
                raw_height = decoded.tag_v2[TiffImagePlugin.IMAGELENGTH]
            if raw_width * raw_height > config.max_pixels:
                raise ImageDecodeError(f"Uploaded image exceeds the {config.max_pixels:,} pixel limit")
            orientation = decoded.getexif().get(274, 1)
            orientation = int(orientation) if orientation in (2, 3, 4, 5, 6, 7, 8) else 1
            input_mode = decoded.mode
            image = ImageOps.exif_transpose(decoded)
    except ImageDecodeError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageDecodeError("Uploaded image exceeds Pillow's safety pixel limit") from exc
    except (OSError, SyntaxError, ValueError, OverflowError, StructError, TypeError) as exc:
        raise ImageDecodeError("Could not decode uploaded image") from exc

    oriented_width, oriented_height = image.size
    alpha_composited = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and image.has_transparency_data
    )
    if alpha_composited:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB")
    elif image.mode in {"L", "1", "P", "CMYK"}:
        image = image.convert("RGB")
    elif image.mode != "RGB":
        raise ImageDecodeError(f"Unsupported uploaded image mode: {input_mode}")

    resized = max(image.size) > config.max_side
    if resized:
        ratio = config.max_side / max(image.size)
        size = (max(1, round(oriented_width * ratio)), max(1, round(oriented_height * ratio)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    width, height = image.size
    source_rgb = np.array(image, dtype=np.uint8)

    luma = (source_rgb @ np.array([0.299, 0.587, 0.114])) / 255
    p5, median, p95 = np.percentile(luma, [5, 50, 95])
    quality_meta = {
        "mode": config.mode,
        "mean_luma": float(luma.mean()),
        "median_luma": float(median),
        "p5_luma": float(p5),
        "p95_luma": float(p95),
        "brightness_threshold": config.brightness_threshold,
        "brightness_stat": config.brightness_stat,
        "input_mode": input_mode,
        "alpha_composited": alpha_composited,
    }
    gamma_applied = (
        config.mode == "gamma" and quality_meta[config.brightness_stat] < config.brightness_threshold
    )
    if gamma_applied:
        lut = np.rint(255 * (np.arange(256) / 255) ** config.gamma).astype(np.uint8)
        detection_rgb = lut[source_rgb]
    else:
        detection_rgb = source_rgb.copy()
    quality_meta.update(
        gamma_applied=gamma_applied,
        gamma=config.gamma if gamma_applied else None,
        preprocess_ms=(perf_counter() - started) * 1000,
    )
    geometry_meta = {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "exif_orientation": orientation,
        "exif_transposed": orientation != 1,
        "oriented_width": oriented_width,
        "oriented_height": oriented_height,
        "width": width,
        "height": height,
        "scale": width / oriented_width,
        "resized": resized,
        "resize_interpolation": "lanczos",
    }
    return PreparedImage(source_rgb, detection_rgb, geometry_meta, quality_meta)
