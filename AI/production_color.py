"""Official HSV147 production contract, cache access and display-only color labels.

No dependency on legacy RGB extraction/similarity. Display colors are bin-center
approximations, never used as retrieval descriptors or ranking distances.
"""
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from AI.color_cache import read_cached_descriptor
from AI.features import color_input_from_detection
from AI.hsv_descriptor import HSVConfig, extract_hsv_descriptor, validate_descriptor, validate_roi

logger = logging.getLogger(__name__)
ROI_POLICY = "source_roi_v1"
DEFAULT_CACHE_PATH = "data/color_cache_hsv147.json"


def decode_uploaded_rgb(image_bytes):
    """Use the same decoder/orientation handling as M2, then explicitly BGR->RGB.

    Bboxes from cv2.imdecode must not be applied to a differently oriented PIL
    image (notably JPEG EXIF rotation). Encoder crops remain untouched.
    """
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def production_config():
    path = os.getenv("HSV_CONFIG_PATH", "").strip()
    config = HSVConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else HSVConfig()
    if config.hue_bins != 18 or config.saturation_bins != 8 or not config.achromatic:
        raise ValueError("Production requires HSV147 (18x8 H-S plus Black/Gray/White)")
    return config


def missing_color_policy():
    policy = os.getenv("HSV_MISSING_POLICY", "zero").strip()
    if policy not in {"zero", "error"}:
        raise ValueError("HSV_MISSING_POLICY must be zero or error")
    return policy


def validate_production_color(color, config=None):
    config = config or production_config()
    values = validate_descriptor(color)
    if (color.config != config.spec() or color.version != config.version or color.roi_policy != ROI_POLICY
            or type(color.valid_pixels) is not int or type(color.total_pixels) is not int
            or not 0 < color.valid_pixels <= color.total_pixels or values.sum() == 0):
        raise ValueError("descriptor_config_policy_or_pixels_mismatch")
    return color


def unavailable(reason, item_id=None):
    logger.warning("HSV unavailable id=%s reason=%s policy=%s", item_id, reason, missing_color_policy())
    if missing_color_policy() == "error":
        raise ValueError(f"HSV unavailable: {reason}")
    return None


def extract_source_color(original_rgb=None, detection=None, item_id=None):
    """Shared query/catalog source priority; absent detection uses original image.

    A provided but invalid ROI/bbox is an error, not an invitation to use RGB.
    Masks continue to follow the existing HSVConfig mask policy.
    """
    config = production_config()  # Configuration errors always fail, even in zero policy.
    try:
        if detection is None:
            roi, mask = original_rgb, None
            metadata = {"roi_source": "original_whole_image", "roi_policy": ROI_POLICY,
                        "roi_fallback_reason": "detection_unavailable"}
        else:
            roi, mask, metadata = color_input_from_detection(detection, original_rgb)
        validate_roi(roi)
        metadata["source_hashes"] = {"roi_pixels": hashlib.sha256(
            str(getattr(roi, "shape", None)).encode() + roi.tobytes()).hexdigest()}
        if isinstance(mask, np.ndarray):
            metadata["source_hashes"]["mask_pixels"] = hashlib.sha256(
                str(mask.shape).encode() + mask.tobytes()).hexdigest()
        color = extract_hsv_descriptor(roi, mask, config, metadata)
        if not color.valid_pixels:
            raise ValueError("empty_descriptor")
        if color.fallback_reason or color.metadata.get("roi_fallback_reason"):
            logger.info("HSV fallback id=%s source=%s mask=%s roi_reason=%s mask_reason=%s",
                        item_id, color.roi_source, color.mask_status,
                        color.metadata.get("roi_fallback_reason"), color.fallback_reason)
        return color
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return unavailable(f"extraction:{type(exc).__name__}:{exc}", item_id)


def descriptor_display(color):
    """Return (RGB swatch, name) from highest-mass histogram branch/bin.

    H-S has no chromatic V; swatches use V=192, so navy/brown/tan cannot be
    reliably recovered. These labels describe bins and do not affect similarity.
    """
    if color is None:
        return None, None
    values = validate_descriptor(color)
    if values.sum() == 0:
        return None, None
    index = int(np.argmax(values))
    hs_bins = color.config["hue_bins"] * color.config["saturation_bins"]
    if index >= hs_bins:
        return ((20, 20, 20), "black") if index == hs_bins else (
            ((128, 128, 128), "gray") if index == hs_bins + 1 else ((240, 240, 240), "white"))
    hue_index, sat_index = divmod(index, color.config["saturation_bins"])
    hue = (hue_index + .5) * 180 / color.config["hue_bins"]
    saturation = (sat_index + .5) * 256 / color.config["saturation_bins"]
    pixel = np.array([[[min(179, int(hue)), min(255, int(saturation)), 192]]], dtype=np.uint8)
    rgb = tuple(int(x) for x in cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0])
    # Hue sectors in OpenCV units; mapping is display-only, not score calibration.
    name = ("red" if hue < 10 or hue >= 170 else "orange" if hue < 25 else
            "yellow" if hue < 35 else "green" if hue < 85 else
            "blue" if hue < 130 else "purple" if hue < 155 else "pink")
    return rgb, name


class ColorStore:
    def __init__(self, cache, config):
        if (cache.get("schema") != "m3-color-cache-v1" or cache.get("config") != config.spec()
                or cache.get("color_version") != config.version):
            raise ValueError("cache_config_or_version_mismatch")
        self.cache, self.config, self.index = cache, config, {}
        for key, entry in cache["entries"].items():
            product_id = str(entry["product_id"])
            if product_id in self.index:
                raise ValueError(f"duplicate_product_id:{product_id}")
            self.index[product_id] = key

    def get(self, product_id, model_name, model_version):
        key = self.index[str(product_id)]
        color = read_cached_descriptor(self.cache, key, model_name, model_version,
                                       ROI_POLICY, self.config.version)
        return validate_production_color(color, self.config)


@lru_cache(maxsize=2)
def _load_store(path, modified, size, config):
    return ColorStore(json.loads(Path(path).read_text(encoding="utf-8")), config)


def configured_color_store():
    config = production_config()
    path = Path(os.getenv("HSV_COLOR_CACHE_PATH", DEFAULT_CACHE_PATH)).resolve()
    stat = path.stat()
    return _load_store(str(path), stat.st_mtime_ns, stat.st_size, config)


def product_color(product_id, model_name, model_version, store=None):
    try:
        store = store if store is not None else configured_color_store()
        return store.get(product_id, model_name, model_version)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return unavailable(f"catalog:{type(exc).__name__}:{exc}", product_id)
