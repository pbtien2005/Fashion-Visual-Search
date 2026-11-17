"""Conditional Otsu masks for color pixels, with explicit geometric heuristics.

Centered foreground against a contrasting ROI border is an assumption, not a
semantic clothing classifier. Parameters are provisional until validation.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

try:
    from .roi import validate_rgb_image
except ImportError:
    from roi import validate_rgb_image


@dataclass(frozen=True)
class MaskConfig:
    mode: str = "off"
    min_side: int = 8
    min_contrast: float = 15.0
    min_area_fraction: float = 0.03
    max_area_fraction: float = 0.95
    min_center_fraction: float = 0.55
    min_center_border_gap: float = 0.10
    border_fraction: float = 0.05
    max_change_fraction: float = 0.15
    kernel_size: int = 3

    def __post_init__(self):
        if self.mode not in {"off", "raw", "opening", "closing", "open_close"}:
            raise ValueError("Unknown mask mode")
        if type(self.min_side) is not int or self.min_side < 2 or self.kernel_size != 3:
            raise ValueError("min_side must be >=2; this experiment fixes kernel_size=3")
        fractions = (self.min_area_fraction, self.max_area_fraction, self.min_center_fraction,
                     self.min_center_border_gap, self.border_fraction, self.max_change_fraction)
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in fractions):
            raise ValueError("Mask fractions must be finite and between 0 and 1")
        if not 0 < self.min_area_fraction < self.max_area_fraction < 1 or not 0 < self.border_fraction < .25:
            raise ValueError("Invalid area or border bounds")
        if not math.isfinite(self.min_contrast) or not 0 < self.min_contrast <= 255:
            raise ValueError("min_contrast must be in (0,255]")

    @property
    def config_id(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"m2-otsu-v1-{digest}"


def load_mask_config():
    path = os.getenv("M2_MASK_CONFIG_PATH", "").strip()
    config = MaskConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else MaskConfig()
    mode = os.getenv("M2_MASK_MODE")
    return replace(config, mode=mode) if mode is not None else config


@dataclass
class SegmentationResult:
    mask: np.ndarray | None
    raw_mask: np.ndarray | None
    processed_mask: np.ndarray | None
    metadata: dict


def _candidate(gray, config, details):
    threshold, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = gray.shape
    border_size = max(1, int(min(height, width) * config.border_fraction))
    border = np.ones(gray.shape, dtype=bool)
    border[border_size:-border_size, border_size:-border_size] = False
    center = np.zeros(gray.shape, dtype=bool)
    center[height // 4:height - height // 4, width // 4:width - width // 4] = True
    candidates = []
    for name, selected in (("bright", bright != 0), ("dark", bright == 0)):
        center_fraction = float(selected[center].mean())
        border_fraction = float(selected[border].mean())
        candidates.append((center_fraction - border_fraction, center_fraction, border_fraction, name, selected))
    gap, central, boundary, polarity, selected = max(candidates, key=lambda candidate: candidate[0])
    details.update(otsu_threshold=float(threshold), polarity=polarity, center_fraction=central,
                   border_foreground_fraction=boundary, center_border_gap=gap)
    raw = selected.astype(np.uint8) * 255
    ambiguous = central < config.min_center_fraction or gap < config.min_center_border_gap
    return raw, ambiguous


def _morphology(raw, config):
    result = raw.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if config.mode in {"opening", "open_close"}:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    if config.mode in {"closing", "open_close"}:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    return result


def segment_roi(source_rgb, config=None):
    validate_rgb_image(source_rgb, "segmentation source")
    config = config or MaskConfig()
    started = time.perf_counter()
    details = {"config": asdict(config), "provisional_not_validation_tuned": True}
    raw, processed = None, None

    def finish(reason=None):
        details["latency_ms"] = (time.perf_counter() - started) * 1000
        enabled = config.mode != "off"
        metadata = {
            "roi_status": "unmasked" if not enabled else "fallback" if reason else "masked",
            "fallback_reason": reason, "mask_method": "otsu" if enabled else "off",
            "mask_config_id": config.config_id, "segmentation": details,
        }
        return SegmentationResult(processed if reason is None else None, raw, processed, metadata)

    if config.mode == "off":
        return finish("mask_disabled")
    if min(source_rgb.shape[:2]) < config.min_side:
        return finish("roi_too_small")
    try:
        gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        details["contrast_p95_p05"] = contrast
        if contrast < config.min_contrast:
            return finish("low_contrast")
        raw, ambiguous = _candidate(gray, config, details)
        if ambiguous:
            return finish("ambiguous_foreground")
        raw_fraction = float(np.count_nonzero(raw) / raw.size)
        details["raw_area_fraction"] = raw_fraction
        if not config.min_area_fraction <= raw_fraction <= config.max_area_fraction:
            return finish("empty_mask" if not raw_fraction else "area_out_of_range")
        processed = _morphology(raw, config)
        fraction = float(np.count_nonzero(processed) / processed.size)
        change = float(np.count_nonzero(processed != raw) / raw.size)
        details.update(area_fraction=fraction, change_fraction=change)
        if not config.min_area_fraction <= fraction <= config.max_area_fraction:
            return finish("empty_mask" if not fraction else "area_out_of_range")
        if change > config.max_change_fraction:
            return finish("morphology_unstable")
        return finish()
    except cv2.error as exc:
        details["error_type"] = type(exc).__name__
        return finish("mask_processing_error")
