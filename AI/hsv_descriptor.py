"""Versioned, RGB-only color descriptors. No encoder or detector dependency."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVConfig:
    hue_bins: int = 18
    saturation_bins: int = 8
    achromatic: bool = True
    # OpenCV uint8 HSV units; starting values, not validated optima.
    achromatic_s_max: int = 32
    black_v_max: int = 40
    white_v_min: int = 220
    mask_policy: str = "roi"  # roi fallback, empty descriptor, or error
    min_mask_fraction: float = 0.001
    max_mask_fraction: float = 1.0
    min_pixels: int = 1

    def __post_init__(self):
        for value in (self.hue_bins, self.saturation_bins, self.min_pixels):
            if type(value) is not int or value <= 0:
                raise ValueError("Bins and min_pixels must be positive integers")
        if type(self.achromatic) is not bool:
            raise ValueError("achromatic must be bool")
        for value in (self.achromatic_s_max, self.black_v_max, self.white_v_min):
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError("HSV thresholds must be integers in [0,255]")
        if self.black_v_max >= self.white_v_min:
            raise ValueError("black_v_max must be less than white_v_min")
        if self.mask_policy not in {"roi", "empty", "error"}:
            raise ValueError("mask_policy must be roi, empty or error")
        if not 0 <= self.min_mask_fraction <= self.max_mask_fraction <= 1:
            raise ValueError("Invalid mask fraction interval")

    @property
    def dimension(self):
        return self.hue_bins * self.saturation_bins + (3 if self.achromatic else 0)

    def spec(self):
        return {**asdict(self), "h_range": [0, 180], "s_range": [0, 256],
                "normalization": "L1", "conversion": "OpenCV_RGB2HSV_uint8",
                "groups": ["black", "gray", "white"] if self.achromatic else []}

    @property
    def version(self):
        digest = hashlib.sha256(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()[:16]
        return f"hsv-hs-v1-{self.dimension}-{digest}"


@dataclass
class ColorDescriptor:
    values: list[float]
    version: str
    config: dict
    roi_source: str
    roi_policy: str
    mask_status: str
    fallback_reason: str | None
    valid_pixels: int
    total_pixels: int
    mask_fraction: float | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def validate_roi(roi_rgb):
    if (not isinstance(roi_rgb, np.ndarray) or roi_rgb.dtype != np.uint8
            or roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3 or 0 in roi_rgb.shape):
        raise ValueError("roi_rgb must be nonempty RGB uint8 HxWx3; convert BGR explicitly")


def select_pixels(roi_rgb, mask, config):
    """Return selection and audit info. Never infer foreground from pixel colors."""
    validate_roi(roi_rgb)
    size = roi_rgb.shape[:2]
    status, fraction = "missing", None
    selected = None
    if mask is not None:
        if (not isinstance(mask, np.ndarray) or mask.shape != size
                or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8))):
            status = "invalid"
        elif not np.isin(mask, [0, 1, 255]).all():
            status = "invalid"  # Reject logits/probabilities and ambiguous label maps.
        else:
            selected = mask != 0
            count = int(selected.sum())
            fraction = count / selected.size
            if count == 0:
                status = "empty"
            elif (count < config.min_pixels or not
                  config.min_mask_fraction <= fraction <= config.max_mask_fraction):
                status = "insufficient"
            else:
                status = "valid"
    if status == "valid":
        return selected, status, None, fraction
    reason = f"mask_{status}"
    if config.mask_policy == "error":
        raise ValueError(reason)
    selected = np.full(size, config.mask_policy == "roi", dtype=bool)
    return selected, status, reason, fraction


def extract_hsv_descriptor(roi_rgb, mask=None, config=None, metadata=None):
    config = config or HSVConfig()
    metadata = dict(metadata or {})
    selected, status, fallback, fraction = select_pixels(roi_rgb, mask, config)
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[selected]
    count = len(pixels)
    histogram = np.zeros(config.dimension, dtype=np.float64)
    if count:
        h, s, v = pixels.T
        chromatic = np.ones(count, dtype=bool)
        if config.achromatic:
            # Low S is required even for black: dark saturated fabrics retain hue.
            neutral = s <= config.achromatic_s_max
            black = neutral & (v <= config.black_v_max)
            white = neutral & (v >= config.white_v_min)
            gray = neutral & ~black & ~white
            chromatic = ~neutral
            histogram[-3:] = [black.sum(), gray.sum(), white.sum()]
        h_index = h[chromatic].astype(np.int64) * config.hue_bins // 180
        s_index = s[chromatic].astype(np.int64) * config.saturation_bins // 256
        histogram[:config.hue_bins * config.saturation_bins] = np.bincount(
            h_index * config.saturation_bins + s_index,
            minlength=config.hue_bins * config.saturation_bins)
        histogram /= count
    return ColorDescriptor(histogram.tolist(), config.version, config.spec(),
                           metadata.get("roi_source", "source_roi_rgb"),
                           metadata.get("roi_policy", "source_roi_v1"), status, fallback,
                           count, selected.size, fraction, metadata)


def validate_descriptor(descriptor):
    config_fields = HSVConfig.__dataclass_fields__
    config = HSVConfig(**{k: v for k, v in descriptor.config.items() if k in config_fields})
    if descriptor.version != config.version or descriptor.config != config.spec():
        raise ValueError("Descriptor version/config mismatch")
    values = np.asarray(descriptor.values, dtype=np.float64)
    if (values.shape != (config.dimension,) or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError("Descriptor must have expected dimension and finite nonnegative values")
    total = float(values.sum())
    if total != 0 and not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("Descriptor must be L1 normalized or all-zero")
    return values


def histogram_similarity(left, right, metric="hellinger"):
    """H=sqrt(max(0,1-sum(sqrt(p*q)))); similarity=1-H.

    Intersection=sum(min(p,q)). Empty on either side => similarity 0,
    including empty/empty. Scores are not relevance probabilities.
    """
    p, q = validate_descriptor(left), validate_descriptor(right)
    if left.version != right.version or p.shape != q.shape:
        raise ValueError("Incompatible color descriptor versions/dimensions")
    if metric not in {"hellinger", "intersection"}:
        raise ValueError("Unknown histogram metric")
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    if metric == "intersection":
        return float(np.minimum(p, q).sum())
    coefficient = float(np.sqrt(p * q).sum())
    return 1.0 - math.sqrt(max(0.0, 1.0 - min(1.0, coefficient)))
