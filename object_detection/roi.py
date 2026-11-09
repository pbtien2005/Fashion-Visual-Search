"""Shared geometry and transport contract for source-image garment ROIs."""

import json
import math
import re
import uuid
from dataclasses import dataclass


SCHEMA_VERSION = "m2.roi.v1"
QUERY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
MINIMUM_VARIANT_SIDE = 50


@dataclass
class DetectionBatch:
    query_id: str
    crops: list[dict]
    diagnostics: dict


def resolve_query_id(query_id: str | None) -> str:
    if query_id is None:
        return uuid.uuid4().hex
    if not isinstance(query_id, str) or re.fullmatch(QUERY_ID_PATTERN, query_id) is None:
        raise ValueError("query_id must contain 1-128 ASCII letters, digits, dot, underscore, colon or hyphen")
    return query_id


def validate_rgb_image(image, name="image"):
    import numpy as np

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError(f"{name} must have nonempty H x W x 3 shape")


def validate_prepared_images(source_rgb, detection_rgb):
    validate_rgb_image(source_rgb, "source_rgb")
    validate_rgb_image(detection_rgb, "detection_rgb")
    if source_rgb.shape != detection_rgb.shape:
        raise ValueError("source_rgb and detection_rgb must have the same geometry")


def copy_metadata(metadata, name):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} must be a JSON-compatible dictionary")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc


def clip_bbox(raw_bbox, width, height):
    if len(raw_bbox) != 4 or not all(math.isfinite(value) for value in raw_bbox):
        return None
    x1, y1, x2, y2 = raw_bbox
    if x2 <= x1 or y2 <= y1:
        return None
    bbox = [max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
            max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))]
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def make_roi_metadata(source_rgb, bbox, raw_bbox, geometry_meta, quality_meta):
    height, width = source_rgb.shape[:2]
    return {
        "bbox_xyxy": bbox.copy(), "raw_bbox_xyxy": raw_bbox,
        "working_width": width, "working_height": height,
        "roi_width": bbox[2] - bbox[0], "roi_height": bbox[3] - bbox[1],
        "coordinate_space": "m1_working_image", "color_space": "RGB",
        "roi_source": "yolo_bbox_tight", "roi_status": "unmasked",
        "fallback_reason": "mask_disabled", "mask_method": "off", "mask_config_id": "m2.mask.off.v1",
        "geometry_mode": geometry_meta.get("mode", "prepared_geometry_unverified"),
        "geometry_meta": geometry_meta, "quality_meta": quality_meta,
    }
