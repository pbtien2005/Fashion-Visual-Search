"""M3 contract: keep encoder input separate from source color ROI."""
from dataclasses import asdict, dataclass
from io import BytesIO
import base64

import numpy as np
from PIL import Image

from AI.hsv_descriptor import ColorDescriptor, extract_hsv_descriptor


def _decode_rgb(payload):
    payload = payload.split(",", 1)[-1] if payload.startswith("data:image") else payload
    with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
        return np.asarray(image.convert("RGB")).copy()


def color_input_from_detection(detection, original_rgb=None):
    """Accept future lossless ROI payload; reconstruct bbox from original when available.

    mask_base64 must be a binary single-channel PNG in ROI coordinates.
    Invalid mask bytes are passed to the configured invalid-mask policy.
    """
    mask = None
    if detection.get("mask_base64"):
        try:
            raw = base64.b64decode(detection["mask_base64"], validate=True)
            with Image.open(BytesIO(raw)) as image:
                mask = np.asarray(image).copy()
        except Exception:
            mask = np.array([], dtype=np.uint8)
    if detection.get("source_roi_base64"):
        roi = _decode_rgb(detection["source_roi_base64"])
        source = "source_roi_base64"
    elif original_rgb is not None:
        from AI.hsv_descriptor import validate_roi
        validate_roi(original_rgb)
        bbox = detection.get("original_bbox") or detection.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(type(x) is not int for x in bbox)):
            raise ValueError("Integer original bbox required")
        x1, y1, x2, y2 = bbox
        height, width = original_rgb.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid source bbox")
        roi = original_rgb[y1:y2, x1:x2].copy()
        source = "original_image_bbox"
    else:
        roi = _decode_rgb(detection["image_base64"])
        source = "legacy_processed_crop"
    return roi, mask, {"roi_source": source, "roi_policy": "source_roi_v1",
                       "roi_fallback_reason": "source_roi_unavailable" if
                       source == "legacy_processed_crop" else None}


@dataclass
class M3Features:
    garment_id: str
    embedding: list[float]
    embedding_model: str
    embedding_version: int
    color: ColorDescriptor
    metadata: dict

    def to_dict(self):
        return asdict(self)


def extract_features(garment_id, embedding_image, roi_rgb, mask=None, config=None, metadata=None):
    # Lazy import keeps HSV usable without Torch, model downloads, DB or HTTP.
    from AI.embedding_model import encode_image, get_embedding_config, validate_embedding_dimension
    color = extract_hsv_descriptor(roi_rgb, mask, config, metadata)
    embedding_config = get_embedding_config()
    embedding = encode_image(embedding_image, model_name=embedding_config.model_name)
    validate_embedding_dimension(embedding, model_name=embedding_config.model_name)
    return M3Features(str(garment_id), embedding, embedding_config.model_name,
                      embedding_config.version, color, dict(metadata or {}))


def extract_detection_features(garment_id, detection, original_rgb=None, config=None):
    """Additive M2 -> M3 entry point using the existing primary encoder crop.

    Pass original_rgb to recover raw color ROI with today's detector response.
    No duplicate inference or new response fields are added to production search.
    """
    roi, mask, metadata = color_input_from_detection(detection, original_rgb)
    metadata["class_name"] = detection.get("class_name")
    metadata["bbox"] = detection.get("original_bbox") or detection.get("bbox")
    embedding_image = Image.fromarray(_decode_rgb(detection["image_base64"]))
    return extract_features(garment_id, embedding_image, roi, mask, config, metadata)
