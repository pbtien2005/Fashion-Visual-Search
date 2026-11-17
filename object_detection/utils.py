import base64


def binary_mask_to_base64_png(mask) -> str:
    import cv2
    import numpy as np

    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size
            or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8))
            or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError("Mask must be a nonempty binary 2D uint8/bool array")
    ok, buffer = cv2.imencode(".png", (mask != 0).astype(np.uint8) * 255)
    if not ok:
        raise ValueError("Could not encode mask as PNG")
    return base64.b64encode(buffer).decode("ascii")


def rgb_to_base64_png(image) -> str:
    import cv2

    try:
        from .roi import validate_rgb_image
    except ImportError:
        from roi import validate_rgb_image

    validate_rgb_image(image, "source_crop")
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode source crop as PNG")
    return base64.b64encode(buffer).decode("ascii")


def ndarray_to_base64_jpeg(image) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode crop as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def crops_to_response(crops: list[dict], *, query_id=None, diagnostics=None):
    try:
        from .schemas import CropResult, CropVariant, DetectionResponse
    except ImportError:
        from schemas import CropResult, CropVariant, DetectionResponse

    try:
        from .roi import SCHEMA_VERSION
    except ImportError:
        from roi import SCHEMA_VERSION

    results = []
    for crop in crops:
        mask = crop.get("mask")
        if mask is not None and (crop.get("source_rgb") is None or mask.shape != crop["source_rgb"].shape[:2]):
            raise ValueError("Mask and source ROI have different geometry")
        crop_variants = [
            CropVariant(
                name=variant["name"],
                image_base64=ndarray_to_base64_jpeg(variant["image"]),
                bbox=variant["bbox"],
                crop_width=variant.get("crop_width"),
                crop_height=variant.get("crop_height"),
                source_crop_width=variant.get("source_crop_width"),
                source_crop_height=variant.get("source_crop_height"),
                crop_strategy=variant.get("crop_strategy"),
            )
            for variant in crop.get("variants", [])
        ]
        results.append(
            CropResult(
                class_id=crop["class_id"],
                class_name=crop["class_name"],
                class_name_vi=crop["class_name_vi"],
                confidence=crop["confidence"],
                bbox=crop["bbox"],
                image_base64=ndarray_to_base64_jpeg(crop["image"]),
                original_bbox=crop.get("original_bbox"),
                expanded_bbox=crop.get("expanded_bbox"),
                padded_bbox=crop.get("padded_bbox"),
                crop_width=crop.get("crop_width"),
                crop_height=crop.get("crop_height"),
                source_crop_width=crop.get("source_crop_width"),
                source_crop_height=crop.get("source_crop_height"),
                crop_strategy=crop.get("crop_strategy"),
                crop_variants=crop_variants,
                query_id=crop.get("query_id"),
                garment_id=crop.get("garment_id"),
                source_crop_png_base64=rgb_to_base64_png(crop["source_rgb"]) if crop.get("source_rgb") is not None else None,
                mask_png_base64=binary_mask_to_base64_png(mask) if mask is not None else None,
                roi_metadata=crop.get("roi_metadata"),
            )
        )
    query_id = query_id or (crops[0].get("query_id") if crops else None)
    return DetectionResponse(success=True, total_crops=len(results), crops=results,
                             schema_version=SCHEMA_VERSION if query_id else None,
                             query_id=query_id, diagnostics=diagnostics)
