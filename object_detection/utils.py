import base64


def ndarray_to_base64_jpeg(image) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode crop as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def crops_to_response(crops: list[dict]):
    try:
        from .schemas import CropResult, CropVariant, DetectionResponse
    except ImportError:
        from schemas import CropResult, CropVariant, DetectionResponse

    results = []
    for crop in crops:
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
            )
        )
    return DetectionResponse(success=True, total_crops=len(results), crops=results)
