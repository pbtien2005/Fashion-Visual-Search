"""Target-level metrics on one M1 working canvas. No detector/model imports."""

from collections import Counter
import math

import numpy as np
from PIL import Image

from evaluation.run_m1_ablation import box_iou as _m1_box_iou


def valid_box(box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("Expected finite nonempty xyxy bbox")
    return box


def box_iou(left, right):
    return _m1_box_iou(valid_box(left), valid_box(right))


def bbox_to_working(normalized, width, height):
    valid_box(normalized)
    if min(normalized) < 0 or max(normalized) > 1 or width < 1 or height < 1:
        raise ValueError("Expected normalized oriented bbox and positive working size")
    return [value * dimension for value, dimension in zip(normalized, (width, height, width, height))]


def binary_mask(mask):
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size or not np.isin(values, [0, 1, 255]).all():
        raise ValueError("Expected nonempty 2D binary mask (0/1/255)")
    return values != 0


def mask_iou(predicted, reference):
    predicted, reference = binary_mask(predicted), binary_mask(reference)
    if predicted.shape != reference.shape:
        raise ValueError("Masks must share a canvas")
    union = np.count_nonzero(predicted | reference)
    return float(np.count_nonzero(predicted & reference) / union) if union else 0.0


def place_roi_mask(mask, bbox, size):
    valid_box(bbox)
    width, height = size
    if (any(int(v) != v for v in bbox) or bbox[0] < 0 or bbox[1] < 0
            or bbox[2] > width or bbox[3] > height):
        raise ValueError("ROI bbox must be integer and inside the working canvas")
    x1, y1, x2, y2 = map(int, bbox)
    mask = binary_mask(mask)
    if mask.shape != (y2 - y1, x2 - x1):
        raise ValueError("ROI mask shape differs from bbox; no implicit resize")
    canvas = np.zeros((height, width), dtype=bool)
    canvas[y1:y2, x1:x2] = mask
    return canvas


def normalize_reference_mask(mask, coordinate_space, geometry):
    """Reference PNG pixels have no implicit EXIF. Transform explicitly once."""
    mask = binary_mask(mask)
    sizes = {"raw_image": (geometry["raw_width"], geometry["raw_height"]),
             "oriented_image": (geometry["oriented_width"], geometry["oriented_height"]),
             "m1_working_image": (geometry["width"], geometry["height"])}
    if coordinate_space not in sizes or mask.shape != tuple(reversed(sizes[coordinate_space])):
        raise ValueError("Reference mask geometry/coordinate space mismatch")
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    if coordinate_space == "raw_image":
        transforms = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                      4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                      6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                      8: Image.Transpose.ROTATE_90}
        transform = transforms.get(geometry["exif_orientation"])
        if transform is not None:
            image = image.transpose(transform)
        if image.size != sizes["oriented_image"]:
            raise ValueError("EXIF orientation is inconsistent with reference geometry")
    if coordinate_space != "m1_working_image":
        image = image.resize(sizes["m1_working_image"], Image.Resampling.NEAREST)
    return np.asarray(image) != 0


def score_target(target, predictions, size, reference=None, threshold=.5):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid matching IoU threshold")
    valid_box(target["bbox"])
    if reference is not None and binary_mask(reference).shape != tuple(reversed(size)):
        raise ValueError("Reference must be on the working canvas")
    candidates = [(box_iou(target["bbox"], pred["bbox"]), str(pred.get("garment_id", "")), index)
                  for index, pred in enumerate(predictions) if pred["class_name"] == target["class_name"]]
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    best_iou = candidates[0][0] if candidates else 0.0
    matched = bool(candidates and best_iou >= threshold)
    index = candidates[0][2] if matched else None
    prediction = predictions[index] if matched else {}
    mask = prediction.get("mask")
    delivered = mask is not None and bool(np.count_nonzero(binary_mask(mask)))
    conditional = mask_iou(place_roi_mask(mask, prediction["bbox"], size), reference) if (
        delivered and reference is not None
    ) else None
    roi_meta = prediction.get("roi_metadata", {})
    return {"matched": matched, "best_class_iou": best_iou, "prediction_index": index,
            "garment_id": prediction.get("garment_id"), "predicted_bbox": prediction.get("bbox"),
            "mask_delivered": delivered, "roi_status": roi_meta.get("roi_status"),
            "fallback_reason": roi_meta.get("fallback_reason"),
            "mask_iou_conditional": conditional,
            "mask_iou_end_to_end": (conditional if conditional is not None else 0.0) if reference is not None else None}


def _summary(rows):
    count = len(rows)
    matched = sum(bool(row["matched"]) for row in rows)
    delivered = sum(bool(row["mask_delivered"]) for row in rows)
    result = {"target_count": count, "matched": matched,
              "target_detection_recall": matched / count if count else None,
              "mask_delivered_count": delivered,
              "mask_coverage_over_targets": delivered / count if count else None,
              "mask_coverage_over_matched": delivered / matched if matched else None,
              "errors": sum(bool(row.get("error")) for row in rows),
              "roi_status_counts": dict(Counter(row.get("roi_status") or "no_match_or_error" for row in rows)),
              "fallback_reasons": dict(Counter(row["fallback_reason"] for row in rows if row.get("roi_status") == "fallback"))}
    for key in ("mask_iou_conditional", "mask_iou_end_to_end", "gt_bbox_mask_iou"):
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key + "_n"] = len(values)
        result[key + "_mean"] = float(np.mean(values)) if values else None
    return result


def summarize_rows(rows):
    return {"reviewed": _summary([r for r in rows if r["label_status"] == "human_reviewed"]),
            "unreviewed_diagnostic": _summary([r for r in rows if r["label_status"] != "human_reviewed"]),
            "policy": "Conditional: matched target with delivered mask only; end-to-end: missing detection/mask=0 among reviewed references. Empty denominators=null."}
