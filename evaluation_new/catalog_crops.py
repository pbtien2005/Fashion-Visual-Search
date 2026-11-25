"""Rebuild the image each catalog vector was made from, so every model embeds the same pixels.

data/fashion_with_vectors.json was vectorised on 2026-04-30 with USE_PRODUCT_CROP_EMBEDDING on:
1709/1752 products have embedding_source == "product_crop", the rest "original_image".
The crop path at that time (git 0b64c85, AI/vectorizer.py + object_detection/detector.py):

  1. R2 bytes -> PIL RGB -> JPEG quality 90 (detect_product_crop upload)
  2. cv2.imdecode (BGR) in the detection service, YOLO conf 0.5
  3. best box (conf x area) -> int(xyxy) = product_crop_bbox, stored in the catalog
  4. "padded" variant: expand 15 % per side, clip, white square pad, cv2.resize 224 INTER_AREA
  5. cv2.imencode(".jpg") default quality -> base64 -> PIL RGB -> embedding

The box is stored, so steps 4-5 are replayed here without running YOLO again. The FashionCLIP
vectors computed from these images are checked against the stored ones (see verify()).
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

EXPAND_RATIO = 0.15
OUTPUT_SIZE = 224
MIN_SIDE = 50


def _padded_crop(bgr, bbox):
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    expand_x = int(round(max(1, x2 - x1) * EXPAND_RATIO))
    expand_y = int(round(max(1, y2 - y1) * EXPAND_RATIO))
    cx1, cy1 = max(0, x1 - expand_x), max(0, y1 - expand_y)
    cx2, cy2 = min(width, x2 + expand_x), min(height, y2 + expand_y)
    if (cx2 - cx1) < MIN_SIDE or (cy2 - cy1) < MIN_SIDE:
        return None
    crop = bgr[cy1:cy2, cx1:cx2]
    h, w = crop.shape[:2]
    side = max(w, h)
    square = np.full((side, side, 3), 255, dtype=crop.dtype)
    ox, oy = (side - w) // 2, (side - h) // 2
    square[oy:oy + h, ox:ox + w] = crop
    return cv2.resize(square, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)


def embedding_image(row, image_bytes):
    """PIL image the stored vector of `row` was computed from (crop or whole photo)."""
    original = Image.open(BytesIO(image_bytes)).convert("RGB")
    if row.get("embedding_source") != "product_crop" or not row.get("product_crop_bbox"):
        return original
    buffer = BytesIO()
    original.save(buffer, format="JPEG", quality=90)
    bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    crop = _padded_crop(bgr, row["product_crop_bbox"])
    if crop is None:
        return original
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        return original
    return Image.open(BytesIO(encoded.tobytes())).convert("RGB")
