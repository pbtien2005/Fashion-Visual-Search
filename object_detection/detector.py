import os
from pathlib import Path


CLASS_NAMES = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "long_sleeve_outwear",
    3: "vest",
    4: "shorts",
    5: "trousers",
    6: "skirt",
    7: "short_sleeve_dress",
    8: "long_sleeve_dress",
    9: "vest_dress",
}

CLASS_NAMES_VI = {
    "short_sleeve_top": "Áo tay ngắn",
    "long_sleeve_top": "Áo tay dài",
    "long_sleeve_outwear": "Áo khoác tay dài",
    "vest": "Áo vest",
    "shorts": "Quần short",
    "trousers": "Quần dài",
    "skirt": "Váy",
    "short_sleeve_dress": "Đầm tay ngắn",
    "long_sleeve_dress": "Đầm tay dài",
    "vest_dress": "Đầm vest",
}


def get_class_name_vi(class_name: str) -> str:
    try:
        from api.category_mapping import get_class_name_vi as mapped_name

        return mapped_name(class_name)
    except Exception:
        return CLASS_NAMES_VI.get(class_name, class_name)


class ClothingDetector:
    def __init__(self, model_path: str | None = None, conf: float = 0.5, expand_ratio: float = 0.15):
        self.model_path = model_path or os.getenv(
            "OBJECT_DETECTION_MODEL_PATH",
            "/models/object-detection/best.pt",
        )
        self.conf = conf
        self.expand_ratio = expand_ratio
        self._model = None

    @property
    def model_available(self) -> bool:
        return Path(self.model_path).exists()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        if not self.model_available:
            raise FileNotFoundError(f"Missing YOLO model at {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing dependency: ultralytics is required for YOLO detection") from exc

        self._model = YOLO(self.model_path)
        return self._model

    def detect_from_bytes(self, image_bytes: bytes) -> list[dict]:
        import cv2
        import numpy as np

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode uploaded image bytes")
        return self._process(image)

    def _process(self, image) -> list[dict]:
        import cv2
        import numpy as np

        model = self._ensure_model()
        height, width = image.shape[:2]
        results = model.predict(image, conf=self.conf, verbose=False)

        crops = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)
                expand_x = int(round(box_width * self.expand_ratio))
                expand_y = int(round(box_height * self.expand_ratio))

                padded_x1 = max(0, x1 - expand_x)
                padded_y1 = max(0, y1 - expand_y)
                padded_x2 = min(width, x2 + expand_x)
                padded_y2 = min(height, y2 + expand_y)

                if (padded_x2 - padded_x1) < 50 or (padded_y2 - padded_y1) < 50:
                    continue

                crop = image[padded_y1:padded_y2, padded_x1:padded_x2]
                crop_height, crop_width = crop.shape[:2]
                square_size = max(crop_width, crop_height)
                square_crop = np.full((square_size, square_size, 3), 255, dtype=crop.dtype)
                offset_x = (square_size - crop_width) // 2
                offset_y = (square_size - crop_height) // 2
                square_crop[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width] = crop
                crop_resized = cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_AREA)
                class_name = CLASS_NAMES.get(class_id)
                if class_name is None:
                    class_name = getattr(model, "names", {}).get(class_id, f"class_{class_id}")

                crops.append(
                    {
                        "image": crop_resized,
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_name_vi": get_class_name_vi(class_name),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "original_bbox": [x1, y1, x2, y2],
                        "expanded_bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
                        "padded_bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
                        "crop_width": 224,
                        "crop_height": 224,
                        "source_crop_width": crop_width,
                        "source_crop_height": crop_height,
                        "crop_strategy": "expand_15_pad_square_resize_224",
                    }
                )

        return crops
