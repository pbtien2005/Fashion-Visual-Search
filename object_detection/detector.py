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
        self.wide_expand_ratio = float(os.getenv("OBJECT_DETECTION_WIDE_EXPAND_RATIO", "0.30"))
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

    def _make_crop_variant(
        self,
        image,
        bbox: list[int],
        image_width: int,
        image_height: int,
        name: str,
        expand_ratio: float,
        output_size: int = 224,
    ) -> dict | None:
        import cv2
        import numpy as np

        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        expand_x = int(round(box_width * expand_ratio))
        expand_y = int(round(box_height * expand_ratio))

        crop_x1 = max(0, x1 - expand_x)
        crop_y1 = max(0, y1 - expand_y)
        crop_x2 = min(image_width, x2 + expand_x)
        crop_y2 = min(image_height, y2 + expand_y)

        if (crop_x2 - crop_x1) < 50 or (crop_y2 - crop_y1) < 50:
            return None

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        source_crop_height, source_crop_width = crop.shape[:2]
        square_size = max(source_crop_width, source_crop_height)
        square_crop = np.full((square_size, square_size, 3), 255, dtype=crop.dtype)
        offset_x = (square_size - source_crop_width) // 2
        offset_y = (square_size - source_crop_height) // 2
        square_crop[
            offset_y:offset_y + source_crop_height,
            offset_x:offset_x + source_crop_width,
        ] = crop
        crop_resized = cv2.resize(square_crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
        strategy = "tight_pad_square_resize_224" if expand_ratio == 0 else (
            f"expand_{int(round(expand_ratio * 100))}_pad_square_resize_224"
        )

        return {
            "name": name,
            "image": crop_resized,
            "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_width": output_size,
            "crop_height": output_size,
            "source_crop_width": source_crop_width,
            "source_crop_height": source_crop_height,
            "crop_strategy": strategy,
        }

    def _process(self, image) -> list[dict]:
        model = self._ensure_model()
        height, width = image.shape[:2]
        results = model.predict(image, conf=self.conf, verbose=False)

        crops = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                original_bbox = [x1, y1, x2, y2]
                variant_specs = [
                    ("tight", 0.0),
                    ("padded", self.expand_ratio),
                    ("wide", self.wide_expand_ratio),
                ]
                variants = [
                    variant
                    for name, expand_ratio in variant_specs
                    if (
                        variant := self._make_crop_variant(
                            image=image,
                            bbox=original_bbox,
                            image_width=width,
                            image_height=height,
                            name=name,
                            expand_ratio=expand_ratio,
                        )
                    )
                ]
                if not variants:
                    continue

                primary = next((variant for variant in variants if variant["name"] == "padded"), variants[0])
                class_name = CLASS_NAMES.get(class_id)
                if class_name is None:
                    class_name = getattr(model, "names", {}).get(class_id, f"class_{class_id}")

                crops.append(
                    {
                        "image": primary["image"],
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_name_vi": get_class_name_vi(class_name),
                        "confidence": confidence,
                        "bbox": original_bbox,
                        "original_bbox": original_bbox,
                        "expanded_bbox": primary["bbox"],
                        "padded_bbox": primary["bbox"],
                        "crop_width": primary["crop_width"],
                        "crop_height": primary["crop_height"],
                        "source_crop_width": primary["source_crop_width"],
                        "source_crop_height": primary["source_crop_height"],
                        "crop_strategy": primary["crop_strategy"],
                        "variants": variants,
                    }
                )

        return crops
