import base64
import binascii
import concurrent.futures
import json
import os
from io import BytesIO
from urllib.parse import urlparse

import boto3
import httpx
from dotenv import load_dotenv
from PIL import Image

from AI.color_utils import extract_dominant_color, name_color
from AI.embedding_model import (
    DEFAULT_EMBEDDING_MODEL,
    EXPECTED_VECTOR_DIMENSION,
    encode_image,
    get_embedding_config,
    load_embedding_model,
    validate_embedding_dimension,
)

load_dotenv()

MODEL_ID = DEFAULT_EMBEDDING_MODEL
VECTOR_DIMENSION = EXPECTED_VECTOR_DIMENSION

_r2_client = None


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_r2_client():
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )
    return _r2_client


def get_clip_model():
    """Backward-compatible lazy loader for the configured embedding model."""
    return load_embedding_model()


def fetch_image_for_ai(image_url: str):
    try:
        if image_url.startswith("s3://"):
            parsed = urlparse(image_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            response = get_r2_client().get_object(Bucket=bucket, Key=key)
            image_bytes = response["Body"].read()
            return Image.open(BytesIO(image_bytes)).convert("RGB")

        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

    except Exception as e:
        print(f"Image fetch failed ({image_url}): {e}")
        return None


def image_from_base64(image_base64: str):
    try:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64, validate=True)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image payload: {e}") from e
    except Exception as e:
        raise ValueError(f"Could not decode base64 image as an RGB image: {e}") from e


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    try:
        return Image.fromarray(image).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unsupported image input for CLIP vectorization: {type(image)!r}") from e


def extract_vector(image) -> list:
    """Convert a PIL image to a normalized embedding vector for the configured model."""
    try:
        return encode_image(ensure_pil_rgb(image))
    except Exception as e:
        print(f"Vector extraction failed: {e}")
        return None


def extract_vector_from_base64(image_base64: str) -> list:
    return extract_vector(image_from_base64(image_base64))


def get_product_crop_detection_url() -> str:
    explicit_url = os.getenv("PRODUCT_CROP_DETECTION_URL", "").strip()
    if explicit_url:
        return explicit_url.rstrip("/")

    object_detection_url = os.getenv("OBJECT_DETECTION_URL", "").strip()
    if object_detection_url and "object-detection" not in object_detection_url:
        return object_detection_url.rstrip("/")

    return "http://localhost:8001"


def choose_best_detection_crop(crops: list[dict]) -> dict | None:
    candidates = [crop for crop in crops if crop.get("image_base64")]
    if not candidates:
        return None

    def crop_score(crop: dict) -> float:
        bbox = crop.get("bbox") or crop.get("expanded_bbox") or [0, 0, 1, 1]
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return float(crop.get("confidence") or 0.0) * area

    return max(candidates, key=crop_score)


def detect_product_crop(image, item_id: str | None = None) -> dict | None:
    image = ensure_pil_rgb(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    detection_url = get_product_crop_detection_url()
    timeout = float(os.getenv("PRODUCT_CROP_DETECTION_TIMEOUT", "60"))

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{detection_url}/detect/upload",
                files={"file": (f"{item_id or 'product'}.jpg", buffer.getvalue(), "image/jpeg")},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as e:
        print(f"Product crop detection skipped for {item_id}: {e}")
        return None

    if not payload.get("success"):
        print(f"Product crop detection returned no success for {item_id}: {payload.get('message')}")
        return None

    return choose_best_detection_crop(payload.get("crops") or [])


def select_embedding_image(item: dict, image):
    if not env_flag("USE_PRODUCT_CROP_EMBEDDING", default=False):
        return image, {"embedding_source": "original_image"}

    best_crop = detect_product_crop(image, item_id=str(item.get("id") or item.get("asin") or "product"))
    if not best_crop:
        return image, {"embedding_source": "original_image"}

    try:
        crop_image = image_from_base64(best_crop["image_base64"])
    except ValueError as e:
        print(f"Product crop decode failed for {item.get('id')}: {e}")
        return image, {"embedding_source": "original_image"}

    return crop_image, {
        "embedding_source": "product_crop",
        "product_crop_class_name": best_crop.get("class_name"),
        "product_crop_confidence": best_crop.get("confidence"),
        "product_crop_bbox": best_crop.get("bbox"),
    }


def process_single_item(item: dict) -> dict:
    image_url = item.get("r2_image_url")
    if not image_url:
        return item

    img = fetch_image_for_ai(image_url)
    if img:
        embedding_image, embedding_metadata = select_embedding_image(item, img)
        item.update(embedding_metadata)

        dominant_color = extract_dominant_color(embedding_image)
        if dominant_color:
            item["dominant_color_rgb"] = list(dominant_color)
            item["dominant_color_name"] = name_color(dominant_color)

        vector = extract_vector(embedding_image)
        if vector:
            validate_embedding_dimension(vector)
            embedding_config = get_embedding_config()
            item["embedding"] = vector
            item["embedding_model"] = embedding_config.model_name
            item["embedding_version"] = embedding_config.version
            print(f"-> Created vector for ID: {item.get('id')}")

    return item


def run_ai_pipeline(input_file="data/fashion_metadata.json", output_file="data/fashion_with_vectors.json"):
    print("\n=== STAGE 2: VECTOR EXTRACTION ===")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            items = json.load(f)
    except FileNotFoundError:
        print(f"Missing {input_file}. Run ETL first.")
        return

    embedding_config = get_embedding_config()
    print(
        f"Embedding model: {embedding_config.model_name} "
        f"(version {embedding_config.version}, expected dim {embedding_config.expected_dimension})"
    )
    print(f"Loaded {len(items)} products. Starting threaded vector extraction...")
    processed_items = []

    max_workers = int(os.getenv("VECTORIZER_MAX_WORKERS", "4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_item, items)

        for result in results:
            if "embedding" in result:
                processed_items.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_items, f, ensure_ascii=False)

    print(f"Done. Extracted {len(processed_items)} vectors.")
    print(f"Data ready for PostgreSQL import at: {output_file}")


if __name__ == "__main__":
    run_ai_pipeline()
