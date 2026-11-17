import os
import math
import threading
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "patrickjohncyh/fashion-clip"
DEFAULT_EMBEDDING_VERSION = 2
EXPECTED_VECTOR_DIMENSION = 512
FASHIONCLIP_MODEL = "patrickjohncyh/fashion-clip"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

_loaded_models: dict[str, tuple[CLIPModel, CLIPProcessor]] = {}
_model_lock = threading.Lock()
_text_embedding_cache: dict[tuple[str, str], list[float]] = {}
_text_cache_lock = threading.Lock()


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    version: int
    expected_dimension: int = EXPECTED_VECTOR_DIMENSION


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_version() -> int:
    raw_version = os.getenv("EMBEDDING_VERSION", str(DEFAULT_EMBEDDING_VERSION)).strip()
    try:
        return int(raw_version)
    except ValueError:
        return DEFAULT_EMBEDDING_VERSION


def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name=get_embedding_model_name(),
        version=get_embedding_version(),
    )


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    try:
        return Image.fromarray(image).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unsupported image input for embedding: {type(image)!r}") from e


def load_embedding_model(model_name: str | None = None):
    model_name = model_name or get_embedding_model_name()
    with _model_lock:
        if model_name not in _loaded_models:
            print(f"Loading embedding model on device {DEVICE}: {model_name}")
            try:
                model = CLIPModel.from_pretrained(model_name).to(DEVICE)
                processor = CLIPProcessor.from_pretrained(model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load embedding model '{model_name}'. "
                    f"Default CLIP is '{DEFAULT_EMBEDDING_MODEL}'. "
                    f"FashionCLIP experiment model is '{FASHIONCLIP_MODEL}'. Original error: {e}"
                ) from e

            model.eval()
            _loaded_models[model_name] = (model, processor)

    return _loaded_models[model_name]


def validate_embedding_dimension(vector: list[float], model_name: str | None = None):
    actual_dimension = len(vector) if vector is not None else 0
    if actual_dimension != EXPECTED_VECTOR_DIMENSION:
        raise ValueError(
            f"Embedding model '{model_name or get_embedding_model_name()}' returned dimension "
            f"{actual_dimension}, but database schema is VECTOR({EXPECTED_VECTOR_DIMENSION}). "
            "Do not change the pgvector schema automatically; create a separate migration/table plan first."
        )
    try:
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("Embedding contains NaN or Inf")
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid embedding values") from exc
    if not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ValueError(f"Embedding must have unit L2 norm, got {norm}")


def encode_image(image, model_name: str | None = None) -> list[float]:
    model_name = model_name or get_embedding_model_name()
    model, processor = load_embedding_model(model_name)
    image = ensure_pil_rgb(image)
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    vector = features.cpu().numpy()[0].tolist()
    validate_embedding_dimension(vector, model_name=model_name)
    return vector


def _normalize_text_input(text: str | None) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized or "clothing product"


def encode_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    model_name = model_name or get_embedding_model_name()
    if not texts:
        return []

    normalized_texts = [_normalize_text_input(text) for text in texts]
    cached_vectors: list[list[float] | None] = []
    missing_texts = []
    missing_indexes = []

    with _text_cache_lock:
        for index, text in enumerate(normalized_texts):
            key = (model_name, text)
            cached = _text_embedding_cache.get(key)
            cached_vectors.append(cached)
            if cached is None:
                missing_texts.append(text)
                missing_indexes.append(index)

    if missing_texts:
        model, processor = load_embedding_model(model_name)
        inputs = processor(
            text=missing_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            features = model.get_text_features(**inputs)

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        new_vectors = features.cpu().numpy().tolist()
        for vector in new_vectors:
            validate_embedding_dimension(vector, model_name=model_name)

        with _text_cache_lock:
            if len(_text_embedding_cache) > 5000:
                _text_embedding_cache.clear()
            for index, text, vector in zip(missing_indexes, missing_texts, new_vectors):
                cached_vectors[index] = vector
                _text_embedding_cache[(model_name, text)] = vector

    return [vector for vector in cached_vectors if vector is not None]


def encode_text(text: str, model_name: str | None = None) -> list[float]:
    return encode_texts([text], model_name=model_name)[0]
