"""Embedding models compared by the new benchmark, behind one small interface.

Every backend returns L2-normalised float vectors. `encode_image` takes one PIL image (the
pipeline calls it per crop, like production); `encode_images` batches catalog photos.
`encode_texts` is None for image-only models (DINOv2), which therefore skip the hybrid
configurations that need text/style scores.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short name used in folders and charts
    hf_id: str          # Hugging Face repo
    backend: str        # hf_clip | open_clip | hf_siglip | dinov2
    group: str          # "1" drop-in 512-d, "2" needs another dimension, "baseline"
    label: str


MODELS = {
    spec.key: spec for spec in (
        ModelSpec("fashionclip", "patrickjohncyh/fashion-clip", "hf_clip", "baseline", "FashionCLIP (hiện tại)"),
        ModelSpec("marqo_fashionclip", "Marqo/marqo-fashionCLIP", "open_clip", "1", "Marqo-FashionCLIP"),
        ModelSpec("clip_b16", "openai/clip-vit-base-patch16", "hf_clip", "1", "OpenAI CLIP ViT-B/16"),
        ModelSpec("laion_b32", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "hf_clip", "1", "LAION CLIP ViT-B/32"),
        ModelSpec("marqo_fashionsiglip", "Marqo/marqo-fashionSigLIP", "open_clip", "2", "Marqo-FashionSigLIP"),
        ModelSpec("siglip_b16", "google/siglip-base-patch16-224", "hf_siglip", "2", "SigLIP ViT-B/16"),
        ModelSpec("dinov2_base", "facebook/dinov2-base", "dinov2", "2", "DINOv2 ViT-B/14"),
    )
}


def _unit(features: torch.Tensor) -> np.ndarray:
    features = features.float()
    return (features / features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()


def _rgb(image) -> Image.Image:
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")


class Backend:
    has_text = True

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._text_cache: dict[str, list[float]] = {}

    # Subclasses implement _images(list[PIL]) -> np.ndarray and _texts(list[str]) -> np.ndarray.
    @torch.no_grad()
    def encode_images(self, images, batch_size=32) -> np.ndarray:
        chunks = [self._images([_rgb(im) for im in images[i:i + batch_size]])
                  for i in range(0, len(images), batch_size)]
        return np.concatenate(chunks) if chunks else np.zeros((0, self.dimension), np.float32)

    def encode_image(self, image) -> list[float]:
        return self.encode_images([image])[0].astype(np.float64).tolist()

    @torch.no_grad()
    def encode_texts(self, texts, model_name=None) -> list[list[float]]:
        # Same normalisation and caching contract as AI.embedding_model.encode_texts.
        texts = [" ".join(str(text or "").split()) or "clothing product" for text in texts]
        missing = list(dict.fromkeys(text for text in texts if text not in self._text_cache))
        for i in range(0, len(missing), 64):
            batch = missing[i:i + 64]
            for text, vector in zip(batch, self._texts(batch)):
                self._text_cache[text] = vector.astype(np.float64).tolist()
        return [self._text_cache[text] for text in texts]

    @cached_property
    def dimension(self) -> int:
        return int(self._images([Image.new("RGB", (224, 224), "white")]).shape[1])


class HFClip(Backend):
    """transformers CLIPModel, exactly what AI/embedding_model.py loads for FashionCLIP."""

    def __init__(self, spec):
        super().__init__(spec)
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(spec.hf_id).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(spec.hf_id)

    def _images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        return _unit(self.model.get_image_features(**inputs))

    def _texts(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        return _unit(self.model.get_text_features(**inputs))


class OpenClip(Backend):
    """open_clip hf-hub models (Marqo)."""

    def __init__(self, spec):
        super().__init__(spec)
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(f"hf-hub:{spec.hf_id}")
        self.model = self.model.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{spec.hf_id}")

    def _images(self, images):
        pixels = torch.stack([self.preprocess(image) for image in images]).to(DEVICE)
        return _unit(self.model.encode_image(pixels))

    def _texts(self, texts):
        return _unit(self.model.encode_text(self.tokenizer(texts).to(DEVICE)))


class HFSiglip(Backend):
    def __init__(self, spec):
        super().__init__(spec)
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(spec.hf_id).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(spec.hf_id)

    def _images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        return _unit(self.model.get_image_features(**inputs))

    def _texts(self, texts):
        # SigLIP was trained with fixed 64-token padding.
        inputs = self.processor(text=texts, padding="max_length", max_length=64, truncation=True,
                                return_tensors="pt").to(DEVICE)
        return _unit(self.model.get_text_features(**inputs))


class Dinov2(Backend):
    has_text = False

    def __init__(self, spec):
        super().__init__(spec)
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(spec.hf_id).to(DEVICE).eval()
        self.processor = AutoImageProcessor.from_pretrained(spec.hf_id)

    def _images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        return _unit(self.model(**inputs).pooler_output)  # layer-normed CLS token

    def encode_texts(self, texts, model_name=None):
        raise RuntimeError("DINOv2 has no text encoder")


BACKENDS = {"hf_clip": HFClip, "open_clip": OpenClip, "hf_siglip": HFSiglip, "dinov2": Dinov2}


def load_backend(key: str) -> Backend:
    spec = MODELS[key]
    return BACKENDS[spec.backend](spec)
