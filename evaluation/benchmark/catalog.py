"""Exact cosine retrieval over the catalog export."""

import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np

CATALOG_PATH = Path("data/fashion_with_vectors.json")
PRODUCT_FIELDS = (
    "title", "brand", "price", "category", "dominant_color_rgb", "dominant_color_name",
    "embedding_model", "embedding_version",
)


class Catalog:
    @classmethod
    def load(cls, path=CATALOG_PATH):
        path = Path(path)
        raw = path.read_bytes()
        # Hash the exact bytes backing this index.
        return cls(json.loads(raw), sha256=hashlib.sha256(raw).hexdigest())

    def __init__(self, rows, dimension=512, sha256=""):
        self.sha256 = sha256
        self.dimension = dimension
        self.products = []
        vectors = []
        seen = set()
        for row in sorted(rows, key=lambda item: str(item.get("id") or "")):
            try:
                vector = np.asarray(row.get("embedding"), dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                continue
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                continue
            raw_id = row.get("id")
            product_id = "" if raw_id is None else str(raw_id)
            if not product_id or product_id in seen:
                raise ValueError(f"Duplicate or empty product ID: {product_id}")
            seen.add(product_id)
            vectors.append(vector / norm)
            self.products.append({
                "id": product_id,
                **{field: row.get(field) for field in PRODUCT_FIELDS},
                "product_url": row.get("original_url"),
                "image_url": row.get("r2_image_url"),
            })
        self.skipped = len(rows) - len(vectors)
        if not vectors:
            raise ValueError("Catalog has no valid embeddings")
        self._matrix = np.stack(vectors)
        self._by_id = {product["id"]: product for product in self.products}
        if self.skipped:
            print(f"Catalog: skipped {self.skipped} invalid embeddings")

    def ordered_by_similarity(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(f"Expected a vector of dimension {self.dimension}")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Expected a finite, nonzero vector")
        scores = np.clip(self._matrix @ (vector / norm), -1.0, 1.0)
        # Index order is already by product ID, so stable sorting resolves ties.
        for index in np.argsort(-scores, kind="stable"):
            yield {**self.products[index], "similarity": float(scores[index])}

    def search(self, vector: list[float], top_k: int = 100) -> list[dict]:
        return list(islice(self.ordered_by_similarity(vector), max(1, min(top_k, 100))))

    def get(self, product_id: str) -> dict | None:
        return self._by_id.get(str(product_id))

    def check_embedding_space(self, model_name, version):
        spaces = {(item["embedding_model"], item["embedding_version"]) for item in self.products}
        if spaces != {(model_name, version)}:
            raise ValueError("Embedding model/version differs from the catalog")

