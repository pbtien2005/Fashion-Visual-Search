"""Adapt the upload API to a result that can be scored by the benchmark."""

from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from evaluation.run_eval import post_query_image
from object_detection.image_preprocessing import PreprocessConfig, prepare_image

from .pipeline import TOP_K, new_timings


class HttpSystem:
    def __init__(self, api_url: str, timeout: float = 300):
        self.api_url = api_url
        self.timeout = timeout

    def run_query(self, image_bytes: bytes, filename: str) -> dict:
        prepared = prepare_image(image_bytes, PreprocessConfig())
        with TemporaryDirectory() as folder, httpx.Client(timeout=self.timeout) as client:
            path = Path(folder) / Path(filename).name
            path.write_bytes(image_bytes)
            response, elapsed = post_query_image(client, self.api_url, path, TOP_K)
        if not response.get("success") or not isinstance(response.get("detected_items"), list):
            raise ValueError("Upload API did not return a search result")
        detections = []
        for item in response["detected_items"]:
            detections.append({
                **{field: item.get(field) for field in (
                    "class_name", "confidence", "bbox", "query_color_name", "query_color_rgb", "reject_reason",
                )},
                "accepted": bool(item["is_confident_match"]),
                "ranked": (item.get("similar_products") or item.get("low_confidence_products") or [])[:TOP_K],
                "candidate_ids": [],
            })
        return {
            "geometry": prepared.geometry_meta, "quality": prepared.quality_meta,
            "detections": detections, "candidates_available": False,
            "timings_ms": new_timings(None) | {"total": elapsed},
        }

