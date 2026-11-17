"""Run production crop, feature and rerank logic against an in-memory catalog."""

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

from PIL import Image

from AI.color_utils import color_similarity, extract_dominant_color, name_color
from api.category_mapping import get_category_terms
from object_detection.detector import ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.utils import ndarray_to_base64_jpeg

from .metrics import TIMING_FIELDS

M1_LOCK = Path("evaluation/results/m1_b1_20261109/locked_config.json")
TOP_K = 10
BASELINE = "B0_current"
GAMMA = "B1_gamma"
WHOLE = "A0_whole_image"
CLASSICAL = "C1p_color_only"
EMBEDDING = "C2_embedding_only"
GROUPED = "C4_group_terms"
HSV_COLOR = "C5_hsv_color"
HSV_COLOR_W1 = "C5_hsv_color_w1"
HSV_FLAG = "BENCH_QUERY_HSV"
ROI_FLAG = "BENCH_COLOR_ROI"
METRIC_FLAG = "BENCH_COLOR_METRIC"
COLOR_VARIANTS = ("C6_center", "C6_intersection", "C6_center_intersection")
SWEEP = (BASELINE, "C3_w1", "C3_w2")
VALIDATION_ONLY = SWEEP[1:] + (GROUPED, HSV_COLOR, HSV_COLOR_W1) + COLOR_VARIANTS
PENDING = {"B2": "Mask thô", "B3": "Mask và morphology", "B4": "Cân bằng trắng và HSV",
           "C1": "Histogram HSV", "F": "Cấu hình cuối"}
WEIGHT_KEYS = tuple(f"RERANK_{name}_WEIGHT" for name in ("IMAGE", "CATEGORY", "COLOR", "TEXT_STYLE"))


@dataclass(frozen=True)
class BenchConfig:
    name: str
    kind: str
    preprocess: PreprocessConfig
    env: dict[str, str]
    conf: float = 0.5


def default_configs(m1_lock_path=M1_LOCK):
    gamma = PreprocessConfig(**json.loads(Path(m1_lock_path).read_text(encoding="utf-8"))["candidate"])
    identity = PreprocessConfig()
    defaults = {
        "MATCH_SCORE_THRESHOLD": "0.70", "MIN_IMAGE_SIMILARITY": "0.60",
        "COLOR_MISMATCH_THRESHOLD": "0.35", "COLOR_MISMATCH_PENALTY": "0.08",
        "OPPOSITE_CATEGORY_PENALTY": "0.10", "ENABLE_TEXT_STYLE_RERANK": "true",
    }

    def hybrid(name, weights, preprocess=identity):
        return BenchConfig(name, "rerank", preprocess, defaults | dict(zip(WEIGHT_KEYS, weights, strict=True)))

    current = ("0.60", "0.15", "0.15", "0.10")
    return {
        WHOLE: BenchConfig(WHOLE, "whole_image", identity, {}),
        CLASSICAL: BenchConfig(CLASSICAL, "color_only", identity, {}),
        # Raw cosine order; only the acceptance thresholds apply.
        EMBEDDING: BenchConfig(EMBEDDING, "embedding", identity, {
            key: defaults[key] for key in ("MATCH_SCORE_THRESHOLD", "MIN_IMAGE_SIMILARITY")}),
        BASELINE: hybrid(BASELINE, current),
        GAMMA: hybrid(GAMMA, current, gamma),
        "C3_w1": hybrid("C3_w1", ("0.70", "0.10", "0.10", "0.10")),
        "C3_w2": hybrid("C3_w2", ("0.50", "0.20", "0.20", "0.10")),
        # B0 with the upper-layer classes sharing one category keyword set. Validation only.
        # B0 / C3_w1 plus the query HSV147 descriptor from the source ROI, as production passes it.
        # B0 itself runs without a query descriptor, so its color term is always 0. Validation only.
        HSV_COLOR: BenchConfig(
            HSV_COLOR, "rerank", identity,
            defaults | dict(zip(WEIGHT_KEYS, current, strict=True)) | {HSV_FLAG: "true"}),
        HSV_COLOR_W1: BenchConfig(
            HSV_COLOR_W1, "rerank", identity,
            defaults | dict(zip(WEIGHT_KEYS, ("0.70", "0.10", "0.10", "0.10"), strict=True)) | {HSV_FLAG: "true"}),
        # Color-score variants on top of C5 (same weights as B0). Validation only.
        "C6_center": BenchConfig("C6_center", "rerank", identity,
                                 defaults | dict(zip(WEIGHT_KEYS, current, strict=True))
                                 | {HSV_FLAG: "true", ROI_FLAG: "center60"}),
        "C6_intersection": BenchConfig("C6_intersection", "rerank", identity,
                                       defaults | dict(zip(WEIGHT_KEYS, current, strict=True))
                                       | {HSV_FLAG: "true", METRIC_FLAG: "intersection"}),
        "C6_center_intersection": BenchConfig("C6_center_intersection", "rerank", identity,
                                              defaults | dict(zip(WEIGHT_KEYS, current, strict=True))
                                              | {HSV_FLAG: "true", ROI_FLAG: "center60", METRIC_FLAG: "intersection"}),
        GROUPED: BenchConfig(GROUPED, "rerank", identity,
                             defaults | dict(zip(WEIGHT_KEYS, current, strict=True)) | {"CATEGORY_GROUP_UPPER": "true"}),
    }


@contextmanager
def temporary_env(values):
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def new_timings(default=0.0):
    return dict.fromkeys(TIMING_FIELDS, default)


def measured(timings, phase, function, *args, **kwargs):
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[phase] += (perf_counter() - started) * 1000


def cosine_ranking(products):
    return [
        {**product, "final_score": product["similarity"], "image_similarity": product["similarity"],
         "category_score": 0.0, "color_score": 0.0, "text_style_score": 0.0}
        for product in products
    ]


def _color_selector():
    """Pick the query color descriptor and (optionally) swap the histogram metric for this call."""
    import api.search_service as search_service
    from AI.hsv_descriptor import histogram_similarity

    if os.environ.get(METRIC_FLAG) == "intersection":
        search_service.histogram_similarity = lambda a, b, metric="hellinger": histogram_similarity(
            a, b, "intersection")
    else:
        search_service.histogram_similarity = histogram_similarity

    def pick(variant):
        if os.environ.get(HSV_FLAG) != "true":
            return None
        if os.environ.get(ROI_FLAG) == "center60" and variant.get("query_color_center") is not None:
            return variant["query_color_center"]
        return variant.get("query_color")
    return pick


class InProcessSystem:
    def __init__(self, catalog, model_path, detect_fn=None, embed_fn=None):
        self.catalog = catalog
        self.detector = ClothingDetector(model_path=str(model_path), preprocess_config=PreprocessConfig())
        self.detect_fn = detect_fn
        if embed_fn is None:
            from AI.vectorizer import extract_vector

            embed_fn = extract_vector
        self.embed_fn = embed_fn
        self._image_hash = None
        self._prepared = {}
        self._crops = {}
        self._features = {}

    def warm_text_cache(self, batch_size=64):
        """Encode catalog texts and style prompts once, as a long-running API would have.

        Otherwise the first text/style configuration pays every cache miss and looks
        slower than later configurations that do the same work.
        """
        from AI.embedding_model import get_embedding_config
        from api.search_service import compute_query_style_profile, compute_text_similarity_scores

        started = perf_counter()
        model_name = get_embedding_config().model_name
        probe = [1.0] + [0.0] * (self.catalog.dimension - 1)
        products = self.catalog.products
        for offset in range(0, len(products), batch_size):
            compute_text_similarity_scores(products[offset:offset + batch_size], probe, model_name=model_name)
        compute_query_style_profile(probe, None, model_name)
        return (perf_counter() - started) * 1000

    @property
    def crop_config(self):
        return {"expand_ratio": self.detector.expand_ratio,
                "wide_expand_ratio": self.detector.wide_expand_ratio,
                "output_size": 224}

    def _detect(self, prepared, conf):
        source_bgr = prepared.source_rgb[..., ::-1].copy()
        detection_bgr = prepared.detection_rgb[..., ::-1].copy()
        self.detector.conf = conf
        detect_fn = self.detect_fn or self.detector._process
        return detect_fn(source_bgr, detection_bgr)

    def _features_for(self, key, crops, source_rgb=None):
        from AI.production_color import extract_source_color
        from AI.vectorizer import image_from_base64

        if key not in self._features:
            started = perf_counter()
            timings = {"embed": 0.0, "retrieve": 0.0}
            by_crop = []
            for crop in crops:
                query_color = query_color_center = None
                if source_rgb is not None:
                    x1, y1, x2, y2 = (int(v) for v in crop["bbox"])
                    mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        query_color_center = extract_source_color(
                            source_rgb, {"bbox": [x1 + mx, y1 + my, x2 - mx, y2 - my]}, crop.get("class_name"))
                    # One source-ROI descriptor per detection, shared by all crop variants (as in production).
                    query_color = extract_source_color(source_rgb, {"bbox": [int(v) for v in crop["bbox"]]},
                                                       crop.get("class_name"))
                variants = crop.get("variants") or [{"name": "primary", "image": crop["image"]}]
                features = []
                for variant in variants:
                    image = image_from_base64(ndarray_to_base64_jpeg(variant["image"]))
                    rgb = extract_dominant_color(image)
                    features.append({
                        "name": variant["name"], "image": image, "rgb": rgb, "color": name_color(rgb),
                        "query_color": query_color, "query_color_center": query_color_center,
                    })
                by_crop.append(features)
            self._features[key] = by_crop, timings, (perf_counter() - started) * 1000
        return self._features[key]

    def _with_candidates(self, features, timings):
        for variants in features:
            for variant in variants:
                if "products" in variant or "error" in variant:
                    continue
                try:
                    vector = measured(timings, "embed", self.embed_fn, variant["image"])
                    if vector is None:
                        raise ValueError("Embedding extraction failed")
                    variant["vector"] = list(vector)
                    variant["products"] = measured(timings, "retrieve", self.catalog.search, vector)
                except Exception as exc:
                    variant["error"] = f"{variant['name']}: {exc}"

    def _color_rank(self, crop, variant):
        from api.search_service import compute_category_score

        classname = crop["class_name"]
        terms = get_category_terms(classname)
        ranked = []
        for product in self.catalog.products:
            color = color_similarity(variant["rgb"], product["dominant_color_rgb"],
                                     variant["color"], product["dominant_color_name"])
            category = compute_category_score(product, terms, classname)
            ranked.append({**product, "final_score": 0.5 * color + 0.5 * category,
                           "image_similarity": None, "category_score": category,
                           "color_score": color, "text_style_score": 0.0})
        ranked.sort(key=lambda item: (-item["final_score"], item["id"]))
        return ranked[:TOP_K], [item["id"] for item in ranked[:100]]

    def _rerank(self, crop, variants, config):
        color_for = _color_selector()
        from AI.embedding_model import get_embedding_config
        from api.search_service import (
            PER_VARIANT_RERANK_LIMIT, apply_match_threshold, merge_variant_rankings, rerank_products,
        )

        classname = crop["class_name"]
        terms = get_category_terms(classname)
        ranked_variants = []
        for variant in variants:
            if config.kind == "embedding":
                # Keep raw cosine order even for negative similarities.
                ranking = cosine_ranking(variant["products"][:PER_VARIANT_RERANK_LIMIT])
            else:
                ranking = rerank_products(
                    variant["products"], terms, top_k=PER_VARIANT_RERANK_LIMIT,
                    query_color_rgb=variant["rgb"], query_color_name=variant["color"],
                    class_name=classname, query_vector=variant["vector"],
                    query_color=color_for(variant),
                    embedding_model=get_embedding_config().model_name,
                )
            ranked_variants.append((variant["name"], ranking))
        ranked = merge_variant_rankings(ranked_variants, TOP_K)
        decision = apply_match_threshold(ranked)
        candidates = dict.fromkeys(product["id"] for variant in variants for product in variant["products"])
        return ranked, list(candidates), decision["is_confident_match"], decision["reject_reason"]

    def _rank_detections(self, crops, features, config):
        detections = []
        with temporary_env(config.env):
            for crop, variants in zip(crops, features, strict=True):
                first, error = variants[0], None
                ranked, candidates, accepted, reason = [], [], False, "search_error"
                if config.kind == "color_only":
                    ranked, candidates = self._color_rank(crop, first)
                    accepted, reason = True, None
                else:
                    searchable = [variant for variant in variants if "products" in variant]
                    if searchable:
                        first = next((variant for variant in searchable if variant["rgb"] is not None), searchable[0])
                        ranked, candidates, accepted, reason = self._rerank(crop, searchable, config)
                    else:
                        # Production answers 500 here. Score an empty list instead of losing the run.
                        error = ", ".join(variant["error"] for variant in variants)
                tight_color = next((variant["color"] for variant in variants if variant["name"] == "tight"), None)
                detections.append({
                    "class_name": crop["class_name"], "bbox": [int(value) for value in crop["bbox"]],
                    "confidence": float(crop["confidence"]), "query_color_name": first["color"],
                    "query_color_rgb": list(first["rgb"]) if first["rgb"] is not None else None,
                    "tight_color_name": tight_color,
                    "accepted": accepted, "reject_reason": reason, "ranked": ranked, "candidate_ids": candidates,
                    **({"error": error} if error else {}),
                })
        return detections

    def run_query(self, image_bytes: bytes, config: BenchConfig) -> dict:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash != self._image_hash:
            # The runner evaluates all configurations per image before moving on.
            self._image_hash = image_hash
            self._prepared.clear()
            self._crops.clear()
            self._features.clear()

        timings = new_timings()
        if config.preprocess not in self._prepared:
            prepared = measured(timings, "preprocess", prepare_image, image_bytes, config.preprocess)
            self._prepared[config.preprocess] = prepared, timings["preprocess"]
        prepared, timings["preprocess"] = self._prepared[config.preprocess]

        if config.kind == "whole_image":
            vector = measured(timings, "embed", self.embed_fn, Image.fromarray(prepared.source_rgb))
            found = vector is not None
            products = measured(timings, "retrieve", self.catalog.search, vector) if found else []
            detections = [{
                "class_name": None, "confidence": None,
                "bbox": [0, 0, prepared.geometry_meta["width"], prepared.geometry_meta["height"]],
                "query_color_name": None, "query_color_rgb": None,
                "accepted": found, "reject_reason": None if found else "search_error",
                "ranked": cosine_ranking(products[:TOP_K]), "candidate_ids": [item["id"] for item in products],
                **({} if found else {"error": "Embedding extraction failed"}),
            }]
        else:
            key = config.preprocess, config.conf
            if key not in self._crops:
                crops = measured(timings, "detect", self._detect, prepared, config.conf)
                self._crops[key] = crops, timings["detect"]
            crops, timings["detect"] = self._crops[key]
            features, feature_times, color_ms = self._features_for(key, crops, prepared.source_rgb)
            if config.kind in ("rerank", "embedding"):
                self._with_candidates(features, feature_times)
                timings.update(feature_times)
            # Crop decoding and dominant color are feature extraction, shared by every ROI configuration.
            timings["embed"] += color_ms
            detections = measured(timings, "rerank", self._rank_detections, crops, features, config)

        timings["total"] = sum(value for name, value in timings.items() if name != "total")
        return {
            "geometry": prepared.geometry_meta, "quality": prepared.quality_meta,
            "detections": detections, "timings_ms": timings,
            "detection_measured": config.kind != "whole_image",
        }

