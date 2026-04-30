import os
from urllib.parse import quote, urlparse

from AI.color_utils import color_similarity, extract_dominant_color, name_color
from AI.embedding_model import encode_texts, get_embedding_config
from AI.vectorizer import extract_vector, image_from_base64
from api.category_mapping import get_category_terms
from db.database import search_similar_by_vector, validate_vector_dimension

CANDIDATE_LIMIT = 100
PER_VARIANT_RERANK_LIMIT = 20
# Default rerank weights. Visual similarity remains primary, while metadata,
# color, and FashionCLIP text/style signals refine candidates without filtering
# them out when YOLO class/style is uncertain.
DEFAULT_IMAGE_WEIGHT = 0.60
DEFAULT_CATEGORY_WEIGHT = 0.15
DEFAULT_COLOR_WEIGHT = 0.15
DEFAULT_TEXT_STYLE_WEIGHT = 0.10
DEFAULT_COLOR_MISMATCH_THRESHOLD = 0.35
DEFAULT_COLOR_MISMATCH_PENALTY = 0.08
DEFAULT_OPPOSITE_CATEGORY_PENALTY = 0.10
TEXT_SCORE_WEIGHT = 0.65
STYLE_SCORE_WEIGHT = 0.35
NO_CONFIDENT_MATCH_MESSAGE = "Detected item, but no confident product match found"
# Default guardrails: avoid showing visually weak matches just because metadata boosted them.
DEFAULT_MATCH_SCORE_THRESHOLD = 0.70
DEFAULT_MIN_IMAGE_SIMILARITY = 0.60

SUBTYPE_KEYWORDS = {
    "button_down": [
        "button-down",
        "button down",
        "buttondown",
        "camp collar",
        "hawaiian",
        "aloha",
        "dress shirt",
        "collared shirt",
    ],
    "tshirt": ["t-shirt", "t shirt", "tee", "tees", "tshirts", "t-shirts"],
    "polo": ["polo"],
    "henley": ["henley"],
    "tank": ["tank", "cami", "camisole"],
    "blouse": ["blouse"],
    "overshirt": ["overshirt", "shirt jacket", "shacket"],
    "sweater": ["sweater", "pullover", "cardigan"],
    "jacket": ["jacket", "coat", "outwear", "outerwear", "anorak", "trench"],
    "vest": ["vest"],
    "shorts": ["shorts"],
    "trousers": ["pants", "trousers", "slacks"],
    "chino": ["chino", "khaki pants"],
    "cargo": ["cargo"],
    "joggers": ["jogger", "joggers", "sweatpants"],
    "wide_leg": ["wide leg", "wide-leg", "palazzo"],
    "dress_pants": ["dress pants", "formal pants", "suit pants"],
    "jeans": ["jeans", "denim"],
    "skirt": ["skirt"],
    "dress": ["dress", "sundress", "gown"],
}

EXPECTED_SUBTYPES_BY_CLASS = {
    "short_sleeve_top": {"button_down", "tshirt", "polo", "blouse", "tank", "overshirt"},
    "long_sleeve_top": {"button_down", "tshirt", "blouse", "henley", "sweater"},
    "long_sleeve_outwear": {"jacket", "sweater", "vest"},
    "vest": {"vest"},
    "shorts": {"shorts"},
    "trousers": {"trousers", "jeans", "chino", "cargo", "joggers", "wide_leg", "dress_pants"},
    "skirt": {"skirt"},
    "short_sleeve_dress": {"dress"},
    "long_sleeve_dress": {"dress"},
    "vest_dress": {"dress", "vest"},
}

STYLE_TEXT_PROMPTS = {
    "button_down": [
        "a photo of a button down shirt",
        "a photo of a collared short sleeve shirt",
        "a photo of a camp collar shirt",
    ],
    "tshirt": [
        "a photo of a plain t-shirt",
        "a photo of a crew neck tee",
    ],
    "polo": ["a photo of a polo shirt"],
    "henley": ["a photo of a henley shirt"],
    "tank": ["a photo of a tank top"],
    "blouse": ["a photo of a blouse"],
    "overshirt": ["a photo of an overshirt", "a photo of a shirt jacket"],
    "sweater": ["a photo of a sweater", "a photo of a cardigan"],
    "jacket": ["a photo of a jacket", "a photo of a coat"],
    "vest": ["a photo of a vest"],
    "shorts": ["a photo of shorts"],
    "trousers": ["a photo of trousers", "a photo of straight leg pants"],
    "chino": ["a photo of chino pants", "a photo of khaki pants"],
    "cargo": ["a photo of cargo pants"],
    "joggers": ["a photo of jogger pants", "a photo of sweatpants"],
    "wide_leg": ["a photo of wide leg pants", "a photo of palazzo pants"],
    "dress_pants": ["a photo of dress pants", "a photo of formal trousers"],
    "jeans": ["a photo of jeans", "a photo of denim pants"],
    "skirt": ["a photo of a skirt"],
    "dress": ["a photo of a dress"],
}

_style_prompt_vector_cache: dict[str, dict[str, list[list[float]]]] = {}

OPPOSITE_CATEGORY_KEYWORDS = {
    "short_sleeve_top": {"pants", "trousers", "jeans", "shorts", "skirt"},
    "long_sleeve_top": {"pants", "trousers", "jeans", "shorts", "skirt"},
    "long_sleeve_outwear": {"pants", "trousers", "jeans", "shorts", "skirt"},
    "vest": {"pants", "trousers", "jeans", "shorts", "skirt"},
    "shorts": {"shirt", "top", "tee", "dress", "skirt", "jacket"},
    "trousers": {"shirt", "top", "tee", "dress", "skirt", "jacket"},
    "skirt": {"shirt", "top", "tee", "pants", "trousers", "jeans", "jacket"},
    "short_sleeve_dress": {"shirt", "top", "tee", "pants", "trousers", "jeans", "skirt"},
    "long_sleeve_dress": {"shirt", "top", "tee", "pants", "trousers", "jeans", "skirt"},
    "vest_dress": {"shirt", "top", "tee", "pants", "trousers", "jeans", "skirt"},
}


def get_match_score_threshold() -> float:
    raw_threshold = os.getenv("MATCH_SCORE_THRESHOLD")
    if raw_threshold is None or not raw_threshold.strip():
        return DEFAULT_MATCH_SCORE_THRESHOLD

    try:
        return float(raw_threshold)
    except ValueError:
        return DEFAULT_MATCH_SCORE_THRESHOLD


def get_min_image_similarity() -> float:
    raw_threshold = os.getenv("MIN_IMAGE_SIMILARITY")
    if raw_threshold is None or not raw_threshold.strip():
        return DEFAULT_MIN_IMAGE_SIMILARITY

    try:
        return float(raw_threshold)
    except ValueError:
        return DEFAULT_MIN_IMAGE_SIMILARITY


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def text_style_rerank_enabled() -> bool:
    return _env_flag("ENABLE_TEXT_STYLE_RERANK", True)


def get_rerank_weights() -> tuple[float, float, float, float]:
    image_weight = max(0.0, _env_float("RERANK_IMAGE_WEIGHT", DEFAULT_IMAGE_WEIGHT))
    category_weight = max(0.0, _env_float("RERANK_CATEGORY_WEIGHT", DEFAULT_CATEGORY_WEIGHT))
    color_weight = max(0.0, _env_float("RERANK_COLOR_WEIGHT", DEFAULT_COLOR_WEIGHT))
    text_style_weight = max(0.0, _env_float("RERANK_TEXT_STYLE_WEIGHT", DEFAULT_TEXT_STYLE_WEIGHT))
    total = image_weight + category_weight + color_weight + text_style_weight

    if total <= 0:
        return (
            DEFAULT_IMAGE_WEIGHT,
            DEFAULT_CATEGORY_WEIGHT,
            DEFAULT_COLOR_WEIGHT,
            DEFAULT_TEXT_STYLE_WEIGHT,
        )

    return (
        image_weight / total,
        category_weight / total,
        color_weight / total,
        text_style_weight / total,
    )


def get_color_mismatch_threshold() -> float:
    return _env_float("COLOR_MISMATCH_THRESHOLD", DEFAULT_COLOR_MISMATCH_THRESHOLD)


def get_color_mismatch_penalty() -> float:
    return max(0.0, _env_float("COLOR_MISMATCH_PENALTY", DEFAULT_COLOR_MISMATCH_PENALTY))


def get_opposite_category_penalty() -> float:
    return max(0.0, _env_float("OPPOSITE_CATEGORY_PENALTY", DEFAULT_OPPOSITE_CATEGORY_PENALTY))


def make_usable_image_url(product: dict) -> str | None:
    image_url = product.get("image_url")
    if not image_url:
        return None
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url

    if image_url.startswith("s3://") or image_url.startswith("r2://"):
        parsed = urlparse(image_url)
        key = parsed.path.lstrip("/")
        public_base_url = os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if public_base_url and key:
            return f"{public_base_url}/{quote(key)}"

        product_id = product.get("id")
        if product_id:
            return f"/api/products/{quote(str(product_id))}/image"

    return None


def make_product_url(product: dict) -> str | None:
    existing_url = product.get("product_url") or product.get("original_url")
    if isinstance(existing_url, str) and (
        existing_url.startswith("http://") or existing_url.startswith("https://")
    ):
        return existing_url

    product_id = product.get("id")
    if isinstance(product_id, str) and len(product_id) == 10 and product_id.isalnum():
        return f"https://www.amazon.com/dp/{product_id}"
    return None


def serialize_product(product: dict) -> dict:
    image_url = product.get("image_url")
    return {
        "id": product.get("id"),
        "title": product.get("title"),
        "brand": product.get("brand"),
        "price": product.get("price"),
        "category": product.get("category"),
        "product_url": make_product_url(product),
        "image_url": image_url,
        "usable_image_url": make_usable_image_url(product),
        "similarity": product.get("similarity"),
        "image_similarity": product.get("image_similarity"),
        "category_score": product.get("category_score"),
        "color_score": product.get("color_score"),
        "text_score": product.get("text_score"),
        "style_score": product.get("style_score"),
        "text_style_score": product.get("text_style_score"),
        "final_score": product.get("final_score"),
        "match_reason": product.get("match_reason"),
        "category_penalty": product.get("category_penalty"),
        "color_penalty": product.get("color_penalty"),
        "query_style_label": product.get("query_style_label"),
        "product_styles": product.get("product_styles"),
        "dominant_color_rgb": product.get("dominant_color_rgb"),
        "dominant_color_name": product.get("dominant_color_name"),
        "embedding_model": product.get("embedding_model"),
        "embedding_version": product.get("embedding_version"),
        "best_crop_variant": product.get("best_crop_variant"),
        "variant_scores": product.get("variant_scores"),
    }


def _text_for_category_match(product: dict) -> str:
    return " ".join(
        str(product.get(field) or "").lower()
        for field in ("category", "title", "brand")
    )


def _contains_any(text: str, keywords: set[str] | list[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def infer_product_subtypes(product: dict) -> set[str]:
    haystack = _text_for_category_match(product)
    return {
        subtype
        for subtype, keywords in SUBTYPE_KEYWORDS.items()
        if _contains_any(haystack, keywords)
    }


def _dot_unit_vectors(left, right) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _normalize_relative_scores(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []

    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_range = max_score - min_score
    if score_range <= 1e-9:
        return [0.5 if score > 0 else 0.0 for score in raw_scores]

    return [max(0.0, min(1.0, (score - min_score) / score_range)) for score in raw_scores]


def _product_text_for_embedding(product: dict) -> str:
    pieces = [
        product.get("title"),
        product.get("brand"),
        product.get("category"),
        f"{product.get('dominant_color_name')} color" if product.get("dominant_color_name") else None,
    ]
    return ". ".join(str(piece).strip() for piece in pieces if str(piece or "").strip())


def _style_labels_for_class(class_name: str | None) -> list[str]:
    expected_labels = EXPECTED_SUBTYPES_BY_CLASS.get(class_name or "")
    labels = expected_labels or set(STYLE_TEXT_PROMPTS.keys())
    return [label for label in STYLE_TEXT_PROMPTS if label in labels]


def _get_style_prompt_vectors(model_name: str) -> dict[str, list[list[float]]]:
    cached = _style_prompt_vector_cache.get(model_name)
    if cached is not None:
        return cached

    labels = []
    prompts = []
    for label, label_prompts in STYLE_TEXT_PROMPTS.items():
        for prompt in label_prompts:
            labels.append(label)
            prompts.append(prompt)

    vectors = encode_texts(prompts, model_name=model_name)
    grouped: dict[str, list[list[float]]] = {}
    for label, vector in zip(labels, vectors):
        grouped.setdefault(label, []).append(vector)

    _style_prompt_vector_cache[model_name] = grouped
    return grouped


def compute_query_style_profile(
    query_vector: list[float] | None,
    class_name: str | None,
    model_name: str,
) -> dict:
    if not query_vector:
        return {"query_style_label": None, "query_style_scores": {}}

    style_vectors = _get_style_prompt_vectors(model_name)
    labels = _style_labels_for_class(class_name)
    raw_by_label = []
    for label in labels:
        vectors = style_vectors.get(label) or []
        if vectors:
            raw_by_label.append((label, max(_dot_unit_vectors(query_vector, vector) for vector in vectors)))

    if not raw_by_label:
        return {"query_style_label": None, "query_style_scores": {}}

    normalized_scores = _normalize_relative_scores([score for _, score in raw_by_label])
    query_style_scores = {
        label: score
        for (label, _), score in zip(raw_by_label, normalized_scores)
    }
    best_label, _ = max(raw_by_label, key=lambda item: item[1])
    return {
        "query_style_label": best_label,
        "query_style_scores": query_style_scores,
    }


def compute_text_similarity_scores(
    products: list[dict],
    query_vector: list[float] | None,
    model_name: str,
) -> dict[str, float]:
    if not query_vector or not products:
        return {}

    product_keys = [_product_key(product) for product in products]
    product_texts = [_product_text_for_embedding(product) for product in products]
    text_vectors = encode_texts(product_texts, model_name=model_name)
    raw_scores = [_dot_unit_vectors(query_vector, vector) for vector in text_vectors]
    normalized_scores = _normalize_relative_scores(raw_scores)
    return {
        product_key: normalized_score
        for product_key, normalized_score in zip(product_keys, normalized_scores)
    }


def compute_text_style_scores(
    products: list[dict],
    query_vector: list[float] | None,
    class_name: str | None,
    model_name: str | None = None,
) -> dict[str, dict]:
    if not text_style_rerank_enabled() or not query_vector or not products:
        return {}

    model_name = model_name or get_embedding_config().model_name
    try:
        text_scores = compute_text_similarity_scores(products, query_vector, model_name=model_name)
        query_style_profile = compute_query_style_profile(
            query_vector=query_vector,
            class_name=class_name,
            model_name=model_name,
        )
    except Exception as exc:
        print(f"Text/style rerank skipped: {exc}")
        return {}

    query_style_scores = query_style_profile.get("query_style_scores") or {}
    query_style_label = query_style_profile.get("query_style_label")
    scored_products = {}
    for product in products:
        product_key = _product_key(product)
        product_styles = sorted(infer_product_subtypes(product))
        style_score = max((query_style_scores.get(style, 0.0) for style in product_styles), default=0.0)
        text_score = float(text_scores.get(product_key, 0.0))
        text_style_score = (TEXT_SCORE_WEIGHT * text_score) + (STYLE_SCORE_WEIGHT * style_score)
        scored_products[product_key] = {
            "text_score": text_score,
            "style_score": style_score,
            "text_style_score": text_style_score,
            "query_style_label": query_style_label,
            "product_styles": product_styles,
        }

    return scored_products


def compute_category_score(product: dict, category_terms: list[str], class_name: str | None = None) -> float:
    if not category_terms:
        return 0.0

    haystack = _text_for_category_match(product)
    if not haystack:
        return 0.0

    if class_name and _contains_any(haystack, OPPOSITE_CATEGORY_KEYWORDS.get(class_name, set())):
        return 0.0

    score = 0.0
    for term in category_terms:
        normalized = term.strip().lower()
        if normalized and normalized in haystack:
            score = max(score, 0.55)

    if score == 0.0:
        return 0.0

    expected_subtypes = EXPECTED_SUBTYPES_BY_CLASS.get(class_name or "", set())
    product_subtypes = infer_product_subtypes(product)
    if expected_subtypes and product_subtypes & expected_subtypes:
        score += 0.15

    if class_name == "short_sleeve_top":
        if "short sleeve" in haystack or "sleeveless" in haystack:
            score += 0.15
        if "long sleeve" in haystack:
            score -= 0.25
    elif class_name in {"long_sleeve_top", "long_sleeve_outwear", "long_sleeve_dress"}:
        if "long sleeve" in haystack:
            score += 0.15
        if "short sleeve" in haystack or "sleeveless" in haystack:
            score -= 0.20

    return max(0.0, min(score, 0.85))


def compute_category_penalty(product: dict, class_name: str | None = None) -> float:
    if not class_name:
        return 0.0

    haystack = _text_for_category_match(product)
    if not haystack:
        return 0.0

    if _contains_any(haystack, OPPOSITE_CATEGORY_KEYWORDS.get(class_name, set())):
        return get_opposite_category_penalty()

    return 0.0


def compute_color_penalty(
    color_score: float,
    query_color_name: str | None = None,
    product_color_name: str | None = None,
) -> float:
    if not query_color_name or not product_color_name:
        return 0.0
    if query_color_name == product_color_name:
        return 0.0
    if color_score >= get_color_mismatch_threshold():
        return 0.0
    return get_color_mismatch_penalty()


def _match_reason(
    category_score: float,
    color_score: float,
    image_similarity: float,
    text_style_score: float = 0.0,
    style_score: float = 0.0,
    category_penalty: float = 0.0,
    color_penalty: float = 0.0,
) -> str:
    reasons = []
    if image_similarity < DEFAULT_MIN_IMAGE_SIMILARITY:
        reasons.append("weak visual similarity")
    if category_score >= 0.75:
        reasons.append("category match boosted")
    elif category_score > 0:
        reasons.append("broad category match")
    if color_score >= 0.95:
        reasons.append("similar color")
    if text_style_score >= 0.70:
        reasons.append("text/style match")
    elif style_score >= 0.70:
        reasons.append("style match")
    if category_penalty > 0:
        reasons.append("opposite category penalty")
    if color_penalty > 0:
        reasons.append("color mismatch penalty")
    return ", ".join(reasons) if reasons else "visual similarity only"


def rerank_products(
    products: list[dict],
    category_terms: list[str],
    top_k: int,
    query_color_rgb=None,
    query_color_name: str | None = None,
    class_name: str | None = None,
    query_vector: list[float] | None = None,
    embedding_model: str | None = None,
) -> list[dict]:
    reranked = []
    image_weight, category_weight, color_weight, text_style_weight = get_rerank_weights()
    text_style_scores = compute_text_style_scores(
        products,
        query_vector=query_vector,
        class_name=class_name,
        model_name=embedding_model,
    )
    if not text_style_scores and text_style_weight > 0:
        active_weight = image_weight + category_weight + color_weight
        if active_weight > 0:
            image_weight = image_weight / active_weight
            category_weight = category_weight / active_weight
            color_weight = color_weight / active_weight
            text_style_weight = 0.0

    for product in products:
        image_similarity = float(product.get("similarity") or 0.0)
        category_score = compute_category_score(product, category_terms, class_name=class_name)
        color_score = color_similarity(
            query_color_rgb,
            product.get("dominant_color_rgb"),
            query_name=query_color_name,
            product_name=product.get("dominant_color_name"),
        )
        category_penalty = compute_category_penalty(product, class_name=class_name)
        color_penalty = compute_color_penalty(
            color_score,
            query_color_name=query_color_name,
            product_color_name=product.get("dominant_color_name"),
        )
        text_style_info = text_style_scores.get(_product_key(product), {})
        text_score = float(text_style_info.get("text_score") or 0.0)
        style_score = float(text_style_info.get("style_score") or 0.0)
        text_style_score = float(text_style_info.get("text_style_score") or 0.0)
        base_score = (
            (image_weight * image_similarity)
            + (category_weight * category_score)
            + (color_weight * color_score)
            + (text_style_weight * text_style_score)
        )
        final_score = max(0.0, base_score - category_penalty - color_penalty)

        ranked_product = {
            **product,
            "image_similarity": image_similarity,
            "category_score": category_score,
            "color_score": color_score,
            "text_score": text_score,
            "style_score": style_score,
            "text_style_score": text_style_score,
            "category_penalty": category_penalty,
            "color_penalty": color_penalty,
            "query_style_label": text_style_info.get("query_style_label"),
            "product_styles": text_style_info.get("product_styles") or [],
            "final_score": final_score,
            "similarity": final_score,
            "match_reason": _match_reason(
                category_score,
                color_score,
                image_similarity,
                text_style_score=text_style_score,
                style_score=style_score,
                category_penalty=category_penalty,
                color_penalty=color_penalty,
            ),
        }
        reranked.append(ranked_product)

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked[:top_k]


def _product_key(product: dict) -> str:
    return str(
        product.get("id")
        or product.get("product_url")
        or product.get("image_url")
        or product.get("title")
        or id(product)
    )


def _rank_score(product: dict) -> float:
    return float(product.get("final_score") or product.get("similarity") or 0.0)


def merge_variant_rankings(ranked_variants: list[tuple[str, list[dict]]], top_k: int) -> list[dict]:
    """Merge ranked products from several crop variants by keeping each product's best score."""
    merged: dict[str, dict] = {}

    for variant_name, products in ranked_variants:
        for product in products:
            key = _product_key(product)
            score = _rank_score(product)
            existing = merged.get(key)

            if existing is None:
                merged[key] = {
                    **product,
                    "best_crop_variant": variant_name,
                    "variant_scores": {variant_name: score},
                }
                continue

            variant_scores = dict(existing.get("variant_scores") or {})
            variant_scores[variant_name] = max(score, variant_scores.get(variant_name, score))
            if score > _rank_score(existing):
                merged[key] = {
                    **product,
                    "best_crop_variant": variant_name,
                    "variant_scores": variant_scores,
                }
            else:
                existing["variant_scores"] = variant_scores

    products = list(merged.values())
    products.sort(key=_rank_score, reverse=True)
    return products[:top_k]


def apply_match_threshold(
    products: list[dict],
    threshold: float | None = None,
    min_image_similarity: float | None = None,
) -> dict:
    threshold = get_match_score_threshold() if threshold is None else threshold
    min_image_similarity = get_min_image_similarity() if min_image_similarity is None else min_image_similarity
    top_score = float(products[0].get("final_score") or products[0].get("similarity") or 0.0) if products else 0.0
    top_image_similarity = float(products[0].get("image_similarity") or 0.0) if products else 0.0
    reject_reasons = []

    if not products:
        reject_reasons.append("no_candidates")
    if products and top_score < threshold:
        reject_reasons.append("final_score_too_low")
    if products and top_image_similarity < min_image_similarity:
        reject_reasons.append("image_similarity_too_low")

    if reject_reasons:
        return {
            "similar_products": [],
            "low_confidence_products": [serialize_product(product) for product in products],
            "is_confident_match": False,
            "message": NO_CONFIDENT_MATCH_MESSAGE,
            "reject_reason": ", ".join(reject_reasons),
            "top_final_score": top_score if products else None,
            "top_image_similarity": top_image_similarity if products else None,
            "match_score_threshold": threshold,
            "min_image_similarity": min_image_similarity,
        }

    return {
        "similar_products": [serialize_product(product) for product in products],
        "low_confidence_products": [],
        "is_confident_match": True,
        "message": None,
        "reject_reason": None,
        "top_final_score": top_score,
        "top_image_similarity": top_image_similarity,
        "match_score_threshold": threshold,
        "min_image_similarity": min_image_similarity,
    }


def crop_variants_for_search(crop: dict) -> list[dict]:
    variants = []
    for variant in crop.get("crop_variants") or []:
        if variant.get("image_base64"):
            variants.append(
                {
                    "name": variant.get("name") or variant.get("crop_strategy") or "variant",
                    "image_base64": variant["image_base64"],
                }
            )

    if not variants and crop.get("image_base64"):
        variants.append(
            {
                "name": crop.get("crop_strategy") or "primary",
                "image_base64": crop["image_base64"],
            }
        )

    return variants


def _search_single_crop_variant(
    crop: dict,
    variant: dict,
    top_k: int,
    category_terms: list[str],
    embedding_config,
) -> tuple[list[dict], tuple[int, int, int] | None, str | None]:
    image = image_from_base64(variant["image_base64"])
    query_color_rgb = extract_dominant_color(image)
    query_color_name = name_color(query_color_rgb)
    vector = extract_vector(image)
    if vector is None:
        raise ValueError(
            f"Embedding extraction failed for crop class {crop.get('class_name')} "
            f"variant {variant.get('name')}"
        )

    validate_vector_dimension(vector)
    products = search_similar_by_vector(
        vector,
        top_k=max(CANDIDATE_LIMIT, top_k),
        category_terms=None,
        embedding_model=embedding_config.model_name,
        embedding_version=embedding_config.version,
    )
    ranked = rerank_products(
        products,
        category_terms=category_terms,
        top_k=max(top_k, PER_VARIANT_RERANK_LIMIT),
        query_color_rgb=query_color_rgb,
        query_color_name=query_color_name,
        class_name=crop.get("class_name", ""),
        query_vector=vector,
        embedding_model=embedding_config.model_name,
    )
    return ranked, query_color_rgb, query_color_name


def search_match_for_crop(crop: dict, top_k: int = 5) -> dict:
    variants = crop_variants_for_search(crop)
    if not variants:
        raise ValueError(f"Missing crop image payload for class {crop.get('class_name')}")

    embedding_config = get_embedding_config()
    category_terms = get_category_terms(crop.get("class_name", ""))
    ranked_variants = []
    query_color_rgb = None
    query_color_name = None
    errors = []

    for variant in variants:
        try:
            ranked, variant_color_rgb, variant_color_name = _search_single_crop_variant(
                crop=crop,
                variant=variant,
                top_k=top_k,
                category_terms=category_terms,
                embedding_config=embedding_config,
            )
        except Exception as exc:
            errors.append(f"{variant.get('name')}: {exc}")
            continue

        if query_color_rgb is None:
            query_color_rgb = variant_color_rgb
            query_color_name = variant_color_name
        ranked_variants.append((variant.get("name") or "variant", ranked))

    if not ranked_variants:
        raise ValueError("; ".join(errors) or f"No searchable crop variants for class {crop.get('class_name')}")

    products = merge_variant_rankings(ranked_variants, top_k=top_k)
    match_result = apply_match_threshold(products)
    match_result["query_style_label"] = next(
        (product.get("query_style_label") for product in products if product.get("query_style_label")),
        None,
    )
    match_result["query_color_rgb"] = list(query_color_rgb) if query_color_rgb else None
    match_result["query_color_name"] = query_color_name
    match_result["search_variants"] = [variant_name for variant_name, _ in ranked_variants]
    match_result["search_variant_count"] = len(ranked_variants)
    return match_result


def search_products_for_crop(crop: dict, top_k: int = 5) -> list[dict]:
    return search_match_for_crop(crop, top_k=top_k)["similar_products"]
