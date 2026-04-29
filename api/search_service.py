import os
from urllib.parse import quote, urlparse

from AI.color_utils import color_similarity, extract_dominant_color, name_color
from AI.embedding_model import get_embedding_config
from AI.vectorizer import extract_vector, image_from_base64
from api.category_mapping import get_category_terms
from db.database import search_similar_by_vector, validate_vector_dimension

CANDIDATE_LIMIT = 50
IMAGE_WEIGHT = 0.75
CATEGORY_WEIGHT = 0.15
COLOR_WEIGHT = 0.10
NO_CONFIDENT_MATCH_MESSAGE = "Detected item, but no confident product match found"
# Default guardrail: below this reranked score, the API should avoid showing weak matches as recommendations.
DEFAULT_MATCH_SCORE_THRESHOLD = 0.45


def get_match_score_threshold() -> float:
    raw_threshold = os.getenv("MATCH_SCORE_THRESHOLD")
    if raw_threshold is None or not raw_threshold.strip():
        return DEFAULT_MATCH_SCORE_THRESHOLD

    try:
        return float(raw_threshold)
    except ValueError:
        return DEFAULT_MATCH_SCORE_THRESHOLD


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
        "final_score": product.get("final_score"),
        "match_reason": product.get("match_reason"),
        "dominant_color_rgb": product.get("dominant_color_rgb"),
        "dominant_color_name": product.get("dominant_color_name"),
        "embedding_model": product.get("embedding_model"),
        "embedding_version": product.get("embedding_version"),
    }


def _text_for_category_match(product: dict) -> str:
    return " ".join(
        str(product.get(field) or "").lower()
        for field in ("category", "title", "brand")
    )


def compute_category_score(product: dict, category_terms: list[str]) -> float:
    if not category_terms:
        return 0.0

    haystack = _text_for_category_match(product)
    if not haystack:
        return 0.0

    for term in category_terms:
        normalized = term.strip().lower()
        if normalized and normalized in haystack:
            return 1.0
    return 0.0


def _match_reason(category_score: float, color_score: float) -> str:
    reasons = []
    if category_score:
        reasons.append("category match boosted")
    if color_score >= 0.75:
        reasons.append("similar color")
    return ", ".join(reasons) if reasons else "visual similarity only"


def rerank_products(
    products: list[dict],
    category_terms: list[str],
    top_k: int,
    query_color_rgb=None,
    query_color_name: str | None = None,
) -> list[dict]:
    reranked = []
    for product in products:
        image_similarity = float(product.get("similarity") or 0.0)
        category_score = compute_category_score(product, category_terms)
        color_score = color_similarity(
            query_color_rgb,
            product.get("dominant_color_rgb"),
            query_name=query_color_name,
            product_name=product.get("dominant_color_name"),
        )
        final_score = (
            (IMAGE_WEIGHT * image_similarity)
            + (CATEGORY_WEIGHT * category_score)
            + (COLOR_WEIGHT * color_score)
        )

        ranked_product = {
            **product,
            "image_similarity": image_similarity,
            "category_score": category_score,
            "color_score": color_score,
            "final_score": final_score,
            "similarity": final_score,
            "match_reason": _match_reason(category_score, color_score),
        }
        reranked.append(ranked_product)

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked[:top_k]


def apply_match_threshold(products: list[dict], threshold: float | None = None) -> dict:
    threshold = get_match_score_threshold() if threshold is None else threshold
    top_score = float(products[0].get("final_score") or products[0].get("similarity") or 0.0) if products else 0.0

    if top_score < threshold:
        return {
            "similar_products": [],
            "is_confident_match": False,
            "message": NO_CONFIDENT_MATCH_MESSAGE,
        }

    return {
        "similar_products": [serialize_product(product) for product in products],
        "is_confident_match": True,
        "message": None,
    }


def search_match_for_crop(crop: dict, top_k: int = 5) -> dict:
    image = image_from_base64(crop["image_base64"])
    query_color_rgb = extract_dominant_color(image)
    query_color_name = name_color(query_color_rgb)

    vector = extract_vector(image)
    if vector is None:
        raise ValueError(f"CLIP vector extraction failed for crop class {crop.get('class_name')}")

    validate_vector_dimension(vector)
    embedding_config = get_embedding_config()
    category_terms = get_category_terms(crop.get("class_name", ""))
    products = search_similar_by_vector(
        vector,
        top_k=max(CANDIDATE_LIMIT, top_k),
        category_terms=None,
        embedding_model=embedding_config.model_name,
        embedding_version=embedding_config.version,
    )
    products = rerank_products(
        products,
        category_terms=category_terms,
        top_k=top_k,
        query_color_rgb=query_color_rgb,
        query_color_name=query_color_name,
    )
    match_result = apply_match_threshold(products)
    match_result["query_color_rgb"] = list(query_color_rgb) if query_color_rgb else None
    match_result["query_color_name"] = query_color_name
    return match_result


def search_products_for_crop(crop: dict, top_k: int = 5) -> list[dict]:
    return search_match_for_crop(crop, top_k=top_k)["similar_products"]
