import os
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

from AI.vectorizer import get_r2_client
from AI.production_color import decode_uploaded_rgb
from api.detection_client import DetectionClient, DetectionServiceError
from api.schemas import DetectedItemResult, ProductListResponse, SearchUploadResponse
from api.search_service import search_match_for_crop, serialize_product
from db.database import get_product_by_id, list_products

MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Fashion Visual Search API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _env_status() -> dict:
    return {
        "object_detection_url": os.getenv("OBJECT_DETECTION_URL", "http://object-detection:8001"),
        "db_host_configured": bool(os.getenv("DB_HOST")),
        "db_name_configured": bool(os.getenv("DB_NAME")),
        "api_port": os.getenv("API_PORT", "8000"),
    }


@app.get("/health")
async def health_check():
    detection = {"available": False, "detail": None}
    try:
        detection = {"available": True, "detail": await DetectionClient().health()}
    except Exception as exc:
        detection = {"available": False, "detail": str(exc)}

    return {
        "success": True,
        "status": "ok",
        "env": _env_status(),
        "object_detection": detection,
    }


@app.post("/api/search/upload", response_model=SearchUploadResponse)
async def search_upload(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    try:
        detection = await DetectionClient().detect_upload(
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
            content_type=file.content_type,
        )
    except DetectionServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    crops = detection.get("crops") or []
    if not detection.get("success"):
        if not crops:
            return SearchUploadResponse(
                success=True,
                detected_items=[],
                message=detection.get("message") or "No clothing items found",
            )
        raise HTTPException(status_code=502, detail=detection.get("message") or "Detection service failed")

    detected_items = []
    try:
        original_rgb = decode_uploaded_rgb(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode uploaded RGB image") from exc
    for crop in crops:
        try:
            match_result = search_match_for_crop(crop, top_k=top_k, original_rgb=original_rgb)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed for crop {crop.get('class_name')}: {exc}") from exc

        detected_items.append(
            DetectedItemResult(
                class_name=crop["class_name"],
                class_name_vi=crop.get("class_name_vi") or crop["class_name"],
                confidence=float(crop["confidence"]),
                bbox=crop["bbox"],
                crop_image_base64=crop["image_base64"],
                similar_products=match_result["similar_products"],
                low_confidence_products=match_result.get("low_confidence_products", []),
                is_confident_match=match_result["is_confident_match"],
                message=match_result["message"],
                reject_reason=match_result.get("reject_reason"),
                top_final_score=match_result.get("top_final_score"),
                top_image_similarity=match_result.get("top_image_similarity"),
                match_score_threshold=match_result.get("match_score_threshold"),
                min_image_similarity=match_result.get("min_image_similarity"),
                query_color_rgb=match_result.get("query_color_rgb"),
                query_color_name=match_result.get("query_color_name"),
                query_style_label=match_result.get("query_style_label"),
                original_bbox=crop.get("original_bbox") or crop.get("bbox"),
                expanded_bbox=crop.get("expanded_bbox"),
                padded_bbox=crop.get("padded_bbox"),
                crop_width=crop.get("crop_width"),
                crop_height=crop.get("crop_height"),
                source_crop_width=crop.get("source_crop_width"),
                source_crop_height=crop.get("source_crop_height"),
                crop_strategy=crop.get("crop_strategy"),
                crop_variants=crop.get("crop_variants") or [],
                search_variants=match_result.get("search_variants", []),
                search_variant_count=match_result.get("search_variant_count"),
            )
        )

    return SearchUploadResponse(success=True, detected_items=detected_items)


@app.get("/api/products", response_model=ProductListResponse)
async def get_products(
    q: str | None = Query(default=None, max_length=120),
    category: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    try:
        result = list_products(limit=limit, offset=offset, q=q, category=category)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Product list query failed: {exc}") from exc

    return ProductListResponse(
        success=True,
        items=[serialize_product(product) for product in result["items"]],
        total=result["total"],
        limit=limit,
        offset=offset,
    )


@app.get("/api/products/{product_id}/image")
async def get_product_image(product_id: str):
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    image_url = product.get("image_url")
    if not image_url:
        raise HTTPException(status_code=404, detail="Product image not found")

    if image_url.startswith("http://") or image_url.startswith("https://"):
        return RedirectResponse(image_url)

    if not (image_url.startswith("s3://") or image_url.startswith("r2://")):
        raise HTTPException(status_code=400, detail="Product image URL is not supported")

    parsed = urlparse(image_url)
    bucket = parsed.netloc or os.getenv("R2_BUCKET_NAME")
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise HTTPException(status_code=400, detail="Product image URL is invalid")

    required_r2_env = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
    if any(not os.getenv(name) for name in required_r2_env):
        raise HTTPException(status_code=503, detail="R2 image proxy is not configured")

    try:
        response = get_r2_client().get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not fetch product image: {exc}") from exc

    content_type = response.get("ContentType") or "image/jpeg"
    return StreamingResponse(response["Body"], media_type=content_type)
