from pydantic import BaseModel, Field


class ProductResult(BaseModel):
    id: str
    title: str | None = None
    brand: str | None = None
    price: float | None = None
    category: str | None = None
    product_url: str | None = None
    image_url: str | None = None
    usable_image_url: str | None = None
    similarity: float | None = None
    image_similarity: float | None = None
    category_score: float | None = None
    color_score: float | None = None
    final_score: float | None = None
    match_reason: str | None = None
    dominant_color_rgb: list[int] | None = None
    dominant_color_name: str | None = None
    embedding_model: str | None = None
    embedding_version: int | None = None


class DetectedItemResult(BaseModel):
    class_name: str
    class_name_vi: str
    confidence: float
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    crop_image_base64: str
    similar_products: list[ProductResult]
    is_confident_match: bool | None = None
    message: str | None = None
    query_color_rgb: list[int] | None = None
    query_color_name: str | None = None
    original_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    expanded_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    padded_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    crop_width: int | None = None
    crop_height: int | None = None
    source_crop_width: int | None = None
    source_crop_height: int | None = None
    crop_strategy: str | None = None


class SearchUploadResponse(BaseModel):
    success: bool
    detected_items: list[DetectedItemResult]
    message: str | None = None


class ProductListResponse(BaseModel):
    success: bool
    items: list[ProductResult]
    total: int
    limit: int
    offset: int
