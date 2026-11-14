from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CropVariant(BaseModel):
    name: str
    image_base64: str
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    crop_width: int | None = None
    crop_height: int | None = None
    source_crop_width: int | None = None
    source_crop_height: int | None = None
    crop_strategy: str | None = None


class RoiMetadata(BaseModel):
    bbox_xyxy: list[int] = Field(..., min_length=4, max_length=4)
    raw_bbox_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    working_width: int = Field(..., gt=0)
    working_height: int = Field(..., gt=0)
    roi_width: int = Field(..., gt=0)
    roi_height: int = Field(..., gt=0)
    coordinate_space: Literal["m1_working_image"]
    color_space: Literal["RGB"]
    roi_source: str
    roi_status: Literal["masked", "unmasked", "fallback"]
    fallback_reason: str | None = None
    mask_method: str
    mask_config_id: str
    geometry_mode: str
    geometry_meta: dict = Field(default_factory=dict)
    quality_meta: dict = Field(default_factory=dict)
    segmentation: dict = Field(default_factory=dict)


class DetectionDiagnostics(BaseModel):
    raw_box_count: int = Field(default=0, ge=0)
    invalid_bbox_count: int = Field(default=0, ge=0)
    invalid_prediction_count: int = Field(default=0, ge=0)
    no_valid_variant_count: int = Field(default=0, ge=0)
    skipped_variant_count: int = Field(default=0, ge=0)
    delivered_crop_count: int = Field(default=0, ge=0)


class CropResult(BaseModel):
    class_id: int
    class_name: str
    class_name_vi: str
    confidence: float
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    image_base64: str
    original_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    expanded_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    padded_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    crop_width: int | None = None
    crop_height: int | None = None
    source_crop_width: int | None = None
    source_crop_height: int | None = None
    crop_strategy: str | None = None
    crop_variants: list[CropVariant] = Field(default_factory=list)
    query_id: str | None = None
    garment_id: str | None = None
    source_crop_png_base64: str | None = None
    mask_png_base64: str | None = None
    roi_metadata: RoiMetadata | None = None


class DetectionResponse(BaseModel):
    success: bool
    total_crops: int
    crops: list[CropResult]
    message: Optional[str] = None
    schema_version: str | None = None
    query_id: str | None = None
    diagnostics: DetectionDiagnostics | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_path: str
    model_available: bool
    model_loaded: bool
