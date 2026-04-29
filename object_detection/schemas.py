from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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


class DetectionResponse(BaseModel):
    success: bool
    total_crops: int
    crops: list[CropResult]
    message: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_path: str
    model_available: bool
    model_loaded: bool
