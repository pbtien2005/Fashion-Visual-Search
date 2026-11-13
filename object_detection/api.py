import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .detector import ClothingDetector
    from .image_preprocessing import ImageDecodeError
    from .schemas import DetectionResponse, HealthResponse
    from .utils import crops_to_response
except ImportError:
    from detector import ClothingDetector
    from image_preprocessing import ImageDecodeError
    from schemas import DetectionResponse, HealthResponse
    from utils import crops_to_response

app = FastAPI(title="Clothing Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector = ClothingDetector(
    model_path=os.getenv("OBJECT_DETECTION_MODEL_PATH", "/models/object-detection/best.pt"),
    conf=float(os.getenv("OBJECT_DETECTION_CONF", "0.5")),
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_path=detector.model_path,
        model_available=detector.model_available,
        model_loaded=detector.model_loaded,
    )


@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_from_upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not detector.model_available:
        raise HTTPException(status_code=503, detail=f"Missing YOLO model at {detector.model_path}")

    try:
        crops = detector.detect_from_bytes(image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {exc}") from exc

    if not crops:
        return DetectionResponse(
            success=False,
            total_crops=0,
            crops=[],
            message="No clothing items found",
        )

    return crops_to_response(crops)
