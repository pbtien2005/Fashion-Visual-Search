import os

import httpx


class DetectionServiceError(RuntimeError):
    pass


class DetectionClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OBJECT_DETECTION_URL") or "http://object-detection:8001").rstrip("/")
        self.timeout = timeout

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def detect_upload(self, image_bytes: bytes, filename: str, content_type: str) -> dict:
        files = {"file": (filename, image_bytes, content_type)}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/detect/upload", files=files)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise DetectionServiceError(f"Detection service returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise DetectionServiceError(f"Detection service unavailable at {self.base_url}: {exc}") from exc

