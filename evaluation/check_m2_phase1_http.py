"""Probe a real detection service; save a small report without image payloads."""

import argparse
import base64
import json
from pathlib import Path

import cv2
import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def probe(api_url, output):
    manifest = json.loads((ROOT / "evaluation/m2/regression_manifest.json").read_text(encoding="utf-8"))
    query = manifest["queries"][0]
    with httpx.Client(base_url=api_url.rstrip("/"), timeout=120, trust_env=False) as client:
        health_before = client.get("/health")
        health_before.raise_for_status()
        assert health_before.json()["model_available"], "Weights are not mounted"
        response = client.post("/detect/upload", files={
            "file": ("regression.jpg", (ROOT / query["image"]).read_bytes(), "image/jpeg"),
        })
        response.raise_for_status()
        payload = response.json()
        assert payload["success"] and payload["total_crops"] == len(payload["crops"]) > 0
        crop = payload["crops"][0]
        jpeg = base64.b64decode(crop["image_base64"], validate=True)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (224, 224, 3)
        health_after = client.get("/health")
        health_after.raise_for_status()
        assert health_after.json()["model_loaded"]
        rejected = client.post("/detect/upload", files={"file": ("invalid.txt", b"text", "text/plain")})
        assert rejected.status_code == 400
    report = {
        "status": "passed", "mode": "real_http_real_yolo_cpu", "query_id": query["query_id"],
        "health_before": health_before.json(), "health_after": health_after.json(),
        "upload_status": response.status_code, "total_crops": payload["total_crops"],
        "jpeg_shape_hwc": list(decoded.shape), "invalid_mime_status": rejected.status_code,
        "schema": "legacy_only_m2_fields_not_implemented_yet",
        "crops": [{key: item[key] for key in ["class_name", "confidence", "bbox"]} for item in payload["crops"]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "mode", "upload_status", "total_crops", "invalid_mime_status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", default="docs/m2/phase1/http_smoke.json")
    args = parser.parse_args()
    probe(args.api_url, ROOT / args.output)
