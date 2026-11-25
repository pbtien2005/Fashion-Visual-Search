"""Run YOLO once per benchmark query and save the raw boxes.

Runs inside the existing `fashion-visual-search-object-detection` image (it has ultralytics,
opencv and torch). Detection does not depend on the embedding model, so every model replays
these boxes through the unchanged `ClothingDetector.detect_prepared` code (see replay.py).

    evaluation_new/docker_run.sh detect python -m evaluation_new.detect_cache
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2

from object_detection.image_preprocessing import PreprocessConfig, prepare_image

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {"validation": ROOT / "evaluation/bench_validation.csv",
             "test": ROOT / "evaluation/bench_test.csv"}
MODEL = ROOT / "object_detection/best.pt"
OUT = ROOT / "evaluation_new/cache/detections.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry_key(image_sha256, conf) -> str:
    return f"{image_sha256}:{conf}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    entries = {}
    for split, manifest in MANIFESTS.items():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                image_bytes = (manifest.parent / row["image"]).read_bytes()
                # Same preparation as the benchmark's identity configs (A0, C2, B0, C3).
                prepared = prepare_image(image_bytes, PreprocessConfig())
                bgr = cv2.cvtColor(prepared.detection_rgb, cv2.COLOR_RGB2BGR)
                results = model.predict(bgr, conf=args.conf, verbose=False)
                boxes = [
                    {"xyxy": [float(v) for v in box.xyxy[0]], "cls": float(box.cls[0]), "conf": float(box.conf[0])}
                    for result in results if result.boxes is not None for box in result.boxes
                ]
                entries[entry_key(sha256(image_bytes), args.conf)] = {
                    "query_id": row["query_id"], "split": split,
                    # The array YOLO saw; replay checks size and reports pixel drift.
                    "detection_sha256": sha256(bgr.tobytes()), "width": bgr.shape[1], "height": bgr.shape[0],
                    "boxes": boxes,
                }
                print(f"{split} {row['query_id']}: {len(boxes)} box", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256(MODEL.read_bytes()), "conf": args.conf,
        "versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python")},
        "names": {str(k): v for k, v in model.names.items()},
        "entries": entries,
    }, indent=1), encoding="utf-8")
    print(f"{len(entries)} queries -> {args.out}")


if __name__ == "__main__":
    main()
