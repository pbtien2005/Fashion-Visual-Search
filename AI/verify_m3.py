"""Real pretrained FashionCLIP + HSV handoff smoke check, without DB or M2/M4.

Run as a separate process so Hugging Face offline/cache flags precede imports.
This verifies invariants and round trips; it is not a retrieval benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Local source RGB ROI image")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache/hub"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir.resolve())
    os.environ["HF_HUB_OFFLINE"] = "0" if args.allow_download else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if args.allow_download else "1"

    import cv2
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from AI.color_cache import build_color_cache, catalog_descriptor_for_query, file_hash
    from AI.embedding_model import get_embedding_config, DEVICE
    from AI.features import FeatureBundle, extract_features
    from AI.hsv_descriptor import HSVConfig, histogram_similarity

    config = HSVConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    model_config = get_embedding_config()
    if model_config.model_name != "patrickjohncyh/fashion-clip" or model_config.version != 2:
        parser.error("This smoke check requires the existing FashionCLIP model/version contract")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"check": "m3-real-weights-smoke-v1", "status": "failed",
              "model": model_config.model_name, "embedding_version": model_config.version,
              "color_version": config.version, "config": config.spec(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__, "numpy": np.__version__,
                              "opencv": cv2.__version__, "pillow": Image.__version__, "device": DEVICE},
              "limitations": "Source image is a smoke fixture, not independently annotated garment ROI. "
                              "No YOLO, HTTP, retrieval quality or production catalog verification."}
    try:
        roi = np.asarray(Image.open(args.image).convert("RGB")).copy()
        mask = np.asarray(Image.open(args.mask)).copy() if args.mask else None
        metadata = {"roi_source": "local_smoke_image", "roi_policy": "source_roi_v1"}
        bundle = extract_features("smoke-product", Image.fromarray(roi), roi, mask, config, metadata)
        payload = bundle.to_dict()
        restored = FeatureBundle.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        records = [{"id": "smoke-product", "image": str(args.image.resolve()),
                    **({"mask": str(args.mask.resolve())} if args.mask else {})}]
        cache = build_color_cache(records, Path.cwd(), config, model_config.model_name,
                                  model_config.version, "source_roi_v1")
        catalog = catalog_descriptor_for_query(cache, restored.garment_id, restored.color,
                                                restored.embedding_model, restored.embedding_version)
        if restored.color.values != catalog.values:
            raise ValueError("Query/catalog descriptors differ for identical ROI/mask/config")
        report.update(status="passed", embedding_dimension=len(restored.embedding),
                      embedding_l2=float(np.linalg.norm(restored.embedding)),
                      color_dimension=len(restored.color.values), color_l1=sum(restored.color.values),
                      query_catalog_equal=True, self_similarity=histogram_similarity(restored.color, catalog),
                      mask_status=restored.color.mask_status, fallback_reason=restored.color.fallback_reason,
                      image_sha256=file_hash(args.image),
                      **({"mask_sha256": file_hash(args.mask)} if args.mask else {}))
        for filename, content in (("bundle.json", payload), ("color_cache.json", cache)):
            (args.output / filename).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
        hub_model = args.cache_dir / "models--patrickjohncyh--fashion-clip"
        ref = hub_model / "refs/main"
        if ref.exists():
            revision = ref.read_text().strip()
            report["model_revision"] = revision
            weights = hub_model / "snapshots" / revision / "model.safetensors"
            if weights.exists():
                report["weights_sha256"] = file_hash(weights)
        root = Path(__file__).resolve().parents[1]
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["source_hashes"] = {str(p.relative_to(root)): file_hash(p) for p in
            [root / "AI/features.py", root / "AI/hsv_descriptor.py", root / "AI/embedding_model.py",
             root / "AI/color_cache.py", Path(__file__).resolve()]}
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
