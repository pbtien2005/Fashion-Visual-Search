"""Render reference/prediction overlays from an existing M2 run, without YOLO.

Recompute only deterministic ROI segmentation and assert it matches saved IoU.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation.m2_annotations import load_manifest, read_json, reference_for, workspace_path, write_json, file_hash
from evaluation.m2_metrics import mask_iou, place_roi_mask
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.segmentation import MaskConfig, segment_roi


def overlay(source, mask, color):
    image = source.copy()
    if mask is not None:
        image[mask] = (.55 * image[mask] + .45 * np.array(color)).astype(np.uint8)
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report = read_json(args.results)
    if report["manifest_sha256"] != file_hash(Path(args.manifest)):
        raise ValueError("Report and annotation manifest differ")
    rows = {(r["query_id"], r["mask_mode"]): r for r in report["rows"]}
    references = [q for q in manifest["queries"] if q["mask_status"] == "human_reviewed"]
    width, height = 250, 300
    sheet = Image.new("RGB", (width * 3, height * len(references)), "white")
    audit = []
    for index, query in enumerate(references):
        prepared = prepare_image(workspace_path(query["image"]).read_bytes(), PreprocessConfig(**manifest["preprocess_config"]))
        source = prepared.source_rgb
        reference = reference_for(query, prepared.geometry_meta)
        ys, xs = np.nonzero(reference)
        margin = 30
        view = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
                min(source.shape[1], int(xs.max()) + margin + 1), min(source.shape[0], int(ys.max()) + margin + 1))
        panels = [(overlay(source, reference, (0, 230, 170)), "USER-REVIEWED REFERENCE")]
        for mode in ("raw", "open_close"):
            row = rows[query["query_id"], mode]
            mask = None
            if row["matched"]:
                x1, y1, x2, y2 = row["predicted_bbox"]
                result = segment_roi(source[y1:y2, x1:x2], MaskConfig(**report["mask_configs"][mode]))
                if result.mask is not None:
                    mask = place_roi_mask(result.mask, row["predicted_bbox"], (source.shape[1], source.shape[0]))
            value = mask_iou(mask, reference) if mask is not None else 0.0
            if not math.isclose(value, row["mask_iou_end_to_end"], abs_tol=1e-12):
                raise ValueError("Reconstructed overlay does not match recorded mask IoU")
            caption = f"{mode} IoU={value:.3f}" + (" | MISS" if not row["matched"] else " | FALLBACK" if mask is None else "")
            panels.append((overlay(source, mask, (255, 135, 20)), caption))
            audit.append(dict(query_id=query["query_id"], mode=mode, saved_IoU=value, recomputed_exact=True))
        for column, (image, title) in enumerate(panels):
            image = image.crop(view)
            image.thumbnail((width - 12, height - 55))
            x, y = column * width, index * height
            sheet.paste(image, (x + (width - image.width) // 2, y + 47))
            draw = ImageDraw.Draw(sheet)
            draw.text((x + 5, y + 5), query["query_id"], fill="black")
            draw.text((x + 5, y + 22), title, fill="black")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    write_json(path.with_suffix(".audit.json"), {"scope": "reference_and_saved_prediction_masks_no_model_inference", "checks": audit})
    print(f"Rendered {len(references)} references; {len(audit)} saved IoUs reproduced exactly")


if __name__ == "__main__":
    main()
