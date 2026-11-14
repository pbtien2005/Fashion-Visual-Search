"""Add explicit assistant polygon suggestions, never promote them to ground truth."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation.m2_annotations import ROOT, load_manifest, read_json, workspace_path, write_json
from object_detection.image_preprocessing import PreprocessConfig, prepare_image


def add_drafts(manifest, draft_spec, output):
    if draft_spec.get("preprocess_config") != manifest["preprocess_config"]:
        raise ValueError("Draft suggestions were drawn on a different M1 geometry configuration")
    result = deepcopy(manifest)
    panels = []
    for row in result["queries"]:
        spec = draft_spec["drafts"].get(row["query_id"])
        if spec is None:
            continue
        if spec.get("image_sha256") != row["sha256"]:
            raise ValueError("Draft image hash differs from annotation image")
        if row.get("mask_status") == "human_reviewed" or row.get("mask_path"):
            raise ValueError("Refusing to replace an existing reference mask")
        x1, y1, x2, y2 = spec["crop_box"]
        row["mask_polygons"] = [{"points": [[x + x1, y + y1] for x, y in p["points"]],
                                  "subtract": p.get("subtract", False)} for p in spec["polygons"]]
        row.update(mask_status="pending_human_review", mask_reviewer=None,
                   mask_origin="assistant_polygon_suggestion_unreviewed",
                   notes="Mask xanh ngọc là gợi ý AI, chưa phải ground truth; xem và sửa da/tóc/nền/vải nếu cần.")
        prepared = prepare_image(workspace_path(row["image"]).read_bytes(), PreprocessConfig(**result["preprocess_config"]))
        source = Image.fromarray(prepared.source_rgb)
        mask = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in row["mask_polygons"]:
            draw.polygon([tuple(p) for p in polygon["points"]], fill=0 if polygon["subtract"] else 255)
        values = np.array(source)
        selected = np.asarray(mask) != 0
        values[selected] = (.6 * values[selected] + .4 * np.array([0, 225, 235])).astype(np.uint8)
        original = source.crop((x1, y1, x2, y2))
        overlay = Image.fromarray(values).crop((x1, y1, x2, y2))
        pair = Image.new("RGB", (600, 440), "white")
        ImageDraw.Draw(pair).text((8, 5), row["query_id"] + " | AI DRAFT - NOT GT", fill="black")
        for image, x in ((original, 0), (overlay, 300)):
            image.thumbnail((290, 400))
            pair.paste(image, (x + (300 - image.width) // 2, 30))
        panels.append(pair)
    contact = Image.new("RGB", (600, 440 * len(panels)), "white")
    for index, panel in enumerate(panels):
        contact.paste(panel, (0, index * 440))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    contact.save(output, quality=90)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/m2/annotations.m1.pending.json")
    parser.add_argument("--drafts", default="evaluation/m2/mask_drafts.ai.json")
    parser.add_argument("--output", default="evaluation/m2/annotations.m1.with_drafts.pending.json")
    parser.add_argument("--contact", default="docs/m2/phase4/mask_drafts_contact.jpg")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit("Draft output exists; choose a new output to preserve review state")
    result = add_drafts(load_manifest(args.manifest), read_json(args.drafts), args.contact)
    write_json(args.output, result)
    print("Prepared 5 assistant mask suggestions; all remain pending human review")


if __name__ == "__main__":
    main()
