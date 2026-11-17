"""Import M1 labels, build an offline review page, or import explicit human edits."""

import argparse
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw

from evaluation.m2_annotations import (
    ROOT, apply_review, file_hash, import_m1_labels, load_manifest, read_json, workspace_path, write_json,
)
from evaluation.m2_metrics import bbox_to_working
from object_detection.image_preprocessing import PreprocessConfig, prepare_image


def build_review_bundle(manifest, output, *, contact_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previews = {}
    cell_width, cell_height = 250, 290
    sheet = Image.new("RGB", (4 * cell_width, ((len(manifest["queries"]) + 3) // 4) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(manifest["queries"]):
        prepared = prepare_image(workspace_path(row["image"]).read_bytes(), PreprocessConfig(**manifest["preprocess_config"]))
        image = Image.fromarray(prepared.source_rgb)
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=88)
        previews[row["query_id"]] = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
        preview = image.copy()
        ImageDraw.Draw(preview).rectangle(bbox_to_working(row["bbox_normalized"], image.width, image.height), outline="#00ff60", width=3)
        preview.thumbnail((240, 230))
        x, y = (index % 4) * cell_width, (index // 4) * cell_height
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y + 46))
        draw.text((x + 5, y + 5), row["query_id"], fill="black")
        draw.text((x + 5, y + 20), f"{row['split']} | {row['class_name']}", fill="black")
        draw.text((x + 5, y + 276), "MASK REQUESTED" if row["mask_requested"] else "bbox/color review", fill="#9a3100")
    if contact_path:
        Path(contact_path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(contact_path, quality=88)
    template = (ROOT / "evaluation/m2_review.html").read_text(encoding="utf-8")
    payload = json.dumps({"manifest": manifest, "previews": previews}, ensure_ascii=False).replace("<", "\\u003c")
    path = output / "review.html"
    path.write_text(template.replace("__M2_REVIEW_PAYLOAD__", payload), encoding="utf-8")
    write_json(output / "manifest.pending.json", manifest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import-m1", "review", "apply-review"))
    parser.add_argument("--manifest", default="evaluation/m2/annotations.m1.pending.json")
    parser.add_argument("--output", help="Review directory or reviewed-manifest JSON")
    parser.add_argument("--review", help="JSON downloaded from the offline annotation page")
    parser.add_argument("--mask-dir", default="evaluation/m2/reference_masks/review1")
    parser.add_argument("--contact", default="docs/m2/phase4/labels_contact.jpg")
    args = parser.parse_args()
    if args.action == "import-m1":
        path = Path(args.manifest)
        if path.exists():
            raise SystemExit("Manifest exists; choose a new name to preserve annotations")
        manifest = import_m1_labels()
        write_json(path, manifest)
        load_manifest(path)
        print(f"Imported {len(manifest['queries'])} pending M1 targets: {path}")
    elif args.action == "review":
        manifest = load_manifest(args.manifest)
        path = build_review_bundle(manifest, args.output or "output/m2/review/phase4", contact_path=args.contact)
        print(f"Offline review page: {path}")
    else:
        if not args.review or not args.output:
            parser.error("apply-review requires --review and --output")
        if Path(args.output).exists():
            raise SystemExit("Reviewed manifest exists; choose a new output name")
        reviewed = apply_review(load_manifest(args.manifest), read_json(args.review), Path(args.mask_dir))
        write_json(args.output, reviewed)
        print(f"Saved reviewed manifest: {args.output}")


if __name__ == "__main__":
    main()
