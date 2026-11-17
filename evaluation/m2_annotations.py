"""M2 annotation provenance, validation, and explicit human-review import."""

import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation.m2_metrics import bbox_to_working, binary_mask, normalize_reference_mask
from evaluation.run_m1_ablation import file_hash, load_labels
from object_detection.detector import CLASS_NAMES
from object_detection.image_preprocessing import PreprocessConfig, prepare_image
from object_detection.roi import resolve_query_id


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "m2.annotation.v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def workspace_path(value, root=ROOT):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Expected a relative workspace path")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Annotation path escapes workspace")
    return path


def import_m1_labels():
    development = {row["sha256"] for row in read_json(ROOT / "evaluation/m2/regression_manifest.json")["queries"]}
    selected_masks = {"real_black_dress_02", "real_white_shirt_02", "m1_harajuku", "m1_brisbane", "m1_ostau"}
    queries = []
    for split in ("validation", "test"):
        csv_path = ROOT / f"evaluation/m1_{split}.csv"
        for row in load_labels(csv_path):
            prepared = prepare_image(row["path"].read_bytes(), PreprocessConfig())
            queries.append({"query_id": row["query_id"], "target_id": row["query_id"] + "_target",
                            "image": row["path"].relative_to(ROOT).as_posix(), "sha256": row["sha256"],
                            "split": split, "bbox_normalized": row["bbox"], "bbox_coordinate_space": "oriented_image_normalized",
                            "class_name": row["class_name"], "color_names": [row["color"]],
                            "lighting_group": row["lighting_group"], "source_url": row["source"],
                            "author": row["author"], "license": row["license"], "license_url": row["license_url"],
                            "source_group": row["source"], "source_csv": csv_path.relative_to(ROOT).as_posix(),
                            "source_csv_sha256": file_hash(csv_path), "label_origin": "M1_AI_assisted_initial_labels",
                            "previously_used_for_m2_development": row["sha256"] in development,
                            "previously_used_by_m1": True, "label_status": "pending_human_review", "reviewer": None,
                            "geometry_meta": prepared.geometry_meta, "mask_requested": row["query_id"] in selected_masks,
                            "mask_path": None, "mask_sha256": None, "mask_status": "not_annotated", "mask_reviewer": None,
                            "mask_coordinate_space": "m1_working_image", "notes": ""})
    return {"schema_version": SCHEMA, "dataset_kind": "shared_M1_regression_pending_M2_review",
            "independent_test": False, "preprocess_config": asdict(PreprocessConfig()),
            "required_counts": {"validation_images": 5, "test_images": 15, "validation_masks": 2, "test_masks": 3},
            "queries": queries}


def validate_manifest(manifest, root=ROOT):
    if manifest.get("schema_version") != SCHEMA or not isinstance(manifest.get("queries"), list):
        raise ValueError("Unsupported M2 annotation schema")
    config = PreprocessConfig(**manifest["preprocess_config"])
    if config.mode != "identity":
        raise ValueError("Phase 4 fixes M1 identity")
    ids, hashes, targets, groups = set(), set(), set(), {}
    for row in manifest["queries"]:
        if not isinstance(row.get("query_id"), str) or not isinstance(row.get("target_id"), str):
            raise ValueError("Query and target IDs must be explicit strings")
        query = resolve_query_id(row["query_id"])
        target_id = resolve_query_id(row["target_id"])
        image = workspace_path(row["image"], root)
        if query in ids or target_id in targets or row["sha256"] in hashes:
            raise ValueError("Duplicate query/target/image hash across splits")
        ids.add(query)
        targets.add(target_id)
        hashes.add(row["sha256"])
        if not image.is_file() or file_hash(image) != row["sha256"]:
            raise ValueError("Image missing or SHA256 mismatch")
        if row["split"] not in {"validation", "test", "regression"} or row["class_name"] not in CLASS_NAMES.values():
            raise ValueError("Unknown split or class")
        if (row.get("lighting_group") not in {"dark", "normal"}
                or not isinstance(row.get("color_names"), list) or not row["color_names"]
                or any(not isinstance(color, str) or not color.strip() for color in row["color_names"])
                or any(not str(row.get(field) or "").strip() for field in ("source_url", "author", "license"))):
            raise ValueError("Missing color/lighting/source/license annotation")
        if row.get("bbox_coordinate_space") != "oriented_image_normalized":
            raise ValueError("Expected normalized oriented-image target bbox")
        bbox_to_working(row["bbox_normalized"], 1, 1)
        if row["label_status"] not in {"pending_human_review", "human_reviewed", "needs_correction"}:
            raise ValueError("Unknown review status")
        if row["label_status"] == "human_reviewed" and not str(row.get("reviewer") or "").strip():
            raise ValueError("Reviewed labels require a named human reviewer")
        group = row.get("source_group")
        if group and groups.get(group, row["split"]) != row["split"]:
            raise ValueError("Same source group occurs in different splits")
        groups[group] = row["split"]
        if row.get("mask_status") not in {"not_annotated", "pending_human_review", "human_reviewed"}:
            raise ValueError("Unknown reference mask status")
        if row.get("mask_path"):
            mask_path = workspace_path(row["mask_path"], root)
            if not mask_path.is_file() or file_hash(mask_path) != row["mask_sha256"]:
                raise ValueError("Reference mask missing or SHA256 mismatch")
        if row.get("mask_status") == "human_reviewed" and (
            not row.get("mask_path") or not str(row.get("mask_reviewer") or "").strip()
            or row["label_status"] != "human_reviewed"
        ):
            raise ValueError("Reviewed mask requires file, reviewer, and reviewed target")
    return manifest


def load_manifest(path, root=ROOT):
    return validate_manifest(read_json(path), root)


def reference_for(row, geometry, root=ROOT):
    if row.get("mask_status") != "human_reviewed" or row["label_status"] != "human_reviewed":
        return None
    with Image.open(workspace_path(row["mask_path"], root)) as image:
        if image.format != "PNG" or image.mode not in {"L", "1"}:
            raise ValueError("Reference must be a binary grayscale PNG")
        mask = np.asarray(image)
    mask = normalize_reference_mask(mask, row["mask_coordinate_space"], geometry)
    if not mask.any():
        raise ValueError("Empty clothing reference mask")
    return mask


def apply_review(original, reviewed, mask_dir, root=ROOT):
    """Only accept review fields. Validate every item before writing any mask."""
    validate_manifest(original, root)
    if reviewed.get("schema_version") != SCHEMA or len(reviewed.get("queries", [])) != len(original["queries"]):
        raise ValueError("Review does not match original manifest")
    if reviewed.get("preprocess_config") != original["preprocess_config"]:
        raise ValueError("Review cannot alter preprocessing geometry")
    mask_dir = mask_dir.resolve()
    if not mask_dir.is_relative_to(root.resolve()):
        raise ValueError("Masks must be saved within workspace")
    originals = {row["query_id"]: row for row in original["queries"]}
    result = deepcopy(original)
    outputs = {row["query_id"]: row for row in result["queries"]}
    writes, seen = [], set()
    mutable = {"bbox_normalized", "class_name", "color_names", "lighting_group", "notes", "label_status", "reviewer",
               "mask_status", "mask_reviewer", "mask_png_base64", "mask_polygons"}
    for entry in reviewed["queries"]:
        query = entry["query_id"]
        if query not in originals or query in seen:
            raise ValueError("Unknown or duplicate reviewed query")
        seen.add(query)
        prior, row = originals[query], outputs[query]
        if any(entry.get(key) != value for key, value in prior.items() if key not in mutable):
            raise ValueError("Review changed immutable provenance/image fields")
        for key in mutable - {"mask_png_base64", "mask_polygons"}:
            if key in entry:
                row[key] = entry[key]
        payload, polygons = entry.get("mask_png_base64"), entry.get("mask_polygons")
        if payload or polygons:
            prepared = prepare_image(workspace_path(row["image"], root).read_bytes(), PreprocessConfig(**original["preprocess_config"]))
            if payload:
                raw = base64.b64decode(payload, validate=True)
                with Image.open(BytesIO(raw)) as image:
                    if image.format != "PNG" or image.mode not in {"L", "1"}:
                        raise ValueError("Manual reference must be binary grayscale PNG")
                    mask = binary_mask(np.asarray(image))
            else:
                canvas = Image.new("L", (prepared.geometry_meta["width"], prepared.geometry_meta["height"]), 0)
                draw = ImageDraw.Draw(canvas)
                for polygon in polygons:
                    points = polygon["points"]
                    if (len(points) < 3 or any(len(p) != 2 or not all(isinstance(v, (float, int)) and np.isfinite(v) for v in p)
                                              or not 0 <= p[0] <= canvas.width or not 0 <= p[1] <= canvas.height for p in points)):
                        raise ValueError("Invalid polygon coordinates")
                    draw.polygon([tuple(p) for p in points], fill=0 if polygon.get("subtract", False) else 255)
                mask = np.asarray(canvas) != 0
            mask = normalize_reference_mask(mask, "m1_working_image", prepared.geometry_meta)
            if not mask.any():
                raise ValueError("Reference mask cannot be empty")
            path = mask_dir / (query + ".png")
            if path.exists():
                raise ValueError("Refusing to overwrite reference mask; choose a new mask directory")
            row.update(mask_path=path.relative_to(root).as_posix(), mask_sha256=None,
                       mask_coordinate_space="m1_working_image",
                       mask_origin=("human_reviewed_assistant_polygon" if prior.get("mask_origin", "").startswith("assistant_")
                                    else "human_polygon_review_tool") if row["mask_status"] == "human_reviewed"
                                   else prior.get("mask_origin", "unreviewed_polygon"),
                       mask_polygons=entry.get("mask_polygons", []))
            stream = BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
            encoded = stream.getvalue()
            row["mask_sha256"] = hashlib.sha256(encoded).hexdigest()
            writes.append((path, encoded))
        if row.get("mask_status") == "human_reviewed" and not row.get("mask_path"):
            raise ValueError("Mask marked reviewed without annotation")
    # Validate all non-file review constraints without writing partial output.
    preview = deepcopy(result)
    for row in preview["queries"]:
        if row.get("mask_status") == "human_reviewed" and row["label_status"] != "human_reviewed":
            raise ValueError("Reference mask requires reviewed target")
        if any(path.relative_to(root).as_posix() == row.get("mask_path") for path, _ in writes):
            row.update(mask_path=None, mask_status="pending_human_review")
    for row in result["queries"]:
        if row.get("mask_status") == "human_reviewed" and not str(row.get("mask_reviewer") or "").strip():
            raise ValueError("Reference mask reviewer required")
    validate_manifest(preview, root)
    for path, encoded in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return validate_manifest(result, root)
