"""Compare identity and gamma through the production detector on labeled outfits."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from object_detection.detector import CLASS_NAMES, ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig, prepare_image


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids, hashes = set(), set()
    for row in rows:
        row["path"] = (path.parent / row["image"]).resolve()
        row["sha256"] = file_hash(row["path"])
        if row["query_id"] in ids or row["sha256"] in hashes:
            raise ValueError(f"Duplicate query ID or image: {row['query_id']}")
        ids.add(row["query_id"])
        hashes.add(row["sha256"])
        if row["lighting_group"] not in {"dark", "normal"} or row["class_name"] not in CLASS_NAMES.values():
            raise ValueError(f"Invalid lighting group or class: {row['query_id']}")
        if not all(row[field].strip() for field in ("query_id", "color", "source", "author", "license")):
            raise ValueError(f"Missing label or source metadata: {row['query_id']}")
        x1, y1, x2, y2 = row["bbox"] = [float(value) for value in row["bbox"].split()]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Expected normalized xyxy bbox: {row['query_id']}")
    if {row["lighting_group"] for row in rows} != {"dark", "normal"}:
        raise ValueError("Each split must contain dark and normal outfits")
    return rows


def box_iou(a, b):
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = overlap_width * overlap_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap) if overlap else 0.0


def measure(detector, labels, config):
    import cv2

    rows = []
    for label in labels:
        prepared = prepare_image(label["path"].read_bytes(), config)
        crops = detector._process(
            cv2.cvtColor(prepared.source_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(prepared.detection_rgb, cv2.COLOR_RGB2BGR),
        )
        width, height = prepared.geometry_meta["width"], prepared.geometry_meta["height"]
        target = [value * size for value, size in zip(label["bbox"], (width, height) * 2, strict=True)]
        predictions = [
            {"class_name": crop["class_name"], "bbox": crop["bbox"], "confidence": crop["confidence"]}
            for crop in crops
        ]
        best_iou = max(
            (box_iou(target, pred["bbox"]) for pred in predictions if pred["class_name"] == label["class_name"]),
            default=0.0,
        )
        rows.append({
            "query_id": label["query_id"],
            "lighting_group": label["lighting_group"],
            "class_name": label["class_name"],
            "target_bbox": target,
            "detection_count": len(predictions),
            "predictions": predictions,
            "best_class_iou": best_iou,
            "matched": best_iou >= 0.5,
            "gamma_applied": prepared.quality_meta["gamma_applied"],
            "preprocess_ms": prepared.quality_meta["preprocess_ms"],
            "first_call": not rows,
        })
    return rows


def summarize(rows):
    report = {}
    for group in ("dark", "normal"):
        subset = [row for row in rows if row["lighting_group"] == group]
        warm = [row["preprocess_ms"] for row in subset if not row["first_call"]]
        report[group] = {
            "images": len(subset),
            "detections": sum(row["detection_count"] for row in subset),
            "matches": sum(row["matched"] for row in subset),
            "gamma_rate": sum(row["gamma_applied"] for row in subset) / len(subset),
            "first_preprocess_ms": next((row["preprocess_ms"] for row in subset if row["first_call"]), None),
            "warm_mean_ms": float(np.mean(warm)) if warm else None,
            "warm_p95_ms": float(np.percentile(warm, 95)) if warm else None,
        }
    return report


def changed_cases(baseline, candidate):
    return [
        {
            "query_id": before["query_id"],
            "lighting_group": before["lighting_group"],
            "outcome": "gain" if after["matched"] else "loss",
        }
        for before, after in zip(baseline, candidate, strict=True)
        if before["matched"] != after["matched"]
    ]


def save_examples(folder, labels, baseline, rows, config, name, cases):
    labels = {row["query_id"]: row for row in labels}
    paired = {before["query_id"]: (before, after) for before, after in zip(baseline, rows, strict=True)}
    for index, case in enumerate(cases):
        label = labels[case["query_id"]]
        prepared = prepare_image(label["path"].read_bytes(), config)
        left, right = Image.fromarray(prepared.source_rgb), Image.fromarray(prepared.detection_rgb)
        for panel, result, title in zip((left, right), paired[case["query_id"]], ("B0", "B1"), strict=True):
            draw = ImageDraw.Draw(panel)
            draw.rectangle(result["target_bbox"], outline="#16ff70", width=3)
            for pred in result["predictions"]:
                draw.rectangle(pred["bbox"], outline="#ff8446", width=2)
            draw.text((7, 7), title, fill="white", stroke_width=2, stroke_fill="black")
        combined = Image.new("RGB", (left.width + right.width, left.height))
        combined.paste(left)
        combined.paste(right, (left.width, 0))
        destination = folder / f"{name}-{case['outcome']}-{index + 1:02}.jpg"
        combined.save(destination, quality=93)
        case.update(
            image=destination.name,
            source=label["source"],
            author=label["author"],
            license=label["license"],
            license_url=label.get("license_url"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model_hash = file_hash(args.model)
    label_hashes = {row["sha256"] for row in labels}
    lock_path = args.run_dir / "locked_config.json"
    baseline_config = PreprocessConfig()
    if args.stage == "validation":
        candidates = [
            replace(baseline_config, mode="gamma", brightness_threshold=threshold, gamma=gamma)
            for threshold in (0.30, 0.35, 0.40)
            for gamma in (0.4, 0.5, 0.6, 0.7)
        ]
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if model_hash != lock["model_sha256"] or args.conf != lock["conf"]:
            parser.error("Test must use the model and confidence locked on validation")
        if label_hashes.intersection(lock["validation_images"]):
            parser.error("Validation and test must use disjoint images")
        baseline_config = PreprocessConfig(**lock["baseline"])
        candidates = [PreprocessConfig(**lock["candidate"])]

    detector = ClothingDetector(model_path=str(args.model.resolve()), conf=args.conf, preprocess_config=baseline_config)
    crop_config = {"expand_ratio": detector.expand_ratio, "wide_expand_ratio": detector.wide_expand_ratio}
    if args.stage == "test" and lock["crop_config"] != crop_config:
        parser.error("Test must use the crop settings locked on validation")

    # Reserve the stage before inference so reruns never replace a test result.
    folder = args.run_dir / args.stage
    folder.mkdir(parents=True, exist_ok=False)
    detector._ensure_model()
    baseline = measure(detector, labels, baseline_config)
    baseline_summary = summarize(baseline)
    runs = [{"name": "B0", "config": asdict(baseline_config), "summary": baseline_summary, "rows": baseline}]
    ranked = []
    for index, config in enumerate(candidates, 1):
        name = f"B1-{index:02}" if args.stage == "validation" else "B1"
        rows = measure(detector, labels, config)
        summary = summarize(rows)
        cases = changed_cases(baseline, rows)
        examples = [case.copy() for case in cases]
        if args.stage == "test" and not any(case["outcome"] == "gain" for case in cases):
            retained = next((
                after for before, after in zip(baseline, rows, strict=True)
                if before["matched"] and after["matched"] and after["gamma_applied"]
            ), None)
            if retained:
                examples.append({"query_id": retained["query_id"], "outcome": "retained"})
        save_examples(folder, labels, baseline, rows, config, name, examples)
        runs.append({
            "name": name, "config": asdict(config), "summary": summary,
            "changes": cases, "examples": examples, "rows": rows,
        })
        normal_losses = sum(case["outcome"] == "loss" and case["lighting_group"] == "normal" for case in cases)
        dark_gain = summary["dark"]["matches"] - baseline_summary["dark"]["matches"]
        ranked.append(((normal_losses, -dark_gain, sum(row["gamma_applied"] for row in rows)), config, name))

    report = {
        "model_sha256": model_hash,
        "conf": args.conf,
        "crop_config": crop_config,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "labels": [{key: value for key, value in row.items() if key != "path"} for row in labels],
        "runs": runs,
    }
    if args.stage == "validation":
        score, winner, winner_name = min(ranked, key=lambda item: item[0])
        report["selected_candidate"] = winner_name
        report["meets_validation_gate"] = score[0] == 0 and score[1] < 0
        lock_path.write_text(json.dumps({
            "model_sha256": model_hash,
            "conf": args.conf,
            "crop_config": crop_config,
            "validation_images": sorted(label_hashes),
            "baseline": asdict(baseline_config),
            "candidate": asdict(winner),
        }, indent=2) + "\n", encoding="utf-8")
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "run", "lighting_group", "images", "detections", "matches",
            "gamma_rate", "first_preprocess_ms", "warm_mean_ms", "warm_p95_ms",
        ])
        writer.writeheader()
        for run in runs:
            for group, summary in run["summary"].items():
                writer.writerow({"run": run["name"], "lighting_group": group, **summary})


if __name__ == "__main__":
    main()
