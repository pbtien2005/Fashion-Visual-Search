"""Build deterministic catalog queries and score product-ID recovery."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .manifest import EVAL_DIR, file_hash, read_csv, write_csv
from .metrics import reciprocal_rank_at_k

SEED = 20261116
CONDITIONS = ("original", "dark", "downsample", "jpeg_q10", "gauss_noise", "rotate15", "offcrop")
FIELDS = ("query_id", "image", "product_id", "condition", "source_image")
SOURCE_MANIFESTS = ("labels.starter_catalog.json", "labels.auto_smoke.json")


def transform(image, condition, rng):
    width, height = image.size
    if condition == "dark":
        # Underexposure scales every pixel, white backgrounds included, so mean luma
        # falls below the M1 gamma threshold. A gamma curve would leave white at 1.
        return Image.fromarray(np.rint(np.asarray(image) * 0.25).astype(np.uint8))
    if condition == "downsample":
        return image.resize((max(1, width // 4), max(1, height // 4)), Image.Resampling.BOX)
    if condition == "gauss_noise":
        pixels = np.asarray(image) + rng.normal(0, 20, (height, width, 3))
        return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    if condition == "rotate15":
        return image.rotate(15, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if condition == "offcrop":
        left, top = round(width * .025), round(height * .025)
        return image.crop((left, top, left + max(1, round(width * .75)), top + max(1, round(height * .75))))
    return image


def source_images(root):
    sources = {}
    for filename in SOURCE_MANIFESTS:
        for row in json.loads((root / filename).read_text(encoding="utf-8-sig")):
            ids = {str(pid) for item in row["expected_items"] for pid in item["acceptable_product_ids"]}
            if len(ids) != 1:
                raise ValueError(f"Expected one product ID for {row['image']}")
            pid = ids.pop()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
                raise ValueError(f"Invalid product ID: {pid!r}")
            sources.setdefault(pid, root / row["image"])
    return sources


def build(root=EVAL_DIR, force=False):
    root = Path(root)
    folder = root / "queries" / "perturbed"
    csv_path = root / "bench_perturbed.csv"
    if not force and (folder.exists() or csv_path.exists()):
        raise FileExistsError(f"Already built: {folder}; use --force to regenerate")
    sources = source_images(root)
    folder.mkdir(parents=True, exist_ok=force)
    rng = np.random.default_rng(SEED)
    rows = []
    for product_id, path in sorted(sources.items()):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            original_is_jpeg = original.format == "JPEG"
        for condition in CONDITIONS:
            destination = folder / condition / f"{product_id}.jpg"
            destination.parent.mkdir(exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            if condition == "original" and original_is_jpeg:
                temporary.write_bytes(path.read_bytes())
            else:
                transform(image, condition, rng).save(
                    temporary, format="JPEG", quality=10 if condition == "jpeg_q10" else 95,
                )
            temporary.replace(destination)
            rows.append({
                "query_id": f"{product_id}_{condition}", "image": destination.relative_to(root).as_posix(),
                "product_id": product_id, "condition": condition, "source_image": path.relative_to(root).as_posix(),
            })
    write_csv(csv_path, FIELDS, rows)
    return rows


def load_perturbed(path):
    rows = read_csv(path, FIELDS)
    ids, conditions_by_product = set(), {}
    for row in rows:
        image = Path(row["image"])
        if image.is_absolute() or image.drive or ".." in image.parts:
            raise ValueError(f"Invalid relative image path: {image}")
        key = row["condition"]
        seen = conditions_by_product.setdefault(row["product_id"], set())
        if key not in CONDITIONS or key in seen or row["query_id"] in ids:
            raise ValueError(f"Invalid or duplicate perturbation row: {row['query_id']}")
        seen.add(key)
        ids.add(row["query_id"])
        row["path"] = Path(path).parent / image
        row["sha256"] = file_hash(row["path"])
    if not conditions_by_product or any(group != set(CONDITIONS) for group in conditions_by_product.values()):
        raise ValueError("Every product must have all seven conditions")
    return rows


def robustness_details(snapshot):
    rows = []
    for query in snapshot["queries"]:
        for name, result in query["results"].items():
            detections = result["detections"]
            rr = max((
                reciprocal_rank_at_k([str(product["id"]) for product in detection["ranked"]],
                                     {query["product_id"]}, 5)
                for detection in detections
            ), default=0.0)
            rows.append({
                "query_id": query["query_id"], "product_id": query["product_id"],
                "condition": query["condition"], "config": name, "rr_at_5": rr,
                "no_detection": int(not detections) if result.get("detection_measured", True) else None,
                "gamma_applied": result["quality"]["gamma_applied"],
            })
    return rows


def robustness_stats(snapshot, details=None):
    details = robustness_details(snapshot) if details is None else details
    rows = []
    for condition in CONDITIONS:
        for name in snapshot["configs"]:
            subset = [row for row in details if row["condition"] == condition and row["config"] == name]
            ranks = [row["rr_at_5"] for row in subset]
            no_detection = [row["no_detection"] for row in subset if row["no_detection"] is not None]
            rows.append({
                "condition": condition, "config": name, "n": len(subset),
                "hits_at_1": ranks.count(1.0), "hits_at_5": sum(rank > 0 for rank in ranks),
                "rr_sum": sum(ranks), "no_detection": sum(no_detection) if no_detection else None,
                "gamma_applied": sum(bool(row["gamma_applied"]) for row in subset),
            })
    return rows


def write_robustness_report(folder, snapshot):
    details = robustness_details(snapshot)
    rows = robustness_stats(snapshot, details)
    reference = {row["config"]: row["hits_at_5"] / row["n"]
                 for row in rows if row["condition"] == "original"}

    def cell(numerator, n):
        if numerator is None:
            return "n/a"
        return f"{numerator/n:.3f} ({numerator:.3f}/{n})" if isinstance(numerator, float) else (
            f"{numerator/n:.3f} ({numerator}/{n})"
        )

    lines = [
        "# Độ bền trên ảnh catalog", "",
        "Mỗi ảnh truy vấn có đúng một product ID tham chiếu. Lấy hạng tốt nhất trong mọi detection trước ngưỡng.",
        "A0 dùng toàn ảnh. Mức thay đổi HitRate@5 được tính so với original của cùng cấu hình.",
        "Cột gamma đếm số ảnh mà tiền xử lý M1 thực sự bật gamma. B1 chỉ khác B0 trên các ảnh đó.", "",
        "| Điều kiện | Cấu hình | HitRate@1 | HitRate@5 | MRR@5 | Không có detection | Gamma bật | Δ HitRate@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = row["n"]
        delta = row["hits_at_5"] / n - reference[row["config"]]
        values = [cell(row[key], n) for key in ("hits_at_1", "hits_at_5", "rr_sum", "no_detection")]
        lines.append("| " + " | ".join([row["condition"], row["config"], *values,
                                        f"{row['gamma_applied']}/{n}", f"{delta:+.3f}"]) + " |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(folder / "summary.csv", tuple(rows[0]), rows)
    write_csv(folder / "per_query.csv", tuple(details[0]), details)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        print(f"Built {len(build(force=args.force))} catalog queries")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

