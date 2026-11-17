"""Score saved inference with human judgments; freeze the validation choice."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from object_detection.image_preprocessing import prepare_image

from .grade import load_judgments
from .manifest import resolve_path, write_csv
from .metrics import TIMING_FIELDS, paired_outcomes, summarize
from .pipeline import BASELINE, GAMMA, PENDING, SWEEP, WEIGHT_KEYS
from .run import read_json, write_json

MAIN = (
    ("hit_at_5", "HitRate@5"), ("mrr_at_5", "MRR@5"), ("precision_at_5", "Precision@5"),
    ("joint_at_5", "Joint loại+màu@5"), ("candidate_hit_at_100", "Candidate hit@100"),
)
DECISION = (("hit_after_at_5", "HitRate@5"), ("mrr_after_at_5", "MRR@5"),
            ("coverage", "Coverage"), ("false_acceptance", "False acceptance"))
ERRORS = ("no_detection", "wrong_class", "candidate_miss", "rank_miss", "rejected", "hit")


def ratio(pair, count_only_below=None):
    numerator, denominator = pair
    if count_only_below is not None and denominator < count_only_below:
        return f"{numerator} ca / {denominator}"
    if not denominator:
        return "n/a"
    count = f"{numerator:.3f}" if isinstance(numerator, float) else str(numerator)
    return f"{numerator / denominator:.3f} ({count}/{denominator})"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def select_c3(reports):
    def mrr(name):
        total, n = reports[name]["mrr_at_5"]
        return total / n if n else 0

    # max preserves the first item on ties. B0 has priority, then w1.
    return max(SWEEP, key=mrr)


def save_lock(run_dir, snapshot, reports, incomplete):
    selected = select_c3(reports) if all(name in reports for name in SWEEP) else None
    reason = ("MRR@5 trước ngưỡng cao nhất trên validation. Thứ tự ưu tiên khi hòa là B0_current, C3_w1, C3_w2."
              if selected else "Lượt HTTP độc lập, không quét trọng số C3.")
    if incomplete:
        reason += " Khóa tạm từ dữ liệu chấm chưa hoàn tất (--allow-incomplete)."
    lock = {field: snapshot[field] for field in ("model_sha256", "catalog_sha256", "conf", "crop_config", "configs")}
    lock.update({
        "validation_manifest": snapshot["manifest"],
        "validation_manifest_sha256": snapshot["manifest_sha256"],
        "validation_images": sorted(query["sha256"] for query in snapshot["queries"]),
        "C3_source": selected,
        "C3_selected": ({**snapshot["configs"][selected], "name": "C3_selected"} if selected else None),
        "validation_mrr_at_5": {
            name: {"numerator": reports[name]["mrr_at_5"][0], "denominator": reports[name]["mrr_at_5"][1]}
            for name in SWEEP if name in reports
        },
        "reason": reason,
    })
    write_json(run_dir / "locked_config.json", lock)
    return selected


def config_rows(snapshot, qrels, status):
    reports = {}
    for name in snapshot["configs"]:
        rows = [
            {**query["results"][name], **{key: query[key] for key in ("query_id", "lighting_group", "color")}}
            for query in snapshot["queries"]
        ]
        reports[name] = summarize(rows, qrels, status)
    baseline = reports.get(BASELINE)
    if baseline:
        reference = {row["query_id"]: row["rr_at_5"] for row in baseline["per_query"] if row["rr_at_5"] is not None}
        for report in reports.values():
            other = {row["query_id"]: row["rr_at_5"] for row in report["per_query"] if row["rr_at_5"] is not None}
            report["paired"] = paired_outcomes(reference, other)
    return reports


def write_examples(folder, snapshot, baseline):
    folder.mkdir(exist_ok=True)
    metadata = []
    queries = {query["query_id"]: query for query in snapshot["queries"]}
    counts = Counter()
    for row in baseline["per_query"]:
        reason = row["failure"]
        if reason is None or counts[reason] >= 3:
            continue
        query = queries[row["query_id"]]
        result = query["results"][BASELINE]
        if result["target"]["bbox"] is None:
            continue
        path = resolve_path(snapshot["manifest"]).parent / query["image"]
        pixels = path.read_bytes()
        if hashlib.sha256(pixels).hexdigest() != query["sha256"]:
            raise ValueError(f"Query image changed after inference: {query['query_id']}")
        image = Image.fromarray(prepare_image(pixels).source_rgb)
        draw = ImageDraw.Draw(image)
        index = result["target"]["index"]
        predictions = result["detections"] if index is None else [result["detections"][index]]
        for prediction in predictions:
            draw.rectangle(prediction["bbox"], outline="#ff8446", width=5)
        draw.rectangle(result["target"]["bbox"], outline="#16ff70", width=3)
        filename = f"{query['query_id']}-{reason}.jpg"
        image.save(folder / filename, quality=93)
        counts[reason] += 1
        metadata.append({"image": filename, "failure": reason,
                         **{key: query[key] for key in ("query_id", "source", "author", "license", "license_url")}})
    paths = {row["image"] for row in metadata}
    for path in folder.glob("*.jpg"):
        if path.name not in paths:
            path.unlink()
    write_csv(folder / "examples.csv", ("image", "failure", "query_id", "source", "author", "license", "license_url"),
              metadata)


def report_markdown(snapshot, reports, warnings, selected, frozen=False):
    lines = [*warnings, "", f"# Benchmark {snapshot['stage']}", ""] if warnings else [
        f"# Benchmark {snapshot['stage']}", "",
    ]
    queries = snapshot["queries"]
    class_counts = Counter(row["class_name"] for row in queries)
    light_counts = Counter(row["lighting_group"] for row in queries)
    colors = Counter(row["color"] for row in queries)
    judged_counts = Counter(row["relevance_status"] for row in next(iter(reports.values()))["per_query"])
    lines.extend([
        "## Tập dữ liệu", "",
        f"- {len(queries)} ảnh. "
        + ", ".join(f"{key} {judged_counts[key]}" for key in ("has_match", "no_match", "unconfirmed")) + ".",
        "- Lớp · " + ", ".join(f"{key} {count}" for key, count in sorted(class_counts.items())) + ".",
        "- Ánh sáng · " + ", ".join(f"{key} {count}" for key, count in sorted(light_counts.items())) + ".",
        "- Màu · " + ", ".join(f"{key} {count}" for key, count in sorted(colors.items())) + ".",
        "",
        "## Trước ngưỡng", "",
        "C1p_color_only là đối chứng màu trội và từ khóa. C1 của nhóm dùng histogram HSV.",
        "HitRate chỉ đo có sản phẩm chấp nhận được trong top-k. Recall cần biết toàn bộ sản phẩm phù hợp. "
        "Candidate hit@100 là cận dưới trong phạm vi đã chấm.", "",
    ])
    main_rows = []
    for name, report in reports.items():
        paired = report.get("paired")
        outcome = "/".join(str(paired[key]) for key in ("win", "loss", "tie")) if paired else "n/a"
        main_rows.append([name, *(ratio(report[key]) for key, _ in MAIN), outcome])
    main_rows.extend([f"{name} ({description})", *(["chưa đo"] * 6)]
                     for name, description in PENDING.items() if name not in reports)
    lines.extend(table(["Cấu hình", *[label for _, label in MAIN], "Thắng/thua/hòa vs B0"], main_rows))
    lines.extend(["## Sau quyết định", ""])
    lines.extend(table(["Cấu hình", *[label for _, label in DECISION]], [
        [name, *(ratio(report[key], 5 if key == "false_acceptance" else None) for key, _ in DECISION)]
        for name, report in reports.items()
    ]))
    lines.extend(["## Phát hiện mục tiêu", ""])
    detect_names = [name for name in (BASELINE, GAMMA) if name in reports]
    detect_names.extend(name for name in reports if name not in detect_names)
    lines.extend(table(["Cấu hình", "Recall", "Dark", "Normal"], [
        [name, ratio(reports[name]["target_recall"]), *(ratio(reports[name]["target_by_lighting"][group])
                                                       for group in ("dark", "normal"))] for name in detect_names
    ]))
    lines.extend(["## Màu truy vấn", ""])
    lines.extend(table(["Cấu hình", "Độ chính xác"], [[name, ratio(report["color_accuracy"])]
                                                     for name, report in reports.items()]))
    confusion = [
        [name, actual, predicted, count] for name, report in reports.items()
        for actual, counts in sorted(report["confusion_matrix"].items())
        for predicted, count in sorted(counts.items())
    ]
    lines.extend(table(["Cấu hình", "Màu nhãn", "Màu dự đoán", "Số ca"], confusion))
    if snapshot["stage"] == "validation":
        lines.extend(["## Trọng số đã thử", ""])
        lines.extend(table(["Cấu hình", "Ảnh", "Loại", "Màu", "Text/style", "MRR@5"], [
            [name, *(f"{float(snapshot['configs'][name]['env'][key]):.3f}" for key in WEIGHT_KEYS),
             ratio(reports[name]["mrr_at_5"])]
            for name in SWEEP if name in reports and "env" in snapshot["configs"][name]
        ]))
        lines.extend([f"C3_selected · {selected or 'chưa khóa'}", ""])
        if frozen:
            lines.extend(["Cấu hình đã khóa được giữ theo lượt test.", ""])
    lines.extend(["## Latency (ms)", ""])
    timing_rows = []
    for name, report in reports.items():
        for stage in TIMING_FIELDS:
            time = report["latency"][stage]
            values = (time["first"], time["warm"]["mean"], time["warm"]["p95"])
            timing_rows.append([name, stage, *(f"{value:.3f}" if value is not None else "n/a" for value in values),
                                time["warm"]["n"]])
    lines.extend(table(["Cấu hình", "Công đoạn", "Ảnh đầu", "Mean sau ảnh đầu", "p95 sau ảnh đầu", "n"],
                       timing_rows))
    lines.extend(["## Phân loại lỗi", ""])
    lines.extend(table(["Cấu hình", *ERRORS], [
        [name, *("n/a" if snapshot["configs"][name]["kind"] == "http" and key in ("candidate_miss", "rank_miss")
                 else report["failures"].get(key, 0) for key in ERRORS)] for name, report in reports.items()
    ]))
    return "\n".join(lines)


def score(run_dir, stage, allow_incomplete=False):
    folder = run_dir / stage
    snapshot = read_json(folder / "results.json")
    if snapshot["stage"] != stage or snapshot.get("smoke"):
        raise ValueError("Use a completed, non-smoke run for scoring")
    qrels, status = load_judgments()
    reports = config_rows(snapshot, qrels, status)
    unconfirmed = sorted({qid for report in reports.values() for qid in report["unconfirmed"]})
    unjudged = {name: report["unjudged_at_5"] for name, report in reports.items() if report["unjudged_at_5"]}
    warnings = []
    if unconfirmed:
        warnings.extend(["CHƯA HOÀN TẤT CHẤM", "", "Truy vấn unconfirmed · " + ", ".join(unconfirmed)])
    if unjudged:
        warnings.append("Cảnh báo unjudged_at_5 · " + ", ".join(f"{name}={n}" for name, n in unjudged.items()))
    incomplete = bool(warnings)
    failed = Counter(
        name for query in snapshot["queries"] for name, result in query["results"].items()
        if result.get("error") or any(detection.get("error") for detection in result["detections"])
    )
    if failed:
        warnings.append("Lỗi suy luận, tính là trượt · " + ", ".join(f"{name}={n}" for name, n in failed.items()))
    frozen = stage == "validation" and (run_dir / "test").exists()
    can_lock = stage == "validation" and not frozen and (allow_incomplete or not incomplete)
    selected = select_c3(reports) if can_lock and all(name in reports for name in SWEEP) else None
    if frozen:
        selected = read_json(run_dir / "locked_config.json").get("C3_source")
    summary_path = folder / "summary.md"
    summary_path.write_text(report_markdown(snapshot, reports, warnings, selected, frozen), encoding="utf-8")
    summary_rows = []
    for name, report in reports.items():
        row = {"config": name}
        for key in (*[key for key, _ in MAIN + DECISION], "target_recall", "color_accuracy"):
            numerator, denominator = report[key]
            row.update({key: f"{numerator / denominator:.3f}" if denominator and (
                key != "false_acceptance" or denominator >= 5) else "",
                        f"{key}_numerator": numerator, f"{key}_denominator": denominator})
        row["unjudged_at_5"] = report["unjudged_at_5"]
        row.update({key: report.get("paired", {}).get(key) for key in ("win", "loss", "tie")})
        summary_rows.append(row)
    write_csv(folder / "summary.csv", tuple(summary_rows[0]), summary_rows)
    per_query = [{"config": name, **row, "ranked_ids": json.dumps(row["ranked_ids"])}
                 for name, report in reports.items() for row in report["per_query"]]
    write_csv(folder / "per_query.csv", tuple(per_query[0]), per_query)
    if BASELINE in reports:
        write_examples(folder / "examples", snapshot, reports[BASELINE])
    if can_lock:
        save_lock(run_dir, snapshot, reports, incomplete)
    elif stage == "validation" and not frozen:
        print("Validation lock withheld until judgments are complete.")
    return summary_path


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("validation", "test"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        print(score(args.run_dir, args.stage, args.allow_incomplete))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

