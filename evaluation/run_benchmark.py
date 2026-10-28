import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from .run_eval import EVAL_DIR, RESULTS_DIR, compute_metrics, evaluate_query, write_csv
except ImportError:
    from run_eval import EVAL_DIR, RESULTS_DIR, compute_metrics, evaluate_query, write_csv


BENCHMARK_SUITES = {
    "auto_smoke": {
        "labels": EVAL_DIR / "labels.auto_smoke.json",
        "weight": 0.20,
        "description": "Fast pipeline smoke test using catalog-like images.",
        "gates": {
            "detection_success_rate": 0.90,
            "category_at_5": 0.70,
            "color_at_5": 0.70,
            "failed_queries": 0,
        },
    },
    "starter_catalog": {
        "labels": EVAL_DIR / "labels.starter_catalog.json",
        "weight": 0.30,
        "description": "Regression set built from catalog images with exact product ids.",
        "gates": {
            "detection_success_rate": 0.95,
            "category_at_5": 0.90,
            "color_at_5": 0.90,
            "recall_at_5": 0.85,
            "mrr": 0.75,
            "failed_queries": 0,
        },
    },
    "real_outfits": {
        "labels": EVAL_DIR / "labels.real.json",
        "weight": 0.50,
        "description": "Harder user-like outfit images from Wikimedia labels.",
        "gates": {
            "detection_success_rate": 0.85,
            "category_at_5": 0.70,
            "color_at_5": 0.45,
            "failed_queries": 0,
        },
    },
}

SCORE_WEIGHTS = {
    "detection_success_rate": 0.20,
    "category_at_5": 0.25,
    "color_at_5": 0.20,
    "recall_at_5": 0.15,
    "mrr": 0.10,
    "latency_score": 0.10,
}

LATENCY_TARGET_MS = 1000.0
LATENCY_MAX_MS = 8000.0


def _load_labels(path: Path) -> list[dict]:
    labels = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(labels, dict):
        labels = [labels]
    if not isinstance(labels, list):
        raise ValueError(f"Labels file must contain a JSON list: {path}")
    return labels


def _format_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def latency_score(avg_latency_ms) -> float | None:
    if avg_latency_ms is None:
        return None
    if avg_latency_ms <= LATENCY_TARGET_MS:
        return 1.0
    if avg_latency_ms >= LATENCY_MAX_MS:
        return 0.0
    return 1.0 - ((avg_latency_ms - LATENCY_TARGET_MS) / (LATENCY_MAX_MS - LATENCY_TARGET_MS))


def suite_score(metrics: dict) -> float | None:
    values = {
        "detection_success_rate": metrics.get("detection_success_rate"),
        "category_at_5": metrics.get("category_at_5"),
        "color_at_5": metrics.get("color_at_5"),
        "recall_at_5": metrics.get("recall_at_5"),
        "mrr": metrics.get("mrr"),
        "latency_score": latency_score(metrics.get("avg_latency_ms")),
    }
    weighted_sum = 0.0
    active_weight = 0.0
    for metric_name, weight in SCORE_WEIGHTS.items():
        value = values.get(metric_name)
        if value is None:
            continue
        weighted_sum += weight * float(value)
        active_weight += weight

    if active_weight <= 0:
        return None
    return weighted_sum / active_weight


def evaluate_gates(metrics: dict, gates: dict) -> tuple[bool, list[str]]:
    failures = []
    for metric_name, threshold in gates.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"{metric_name}=n/a expected >= {threshold}")
            continue

        if metric_name == "failed_queries":
            if value != threshold:
                failures.append(f"{metric_name}={value} expected {threshold}")
        elif float(value) < float(threshold):
            failures.append(f"{metric_name}={_format_metric(value)} expected >= {threshold}")

    return not failures, failures


def run_suite(name: str, config: dict, api_url: str, top_k: int, timeout: float) -> dict:
    labels_path = Path(config["labels"])
    labels = _load_labels(labels_path)
    with httpx.Client(timeout=timeout) as client:
        per_query_results = [
            evaluate_query(label, client=client, api_url=api_url, top_k=top_k)
            for label in labels
        ]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": name,
        "api_url": api_url,
        "labels": str(labels_path),
        "total_queries": len(labels),
        "metrics": compute_metrics(per_query_results),
        "per_query_results": per_query_results,
    }
    return report


def write_suite_outputs(report: dict, run_name: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_name}.json"
    csv_path = RESULTS_DIR / f"{run_name}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, report["per_query_results"])
    return json_path, csv_path


def load_existing_report(run_name: str) -> dict:
    path = RESULTS_DIR / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing benchmark suite result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_suite(name: str, config: dict, report: dict) -> dict:
    metrics = report.get("metrics", {})
    score = suite_score(metrics)
    gates_passed, gate_failures = evaluate_gates(metrics, config.get("gates", {}))
    return {
        "name": name,
        "description": config.get("description"),
        "weight": config.get("weight", 0.0),
        "labels": report.get("labels"),
        "total_queries": report.get("total_queries"),
        "metrics": metrics,
        "score": score,
        "gates_passed": gates_passed,
        "gate_failures": gate_failures,
        "run_name": report.get("run_name"),
        "api_url": report.get("api_url"),
    }


def aggregate_score(summaries: list[dict]) -> float | None:
    weighted_sum = 0.0
    active_weight = 0.0
    for summary in summaries:
        score = summary.get("score")
        if score is None:
            continue
        weight = float(summary.get("weight") or 0.0)
        weighted_sum += weight * score
        active_weight += weight

    if active_weight <= 0:
        return None
    return weighted_sum / active_weight


def write_benchmark_csv(path: Path, summaries: list[dict]):
    fieldnames = [
        "suite",
        "weight",
        "score",
        "gates_passed",
        "queries",
        "detection_success_rate",
        "avg_detected_items",
        "category_at_5",
        "color_at_5",
        "recall_at_5",
        "mrr",
        "avg_latency_ms",
        "failed_queries",
        "gate_failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            writer.writerow(
                {
                    "suite": summary["name"],
                    "weight": summary["weight"],
                    "score": summary["score"],
                    "gates_passed": summary["gates_passed"],
                    "queries": summary["total_queries"],
                    "detection_success_rate": metrics.get("detection_success_rate"),
                    "avg_detected_items": metrics.get("avg_detected_items"),
                    "category_at_5": metrics.get("category_at_5"),
                    "color_at_5": metrics.get("color_at_5"),
                    "recall_at_5": metrics.get("recall_at_5"),
                    "mrr": metrics.get("mrr"),
                    "avg_latency_ms": metrics.get("avg_latency_ms"),
                    "failed_queries": metrics.get("failed_queries"),
                    "gate_failures": "; ".join(summary["gate_failures"]),
                }
            )


def write_markdown_report(path: Path, benchmark: dict):
    lines = [
        "# Fashion Visual Search Benchmark",
        "",
        f"- Timestamp: `{benchmark['timestamp']}`",
        f"- API URL: `{benchmark['api_url']}`",
        f"- Top K: `{benchmark['top_k']}`",
        f"- Overall score: `{_format_metric(benchmark['overall_score'])}`",
        f"- Overall gates: `{'PASS' if benchmark['overall_gates_passed'] else 'FAIL'}`",
        "",
        "## Suite Summary",
        "",
        "| Suite | Weight | Score | Gates | Queries | Detection | Category@5 | Color@5 | Recall@5 | MRR | Latency ms |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in benchmark["suites"]:
        metrics = summary["metrics"]
        lines.append(
            "| {suite} | {weight:.2f} | {score} | {gates} | {queries} | {detection} | {category} | "
            "{color} | {recall} | {mrr} | {latency} |".format(
                suite=summary["name"],
                weight=float(summary["weight"]),
                score=_format_metric(summary["score"]),
                gates="PASS" if summary["gates_passed"] else "FAIL",
                queries=summary["total_queries"],
                detection=_format_metric(metrics.get("detection_success_rate")),
                category=_format_metric(metrics.get("category_at_5")),
                color=_format_metric(metrics.get("color_at_5")),
                recall=_format_metric(metrics.get("recall_at_5")),
                mrr=_format_metric(metrics.get("mrr")),
                latency=_format_metric(metrics.get("avg_latency_ms")),
            )
        )

    lines.extend(["", "## Gate Failures", ""])
    any_failures = False
    for summary in benchmark["suites"]:
        if summary["gate_failures"]:
            any_failures = True
            lines.append(f"### {summary['name']}")
            for failure in summary["gate_failures"]:
                lines.append(f"- {failure}")
            lines.append("")
    if not any_failures:
        lines.append("All benchmark gates passed.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `auto_smoke` catches basic API and pipeline breakage quickly.",
            "- `starter_catalog` checks catalog/product-id retrieval regressions on easier clean images.",
            "- `real_outfits` is the strongest user-like suite and should be weighted highest.",
            "- Overall score is a weighted aggregate; gate failures should be reviewed even if the score is high.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the full Fashion Visual Search benchmark suite.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--run-prefix", default=f"benchmark_{int(time.time())}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=sorted(BENCHMARK_SUITES),
        default=["auto_smoke", "starter_catalog", "real_outfits"],
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Build aggregate report from existing suite JSON files named {run_prefix}_{suite}.json.",
    )
    args = parser.parse_args()

    summaries = []
    suite_outputs = {}
    for suite_name in args.suites:
        config = BENCHMARK_SUITES[suite_name]
        run_name = f"{args.run_prefix}_{suite_name}"
        if args.use_existing:
            report = load_existing_report(run_name)
            json_path = RESULTS_DIR / f"{run_name}.json"
            csv_path = RESULTS_DIR / f"{run_name}.csv"
        else:
            report = run_suite(
                suite_name,
                config,
                api_url=args.api_url,
                top_k=args.top_k,
                timeout=args.timeout,
            )
            report["run_name"] = run_name
            json_path, csv_path = write_suite_outputs(report, run_name)

        summaries.append(summarize_suite(suite_name, config, report))
        suite_outputs[suite_name] = {"json": str(json_path), "csv": str(csv_path)}

    overall_score = aggregate_score(summaries)
    benchmark = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "top_k": args.top_k,
        "run_prefix": args.run_prefix,
        "overall_score": overall_score,
        "overall_gates_passed": all(summary["gates_passed"] for summary in summaries),
        "suites": summaries,
        "suite_outputs": suite_outputs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_json = RESULTS_DIR / f"{args.run_prefix}_summary.json"
    aggregate_csv = RESULTS_DIR / f"{args.run_prefix}_summary.csv"
    aggregate_md = RESULTS_DIR / f"{args.run_prefix}_summary.md"
    latest_json = RESULTS_DIR / "latest_benchmark.json"
    latest_csv = RESULTS_DIR / "latest_benchmark.csv"
    latest_md = RESULTS_DIR / "latest_benchmark.md"

    aggregate_json.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    write_benchmark_csv(aggregate_csv, summaries)
    write_markdown_report(aggregate_md, benchmark)
    shutil.copyfile(aggregate_json, latest_json)
    shutil.copyfile(aggregate_csv, latest_csv)
    shutil.copyfile(aggregate_md, latest_md)

    print("Benchmark summary")
    print("=================")
    print(f"Run prefix: {args.run_prefix}")
    print(f"Overall score: {_format_metric(overall_score)}")
    print(f"Overall gates: {'PASS' if benchmark['overall_gates_passed'] else 'FAIL'}")
    for summary in summaries:
        print(
            f"- {summary['name']}: score={_format_metric(summary['score'])}, "
            f"gates={'PASS' if summary['gates_passed'] else 'FAIL'}"
        )
    print(f"\nWrote: {aggregate_json}")
    print(f"Wrote: {aggregate_csv}")
    print(f"Wrote: {aggregate_md}")


if __name__ == "__main__":
    main()
