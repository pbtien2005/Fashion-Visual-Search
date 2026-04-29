import argparse
import json
from pathlib import Path


def load_report(path: str) -> dict:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Missing eval report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Compare two evaluation JSON reports.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    metric_names = sorted(set(baseline.get("metrics", {})) | set(candidate.get("metrics", {})))

    print("Metric comparison")
    print("=================")
    print(f"Baseline:  {baseline.get('run_name') or args.baseline}")
    print(f"Candidate: {candidate.get('run_name') or args.candidate}")
    print("")
    print(f"{'metric':28} {'baseline':>12} {'candidate':>12} {'delta':>12}")
    for name in metric_names:
        base_value = baseline.get("metrics", {}).get(name)
        candidate_value = candidate.get("metrics", {}).get(name)
        delta = None
        if isinstance(base_value, (int, float)) and isinstance(candidate_value, (int, float)):
            delta = candidate_value - base_value
        print(
            f"{name:28} {format_value(base_value):>12} "
            f"{format_value(candidate_value):>12} {format_value(delta):>12}"
        )


if __name__ == "__main__":
    main()
