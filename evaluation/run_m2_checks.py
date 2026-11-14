"""Run the merged offline suite and save native exit status plus a compact report."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/m2/phase3")
    parser.add_argument("--coverage-source", default="object_detection,AI.features,AI.hsv_descriptor")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "coverage", "run", "--append",
               "--source=" + args.coverage_source, "-m", "unittest", "discover", "-s", "tests", "-v"]
    with (output / "tests.txt").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    log = (output / "tests.txt").read_text(encoding="utf-8")
    matched = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", log)
    skip = re.search(r"OK \(skipped=(\d+)\)", log)
    count = int(matched[1]) if matched else None
    skipped = int(skip[1]) if skip else 0
    report = {"status": "passed" if completed.returncode == 0 and count else "failed",
              "exit_code": completed.returncode, "tests_run": count, "skipped": skipped,
              "passed": count - skipped if count and completed.returncode == 0 else None,
              "seconds": float(matched[2]) if matched else None,
              "skip_reason": "pretrained encoder opt-in disabled; no model download or live DB in this suite",
              "command": command[1:]}
    (output / "test_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
