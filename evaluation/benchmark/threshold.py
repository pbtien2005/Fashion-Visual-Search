"""Calibrate the acceptance thresholds offline from stored inference.

Acceptance rule (same as api.search_service.apply_match_threshold on the top-1 product):
    accepted = top1.final_score >= T and (M is None or top1.image_similarity is None or >= M)

Selection rule, fixed before looking at the sweep: maximise HitRate@5 after the decision on
validation; on ties pick the highest T, then the highest M. The test stage only re-applies the
locked pair; it never searches.
"""
import argparse
import csv
import json
from pathlib import Path

from .grade import load_judgments
from .run import read_json

T_GRID = tuple(round(0.30 + 0.05 * step, 2) for step in range(9))   # 0.30 .. 0.70
M_GRID = (None, 0.30, 0.40, 0.50, 0.60)
CONFIGS = ("C2_embedding_only", "B0_current", "C3_w1", "C3_w2")
FIELDS = ("config", "T", "M", "hit_after", "accepted_has_match", "has_match", "accepted_precision",
          "coverage", "false_acceptance", "no_match")


def is_accepted(top, threshold, min_image_similarity):
    if top is None:
        return False
    similarity = top.get("image_similarity")
    return top["final_score"] >= threshold and (
        min_image_similarity is None or similarity is None or similarity >= min_image_similarity)


def evaluate(snapshot, config, threshold, min_image_similarity, relevant, statuses):
    total = positives = hits = accepted_positive = accepted_all = negatives = false_accepted = 0
    for query in snapshot["queries"]:
        qid = query["query_id"]
        result = query["results"].get(config)
        if result is None:
            continue
        target = result["target"]["index"]
        ranked = result["detections"][target]["ranked"] if target is not None else []
        accepted = is_accepted(ranked[0] if ranked else None, threshold, min_image_similarity)
        total += 1
        accepted_all += accepted
        if statuses.get(qid) == "has_match":
            positives += 1
            hit = any(product["id"] in relevant.get(qid, ()) for product in ranked[:5])
            hits += hit and accepted
            accepted_positive += accepted
        elif statuses.get(qid) == "no_match":
            negatives += 1
            false_accepted += accepted
    return {
        "hit_after": hits / positives if positives else None, "accepted_has_match": accepted_positive,
        "has_match": positives,
        "accepted_precision": hits / accepted_positive if accepted_positive else None,
        "coverage": accepted_all / total if total else None,
        "false_acceptance": false_accepted, "no_match": negatives,
    }


def judgments(root=None):
    pairs, statuses = load_judgments() if root is None else load_judgments(root)
    relevant = {}
    for (qid, pid), pair in pairs.items():
        if pair["acceptable"]:
            relevant.setdefault(qid, set()).add(pid)
    return relevant, statuses


def sweep(snapshot, relevant, statuses):
    rows = []
    for config in CONFIGS:
        for threshold in T_GRID:
            for minimum in M_GRID:
                metrics = evaluate(snapshot, config, threshold, minimum, relevant, statuses)
                rows.append({"config": config, "T": threshold, "M": minimum, **metrics})
    return rows


def select(rows, config):
    candidates = [row for row in rows if row["config"] == config]
    return max(candidates, key=lambda row: (row["hit_after"], row["T"], -1 if row["M"] is None else row["M"]))


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: "" if row[key] is None else row[key] for key in FIELDS} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    relevant, statuses = judgments()
    lock_path = args.run_dir / "locked_thresholds.json"
    if args.command == "fit":
        snapshot = read_json(args.run_dir / "validation" / "results.json")
        rows = sweep(snapshot, relevant, statuses)
        write_csv(args.run_dir / "validation" / "threshold_sweep.csv", rows)
        lock = {"rule": "max HitRate@5 after decision on validation; ties -> highest T, then highest M",
                "validation_manifest_sha256": snapshot["manifest_sha256"], "thresholds": {}}
        for config in CONFIGS:
            best = select(rows, config)
            lock["thresholds"][config] = {"T": best["T"], "M": best["M"], "validation": {
                key: best[key] for key in FIELDS[3:]}}
            print(config, {key: best[key] for key in ("T", "M", "hit_after", "accepted_has_match",
                                                      "accepted_precision", "coverage", "false_acceptance")})
        lock_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False), encoding="utf-8")
        print(lock_path)
    else:
        lock = read_json(lock_path)
        snapshot = read_json(args.run_dir / "test" / "results.json")
        for config, chosen in lock["thresholds"].items():
            if all(config in query["results"] for query in snapshot["queries"]):
                metrics = evaluate(snapshot, config, chosen["T"], chosen["M"], relevant, statuses)
                print(config, {"T": chosen["T"], "M": chosen["M"], **metrics})


if __name__ == "__main__":
    main()
