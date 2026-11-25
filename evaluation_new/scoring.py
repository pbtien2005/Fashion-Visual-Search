"""Score saved runs with the old benchmark's scorer and collect tidy tables for the notebook.

Main metrics come from evaluation.benchmark.score.config_rows (unchanged). Two additions:

* judged-only ("condensed") HitRate@5 / MRR@5: drop products nobody graded from the top-10,
  then score the first 5 that remain. Judgments mostly cover products FashionCLIP ranked
  high, so the strict numbers count every new, ungraded product as a miss.
* paired comparison per query against FashionCLIP (re-embedded) with an exact sign test.
"""

import json
from math import comb
from pathlib import Path

import pandas as pd

from evaluation.benchmark.grade import load_judgments
from evaluation.benchmark.metrics import reciprocal_rank_at_k
from evaluation.benchmark.score import config_rows

from .models import MODELS
from .run_models import RESULTS, STORED

REFERENCE = "fashionclip"
CONFIG_ORDER = ["A0_whole_image", "C2_embedding_only", "F2_C2_whole", "B0_current", "F1_B0_nodet",
                "F2_B0_whole", "C3_selected"]
CONFIG_LABELS = {"A0_whole_image": "A0 ảnh nguyên", "C2_embedding_only": "C2 ROI + cosine",
                 "B0_current": "B0 hybrid", "C3_selected": "C3 hybrid (w1)",
                 # benchmark v1 (evaluation_new/benchmark_v1.py)
                 "F1_B0_nodet": "F1 B0, YOLO rỗng → ảnh nguyên", "F2_B0_whole": "F2 B0 + nhóm ảnh nguyên",
                 "F2_C2_whole": "F2 C2 + nhóm ảnh nguyên"}


def model_label(key):
    return "FashionCLIP (vector gốc)" if key == STORED else MODELS[key].label


def model_order(results=RESULTS):
    return [key for key in (STORED, *MODELS) if any((results / key).glob("*/results.json"))]


def load_run(model_key, split, results=RESULTS):
    path = results / model_key / split / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ratio(pair):
    numerator, denominator = pair
    return numerator / denominator if denominator else float("nan")


def judged_only(snapshot, config, qrels, status, k=5):
    """Condensed HitRate@k / MRR@k over has_match queries (unjudged products removed)."""
    graded = {}
    for (qid, pid), grade in qrels.items():
        graded.setdefault(qid, {})[pid] = grade
    hits = rr_sum = n = 0
    for query in snapshot["queries"]:
        qid = query["query_id"]
        grades = graded.get(qid, {})
        relevant = {pid for pid, grade in grades.items() if grade["acceptable"]}
        if not relevant:
            continue
        n += 1
        result = query["results"][config]
        index = result["target"]["index"]
        if index is None:
            continue
        ranked = [str(p["id"]) for p in result["detections"][index].get("ranked", [])]
        condensed = [pid for pid in ranked if pid in grades][:k]
        rr = reciprocal_rank_at_k(condensed, relevant, k)
        hits += bool(rr)
        rr_sum += rr
    return hits, rr_sum, n


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def collect(splits=("test", "validation"), results=RESULTS):
    """Return (summary, per_query, failures) DataFrames over every finished run in `results`."""
    qrels, status = load_judgments()
    summary, per_query, failures = [], [], []
    for split in splits:
        reports_by_model = {}
        for key in model_order(results):
            snapshot = load_run(key, split, results)
            if snapshot is None:
                continue
            reports = config_rows(snapshot, qrels, status)
            reports_by_model[key] = reports
            for config, report in reports.items():
                hits, rr_sum, n = judged_only(snapshot, config, qrels, status)
                latency = report["latency"]
                summary.append({
                    "split": split, "model": key, "label": model_label(key), "config": config,
                    "group": snapshot["model"]["group"], "dimension": snapshot["dimension"],
                    "hit_at_5": _ratio(report["hit_at_5"]), "hit_at_5_n": report["hit_at_5"][0],
                    "has_match": report["hit_at_5"][1],
                    "mrr_at_5": _ratio(report["mrr_at_5"]),
                    "precision_at_5": _ratio(report["precision_at_5"]),
                    "joint_at_5": _ratio(report["joint_at_5"]),
                    "candidate_hit_at_100": _ratio(report["candidate_hit_at_100"]),
                    "hit_after_at_5": _ratio(report["hit_after_at_5"]),
                    "coverage": _ratio(report["coverage"]),
                    # no_match queries where a result was still accepted (after the 0.70/0.60 thresholds)
                    "false_acceptance_n": report["false_acceptance"][0],
                    "no_match": report["false_acceptance"][1],
                    "target_recall": _ratio(report["target_recall"]),
                    "unjudged_at_5": report["unjudged_at_5"],
                    "judged_hit_at_5": hits / n if n else float("nan"),
                    "judged_mrr_at_5": rr_sum / n if n else float("nan"),
                    "embed_ms_mean": latency["embed"]["warm"]["mean"],
                    "total_ms_mean": latency["total"]["warm"]["mean"],
                    "total_ms_p95": latency["total"]["warm"]["p95"],
                    "catalog_embed_seconds": snapshot.get("catalog_embed_seconds"),
                    "errors": sum(1 for q in snapshot["queries"] if q["results"][config].get("error")),
                    "text_style_skipped_batches": snapshot.get("text_style_skipped_batches"),
                })
                for row in report["per_query"]:
                    per_query.append({"split": split, "model": key, "config": config,
                                      **{f: row[f] for f in ("query_id", "relevance_status", "matched",
                                                             "hit_at_5", "rr_at_5", "unjudged_at_5",
                                                             "failure")},
                                      "ranked_ids": json.dumps(row["ranked_ids"])})
                failures.append({"split": split, "model": key, "config": config, **report["failures"]})
    summary = pd.DataFrame(summary)
    per_query = pd.DataFrame(per_query)
    failures = pd.DataFrame(failures).fillna(0)
    return summary, per_query, failures


def paired_vs_reference(per_query, reference=REFERENCE):
    """Wins/losses/ties of RR@5 per has_match query against the reference model, same config."""
    rows = []
    positive = per_query[per_query["relevance_status"] == "has_match"]
    for (split, config), group in positive.groupby(["split", "config"]):
        base = group[group["model"] == reference].set_index("query_id")["rr_at_5"]
        if base.empty:
            continue
        for model, other in group.groupby("model"):
            if model == reference:
                continue
            diff = other.set_index("query_id")["rr_at_5"].reindex(base.index) - base
            wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
            rows.append({"split": split, "config": config, "model": model, "win": wins, "loss": losses,
                         "tie": int((diff == 0).sum()), "sign_test_p": sign_test(wins, losses)})
    return pd.DataFrame(rows)


def class_breakdown(per_query, split="test", config="B0_current"):
    """HitRate@5 by labelled garment class."""
    from evaluation.benchmark.manifest import load_manifest, manifest_path

    classes = {row["query_id"]: row["class_name"] for row in load_manifest(manifest_path(split))}
    frame = per_query[(per_query["split"] == split) & (per_query["config"] == config)
                      & (per_query["relevance_status"] == "has_match")].copy()
    frame["class_name"] = frame["query_id"].map(classes)
    frame["hit_at_5"] = frame["hit_at_5"].astype(float)
    return frame.pivot_table(index="class_name", columns="model", values="hit_at_5", aggfunc="mean")


def write_tables(out_dir: Path, summary, per_query, failures, paired):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_query.to_csv(out_dir / "per_query.csv", index=False)
    failures.to_csv(out_dir / "failures.csv", index=False)
    paired.to_csv(out_dir / "paired_vs_fashionclip.csv", index=False)
