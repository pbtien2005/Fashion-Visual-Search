"""Re-score stored inference with ambiguous garment classes merged into one group.

The group is fixed before looking at any result: long_sleeve_top, long_sleeve_outwear and vest
are one "layered upper-body" group because their boundary is subjective (blazer, cardigan,
open shirt, gilet). Nothing is re-run: detections and rankings come from results.json and the
human judgments from bench_qrels.csv. A product that was never graded counts as not relevant,
so the grouped numbers are a lower bound; `ungraded_top5` reports how large that gap can be.
"""
import argparse
import json
from pathlib import Path

from evaluation.run_m1_ablation import box_iou

from .grade import load_judgments
from .metrics import hit_at_k, reciprocal_rank_at_k

GROUPS = {"long_sleeve_top": "layered_upper", "long_sleeve_outwear": "layered_upper", "vest": "layered_upper"}
CONFIGS = ("B0_current", "B1_gamma", "C2_embedding_only", "C3_w1", "C3_w2", "C3_selected", "C4_group_terms", "C5_hsv_color", "C5_hsv_color_w1", "C6_center", "C6_intersection", "C6_center_intersection")


def group_of(name):
    return GROUPS.get(name, name)


def match(detections, class_name, box, grouped, threshold=0.5):
    best, index = 0.0, None
    for position, detection in enumerate(detections):
        same = group_of(detection["class_name"]) == group_of(class_name) if grouped else (
            detection["class_name"] == class_name)
        iou = box_iou(detection["bbox"], box)
        if same and iou > best:
            best, index = iou, position
    return index if best >= threshold else None


def score(snapshot, config, relevant, graded, statuses, grouped):
    n = matched = hits = ungraded = shown = 0
    rr_total = 0.0
    for query in snapshot["queries"]:
        result = query["results"].get(config)
        qid = query["query_id"]
        if result is None or statuses.get(qid) != "has_match":
            continue
        n += 1
        width, height = result["geometry"]["width"], result["geometry"]["height"]
        x1, y1, x2, y2 = query["bbox"]
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
        index = match(result["detections"], query["class_name"], box, grouped)
        if index is None:
            continue
        matched += 1
        ids = [product["id"] for product in result["detections"][index]["ranked"][:5]]
        ungraded += sum((qid, pid) not in graded for pid in ids)
        shown += len(ids)
        hits += hit_at_k(ids, relevant.get(qid, set()), 5)
        rr_total += reciprocal_rank_at_k(ids, relevant.get(qid, set()), 5)
    return {"has_match": n, "detected": matched, "hit_at_5": hits, "mrr_sum": round(rr_total, 3),
            "ungraded_top5": f"{ungraded}/{shown}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    snapshot = json.loads((args.run_dir / args.stage / "results.json").read_text(encoding="utf-8"))
    pairs, statuses = load_judgments()
    relevant = {}
    for (qid, pid), pair in pairs.items():
        if pair["acceptable"]:
            relevant.setdefault(qid, set()).add(pid)
    for config in CONFIGS:
        if not all(config in query["results"] for query in snapshot["queries"]):
            continue
        for grouped in (False, True):
            row = score(snapshot, config, relevant, set(pairs), statuses, grouped)
            print(config, "grouped" if grouped else "strict ", row)


if __name__ == "__main__":
    main()
