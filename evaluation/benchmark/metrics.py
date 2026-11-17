"""Target-level retrieval metrics over human judgments."""

from collections import Counter
from math import ceil, floor
from statistics import mean

from evaluation.run_m1_ablation import box_iou

TIMING_FIELDS = ("preprocess", "detect", "embed", "retrieve", "rerank", "total")


def match_target(detections: list[dict], class_name: str, target_bbox: list[float],
                 iou_threshold: float = 0.5) -> dict:
    index, best_iou, best_any = None, 0.0, 0.0
    for position, detection in enumerate(detections):
        iou = box_iou(detection["bbox"], target_bbox)
        best_any = max(best_any, iou)
        if detection["class_name"] == class_name and iou > best_iou:
            index, best_iou = position, iou
    return {
        "index": index if best_iou >= iou_threshold else None,
        "iou": best_iou,
        "best_iou_any_class": best_any,
    }


def first_relevant_rank(ranked_ids, relevant, k):
    return next((rank for rank, pid in enumerate(ranked_ids[:k], 1) if pid in relevant), None)


def hit_at_k(ranked_ids, relevant, k):
    return first_relevant_rank(ranked_ids, relevant, k) is not None


def reciprocal_rank_at_k(ranked_ids, relevant, k):
    rank = first_relevant_rank(ranked_ids, relevant, k)
    return 1 / rank if rank else 0.0


def precision_at_k(ranked_ids, relevant, k):
    return sum(pid in relevant for pid in ranked_ids[:k]) / k


def joint_at_k(ranked_ids, type_ok, color_ok, k):
    return hit_at_k(ranked_ids, type_ok & color_ok, k)


def failure_category(matched, best_iou_any_class, candidate_hit, hit, accepted):
    if not matched:
        return "no_detection" if best_iou_any_class < 0.5 else "wrong_class"
    if candidate_hit is False:
        return "candidate_miss"
    if not hit:
        return "rank_miss" if candidate_hit else None
    return "hit" if accepted else "rejected"


def paired_outcomes(baseline_rr: dict[str, float], other_rr: dict[str, float]) -> dict:
    counts = dict.fromkeys(("win", "loss", "tie"), 0)
    for query_id in baseline_rr.keys() & other_rr.keys():
        diff = other_rr[query_id] - baseline_rr[query_id]
        counts["win" if diff > 0 else "loss" if diff < 0 else "tie"] += 1
    return counts


def latency_stats(values_ms: list[float]) -> dict:
    if not values_ms:
        return {"mean": None, "p95": None, "n": 0}
    ordered = sorted(values_ms)
    position = (len(ordered) - 1) * 0.95
    lower, upper = floor(position), ceil(position)
    p95 = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {"mean": mean(ordered), "p95": p95, "n": len(ordered)}


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict:
    matrix = {}
    for actual, predicted in pairs:
        counts = matrix.setdefault(actual, {})
        counts[predicted] = counts.get(predicted, 0) + 1
    return matrix


def summarize(per_query: list[dict], qrels, status) -> dict:
    """Return numerator/denominator pairs, per-query rows and diagnostic counts."""
    judgments = {}
    for (qid, pid), grade in qrels.items():
        judgments.setdefault(qid, {})[pid] = grade

    details = []
    for row in per_query:
        qid = row["query_id"]
        index = row["target"]["index"]
        detection = row["detections"][index] if index is not None else {}
        ranked = [str(product["id"]) for product in detection.get("ranked", [])][:5]
        grades = judgments.get(qid, {})
        relevant = {pid for pid, grade in grades.items() if grade["acceptable"]}
        relevance = "has_match" if relevant else status.get(qid, "unconfirmed")
        acceptable_count = sum(pid in relevant for pid in ranked)
        rr = reciprocal_rank_at_k(ranked, relevant, 5)
        matched, accepted = index is not None, bool(detection.get("accepted"))
        has_match = relevance == "has_match"
        candidates_available = row.get("candidates_available", True)
        candidate_hit = bool(relevant.intersection(detection.get("candidate_ids", ()))) if candidates_available else None
        failure = failure_category(matched, row["target"]["best_iou_any_class"],
                                   candidate_hit, bool(rr), accepted) if has_match else None
        details.append({
            "query_id": qid, "relevance_status": relevance, "lighting_group": row["lighting_group"],
            "matched": matched, "accepted": accepted, "ranked_ids": ranked,
            "unjudged_at_5": sum(pid not in grades for pid in ranked),
            "hit_at_5": bool(rr) if has_match else None, "rr_at_5": rr if has_match else None,
            "acceptable_count": acceptable_count if has_match else None,
            "joint_at_5": any(grades.get(pid, {}).get("type_ok") and grades[pid]["color_ok"] for pid in ranked),
            "hit_after_at_5": bool(rr and accepted) if has_match else None,
            "rr_after_at_5": rr * accepted if has_match else None,
            "candidate_hit_at_100": candidate_hit if has_match else None,
            "label_color": row.get("color"),
            "predicted_color": detection.get("tight_color_name", detection.get("query_color_name")),
            "detect_measured": row.get("detection_measured", True), "failure": failure,
        })

    confirmed = [row for row in details if row["relevance_status"] != "unconfirmed"]
    positive = [row for row in confirmed if row["relevance_status"] == "has_match"]
    negative = [row for row in confirmed if row["relevance_status"] == "no_match"]
    # Target matching and query color need no product judgments, so every query counts.
    detectable = [row for row in details if row["detect_measured"]]
    colors = [(row["label_color"], row["predicted_color"] or "(trống)") for row in detectable if row["matched"]]

    def ratio(rows, field):
        return sum(row[field] for row in rows), len(rows)

    target_by_lighting = {
        group: ratio([row for row in detectable if row["lighting_group"] == group], "matched")
        for group in ("dark", "normal")
    }
    latency = {}
    for field in TIMING_FIELDS:
        values = [row["timings_ms"].get(field) for row in per_query]
        latency[field] = {
            "first": values[0] if values else None,
            "warm": latency_stats([value for value in values[1:] if value is not None]),
        }

    return {
        "target_recall": ratio(detectable, "matched"),
        "target_by_lighting": target_by_lighting,
        "hit_at_5": ratio(positive, "hit_at_5"),
        "mrr_at_5": ratio(positive, "rr_at_5"),
        "precision_at_5": (sum(row["acceptable_count"] for row in positive), 5 * len(positive)),
        "joint_at_5": (sum(bool(row["joint_at_5"]) for row in confirmed), len(confirmed)),
        "hit_after_at_5": ratio(positive, "hit_after_at_5"),
        "mrr_after_at_5": ratio(positive, "rr_after_at_5"),
        "coverage": ratio(confirmed, "accepted"),
        "false_acceptance": ratio(negative, "accepted"),
        "candidate_hit_at_100": ratio([row for row in positive if row["candidate_hit_at_100"] is not None],
                                      "candidate_hit_at_100"),
        "color_accuracy": (sum(actual == predicted for actual, predicted in colors), len(colors)),
        "confusion_matrix": confusion_matrix(colors),
        "latency": latency,
        "failures": dict(Counter(row["failure"] for row in positive if row["failure"])),
        "unconfirmed": [row["query_id"] for row in details if row["relevance_status"] == "unconfirmed"],
        "unjudged_at_5": sum(row["unjudged_at_5"] for row in details),
        "per_query": details,
    }

