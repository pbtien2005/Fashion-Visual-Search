"""Benchmark v1: the embedding benchmark (v0, run_models.py) plus whole-image fallback configurations.

Why: in v0, 22/47 test queries fail before retrieval, because YOLO either labels the target with the
wrong class (14) or puts no box on it (8). Inside the 25 queries where YOLO finds the target, B0
hits 20.

Finding that changed the plan: at conf 0.5 YOLO returns at least one box on all 75/75 benchmark
images. "no_detection" means the boxes miss the target, not that YOLO returned nothing. So the
first idea (use the whole image only when YOLO finds nothing) never triggers. It is kept as F1 so
that this is measured, not assumed.

Configurations added (all other settings from v0, nothing re-tuned):
  F1_B0_nodet  B0; if YOLO returns no box at all -> whole-image search (A0 path).
  F2_B0_whole  B0 groups plus one extra "whole image" group (A0 cosine ranking).
  F2_C2_whole  C2 groups plus the same whole-image group (pure embedding comparison).

Which group the user is assumed to pick (same assumption as v0, which already scores "the group
that matches the labelled item"): the YOLO group of the labelled class with IoU >= 0.5 if there is
one, otherwise the whole-image group. The whole-image group goes through the production decision
(apply_match_threshold, 0.70 / 0.60) like every ROI group, so false acceptance on no_match queries
is measured.

The fallbacks are composed from the stored v0 results of their parts (B0 or C2, and A0). Each part
is deterministic and independent of the others, so the result equals running the composite
pipeline; latency adds the whole-image embedding and retrieval to the ROI time (preprocessing is
shared, counted once).

Protocol:
  1. python -m evaluation_new.benchmark_v1 --stage validation
     composes validation and writes evaluation_new/locked_v1.json (definitions, adoption rule,
     validation numbers). Test results are not read at this stage.
  2. python -m evaluation_new.benchmark_v1 --stage test
     refuses to run without the lock, then composes test.
Caveat: the fallback idea came from the v0 test failure analysis (not from validation), so the v1
test numbers are confirmatory, not blind. The decision rule is applied as locked.
"""

import argparse
import copy
import json
from datetime import datetime, timezone

from .run_models import RESULTS, ROOT, sha256_file

V1_RESULTS = ROOT / "evaluation_new/results_v1"
LOCK = ROOT / "evaluation_new/locked_v1.json"
WHOLE = "A0_whole_image"
FALLBACKS = {
    "F1_B0_nodet": {"base": "B0_current", "mode": "no_box"},
    "F2_B0_whole": {"base": "B0_current", "mode": "whole_group"},
    "F2_C2_whole": {"base": "C2_embedding_only", "mode": "whole_group"},
}
ADOPTION_RULE = (
    "F2_B0_whole replaces B0_current as the main configuration if, on validation for FashionCLIP "
    "(the production model; thresholds are tuned for it): HitRate@5 gains >= 1 query, MRR@5 does not "
    "drop, and false acceptance on no_match queries does not increase. Test is then reported as is."
)


def _whole_decision(detection):
    """Production decision for the whole-image group (A0 itself reports accepted=True, no threshold)."""
    from api.search_service import apply_match_threshold

    decision = apply_match_threshold(detection["ranked"])
    return {**detection, "accepted": decision["is_confident_match"], "reject_reason": decision["reject_reason"],
            "group": "whole_image"}


def _timings(roi, whole):
    timings = dict(roi["timings_ms"])
    for phase, value in whole["timings_ms"].items():
        if phase not in ("preprocess", "total"):
            timings[phase] = timings.get(phase, 0.0) + value
    timings["total"] = sum(value for phase, value in timings.items() if phase != "total")
    return timings


def compose(roi, whole, mode):
    if "error" in roi or "error" in whole:
        bad = roi if "error" in roi else whole
        return {**copy.deepcopy(bad), "fallback_used": None}
    if mode == "no_box":
        if roi["detections"]:
            return {**copy.deepcopy(roi), "fallback_used": False}
        result = copy.deepcopy(whole)
        result.update(detections=[_whole_decision(whole["detections"][0])], timings_ms=_timings(roi, whole),
                      fallback_used=True)
        return result  # whole-image result: target index 0, as attach_target does for A0
    result = copy.deepcopy(roi)
    group = _whole_decision(copy.deepcopy(whole["detections"][0]))
    result["detections"] = [*result["detections"], group]
    result["timings_ms"] = _timings(roi, whole)
    target = dict(roi["target"])
    used = target["index"] is None and target.get("bbox") is not None
    if used:
        target["index"] = len(result["detections"]) - 1  # the user opens the whole-image group
    result.update(target=target, fallback_used=used)
    return result


def build_split(model_key, split):
    source = RESULTS / model_key / split / "results.json"
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    for query in snapshot["queries"]:
        results = query["results"]
        for name, spec in FALLBACKS.items():
            if spec["base"] in results:
                results[name] = compose(results[spec["base"]], results[WHOLE], spec["mode"])
    for name, spec in FALLBACKS.items():
        if spec["base"] in snapshot["configs"]:
            snapshot["configs"][name] = {"name": name, "kind": "fallback", **spec, "whole": WHOLE}
    snapshot["benchmark_version"] = "v1"
    snapshot["v0_results_sha256"] = sha256_file(source)
    snapshot["fallback_usage"] = {
        name: sum(bool(q["results"][name].get("fallback_used")) for q in snapshot["queries"])
        for name in FALLBACKS if name in snapshot["configs"]}
    out = V1_RESULTS / model_key / split / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    return snapshot


def models_with_runs():
    return sorted({path.parent.parent.name for path in RESULTS.glob("*/*/results.json")})


def run_validation():
    from .scoring import collect

    models = models_with_runs()
    usage = {key: build_split(key, "validation")["fallback_usage"] for key in models}
    summary, _, _ = collect(("validation",), V1_RESULTS)
    rows = summary[summary["model"] == "fashionclip"].set_index("config")
    base, f2 = rows.loc["B0_current"], rows.loc["F2_B0_whole"]
    checks = {
        "hit_gain_queries": int(f2["hit_at_5_n"] - base["hit_at_5_n"]),
        "mrr_delta": float(f2["mrr_at_5"] - base["mrr_at_5"]),
        "false_acceptance_delta": int(f2["false_acceptance_n"] - base["false_acceptance_n"]),
    }
    adopted = checks["hit_gain_queries"] >= 1 and checks["mrr_delta"] >= 0 and checks["false_acceptance_delta"] <= 0
    lock = {
        "benchmark_version": "v1", "locked_at": datetime.now(timezone.utc).isoformat(),
        "definitions": FALLBACKS, "whole_group_source": WHOLE,
        "group_choice": "labelled-class YOLO group with IoU >= 0.5, else the whole-image group",
        "whole_group_decision": "api.search_service.apply_match_threshold (production 0.70 / 0.60)",
        "adoption_rule": ADOPTION_RULE,
        "validation": {"fashionclip": {
            config: {"hit_at_5": f"{int(rows.loc[config, 'hit_at_5_n'])}/{int(rows.loc[config, 'has_match'])}",
                     "mrr_at_5": round(float(rows.loc[config, "mrr_at_5"]), 4),
                     "false_acceptance": f"{int(rows.loc[config, 'false_acceptance_n'])}/{int(rows.loc[config, 'no_match'])}"}
            for config in ("B0_current", *FALLBACKS)}, "fallback_usage": usage, "checks": checks},
        "adopted_main_config": "F2_B0_whole" if adopted else "B0_current",
        "v0_validation_results_sha256": {key: sha256_file(RESULTS / key / "validation/results.json")
                                         for key in models},
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: lock[k] for k in ("validation", "adopted_main_config")}, ensure_ascii=False, indent=1))


def run_test():
    if not LOCK.exists():
        raise SystemExit("Chưa có evaluation_new/locked_v1.json: chạy --stage validation trước")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["definitions"] != FALLBACKS:
        raise SystemExit("Định nghĩa cấu hình đã đổi sau khi khóa trên validation")
    for key in models_with_runs():
        if sha256_file(RESULTS / key / "validation/results.json") != lock["v0_validation_results_sha256"].get(key):
            raise SystemExit(f"Kết quả validation v0 của {key} đã đổi sau khi khóa")
        print(key, build_split(key, "test")["fallback_usage"])
    print("Cấu hình chính (đã khóa):", lock["adopted_main_config"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    run_validation() if parser.parse_args().stage == "validation" else run_test()
