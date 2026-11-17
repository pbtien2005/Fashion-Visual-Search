import unittest

from evaluation.benchmark.metrics import (
    failure_category, hit_at_k, joint_at_k, latency_stats, match_target,
    precision_at_k, reciprocal_rank_at_k, summarize,
)


class MetricsTests(unittest.TestCase):
    def test_detection_threshold_class_and_order(self):
        target = [0, 0, 100, 100]

        def detection(width, name="top"):
            return {"bbox": [0, 0, width, 100], "class_name": name}

        for width, index in [(49, None), (50, 0)]:
            self.assertEqual(match_target([detection(width)], "top", target)["index"], index)
        self.assertEqual(match_target([detection(52), detection(80)], "top", target)["index"], 1)
        wrong = match_target([detection(90, "skirt")], "top", target)
        self.assertIsNone(wrong["index"])
        self.assertEqual(wrong["best_iou_any_class"], 0.9)

    def test_top_five_and_joint(self):
        ids = list("abcdef")
        self.assertFalse(hit_at_k(ids, {"f"}, 5))
        self.assertEqual(reciprocal_rank_at_k(ids, {"f"}, 5), 0)
        self.assertFalse(joint_at_k(ids, {"a"}, {"b"}, 5))
        self.assertEqual(precision_at_k(["a"], {"a"}, 5), 0.2)

    def test_failure_priority(self):
        for name, inputs in {
            "no_detection": (False, .49, False, False, False),
            "wrong_class": (False, .50, False, False, False),
            "candidate_miss": (True, 1, False, True, True),
            "rank_miss": (True, 1, True, False, False),
            "rejected": (True, 1, True, True, False),
            "hit": (True, 1, True, True, True),
        }.items():
            with self.subTest(name=name):
                self.assertEqual(failure_category(*inputs), name)

    def test_aggregate_denominators_and_unjudged(self):
        rows = []
        for query, index, accepted in [("a", 0, False), ("b", 0, True), ("c", None, False),
                                       ("unconfirmed", 0, True)]:
            rows.append({
                "query_id": query, "lighting_group": "dark", "color": "red",
                "target": {"index": index, "best_iou_any_class": float(index is not None)},
                "detections": [{"accepted": accepted, "ranked": [{"id": "yes"}, {"id": "missing"}],
                                "candidate_ids": ["yes"], "query_color_name": "orange"}],
                "timings_ms": {"total": 5},
            })
        valid = {"type_ok": 1, "color_ok": 1, "acceptable": 1}
        invalid = dict.fromkeys(valid, 0)
        qrels = {("a", "yes"): valid, ("b", "yes"): invalid, ("c", "yes"): valid}
        report = summarize(rows, qrels, {"a": "has_match", "b": "no_match", "c": "has_match"})
        self.assertEqual(report["hit_at_5"], (1, 2))
        self.assertEqual(report["mrr_at_5"], (1.0, 2))
        self.assertEqual(report["hit_after_at_5"], (0, 2))
        self.assertEqual(report["false_acceptance"], (1, 1))
        # Detection needs no product judgments, so the unconfirmed query still counts.
        self.assertEqual(report["target_recall"], (3, 4))
        self.assertEqual(report["coverage"], (1, 3))
        self.assertEqual(report["unjudged_at_5"], 4)
        self.assertEqual(report["unconfirmed"], ["unconfirmed"])
        self.assertEqual(report["failures"], {"rejected": 1, "no_detection": 1})
        self.assertEqual(report["latency"]["total"]["warm"]["n"], 3)

    def test_latency_empty_and_single(self):
        self.assertEqual(latency_stats([]), {"mean": None, "p95": None, "n": 0})
        self.assertEqual(latency_stats([2]), {"mean": 2, "p95": 2, "n": 1})
        self.assertAlmostEqual(latency_stats([0, 10])["p95"], 9.5)

