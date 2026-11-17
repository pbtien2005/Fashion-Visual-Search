import unittest

from evaluation.benchmark.threshold import evaluate, is_accepted, select


def query(qid, scores, ids):
    ranked = [{"id": pid, "final_score": score, "image_similarity": score} for pid, score in zip(ids, scores)]
    return {"query_id": qid, "results": {"X": {"target": {"index": 0}, "detections": [{"ranked": ranked}]}}}


class ThresholdTests(unittest.TestCase):
    def test_boundary_is_inclusive_and_similarity_floor_applies(self):
        self.assertTrue(is_accepted({"final_score": 0.5, "image_similarity": 0.4}, 0.5, 0.4))
        self.assertFalse(is_accepted({"final_score": 0.49, "image_similarity": 0.9}, 0.5, None))
        self.assertFalse(is_accepted({"final_score": 0.9, "image_similarity": 0.3}, 0.5, 0.4))
        self.assertFalse(is_accepted(None, 0.0, None))

    def test_rejected_hit_does_not_count_and_no_match_counts_false_acceptance(self):
        snapshot = {"queries": [query("a", [0.8], ["p1"]), query("b", [0.3], ["p2"]), query("n", [0.8], ["p3"])]}
        relevant = {"a": {"p1"}, "b": {"p2"}}
        statuses = {"a": "has_match", "b": "has_match", "n": "no_match"}
        result = evaluate(snapshot, "X", 0.5, None, relevant, statuses)
        self.assertEqual((result["hit_after"], result["accepted_has_match"], result["false_acceptance"]), (0.5, 1, 1))

    def test_tie_prefers_higher_threshold(self):
        rows = [{"config": "X", "T": 0.4, "M": None, "hit_after": 0.5},
                {"config": "X", "T": 0.5, "M": 0.3, "hit_after": 0.5},
                {"config": "X", "T": 0.5, "M": None, "hit_after": 0.5},
                {"config": "X", "T": 0.6, "M": None, "hit_after": 0.4}]
        self.assertEqual((select(rows, "X")["T"], select(rows, "X")["M"]), (0.5, 0.3))


if __name__ == "__main__":
    unittest.main()
