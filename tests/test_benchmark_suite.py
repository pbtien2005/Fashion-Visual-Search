import unittest

from evaluation.run_benchmark import evaluate_gates, latency_score, suite_score


class BenchmarkSuiteTests(unittest.TestCase):
    def test_latency_score_rewards_fast_responses(self):
        self.assertEqual(latency_score(500), 1.0)
        self.assertEqual(latency_score(8000), 0.0)
        self.assertGreater(latency_score(3000), latency_score(6000))

    def test_suite_score_skips_missing_exact_match_metrics(self):
        score = suite_score(
            {
                "detection_success_rate": 0.9,
                "category_at_5": 0.8,
                "color_at_5": 0.5,
                "recall_at_5": None,
                "mrr": None,
                "avg_latency_ms": 1000,
            }
        )

        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)

    def test_evaluate_gates_reports_threshold_failures(self):
        passed, failures = evaluate_gates(
            {"category_at_5": 0.6, "failed_queries": 1},
            {"category_at_5": 0.7, "failed_queries": 0},
        )

        self.assertFalse(passed)
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
