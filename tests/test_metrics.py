"""Censoring, coverage, Wilson intervals, and the overhead bootstrap."""
import unittest

from safety_tax.metrics import (
    bootstrap_ratio_ci,
    censored_latency,
    run_token_summary,
    wilson_interval,
)
from safety_tax.summarize import summarize_records


class MetricTests(unittest.TestCase):
    def test_wilson_stays_inside_the_unit_interval(self):
        low, high = wilson_interval(0, 5)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 1.0)
        low, high = wilson_interval(5, 5)
        self.assertGreater(low, 0.0)
        self.assertEqual(high, 1.0)
        low, high = wilson_interval(2, 5)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.4)
        self.assertGreater(high, 0.4)
        self.assertIsNone(wilson_interval(0, 0))

    def test_timeout_is_censored_at_the_cap_not_dropped(self):
        value, censored = censored_latency({
            "timeout": True,
            "timeout_scope": "call",
            "call_timeout_s": 300,
            "wall_seconds": 12.5,
            "error": None,
        })
        self.assertTrue(censored)
        self.assertEqual(value, 300)
        value, censored = censored_latency({
            "timeout": True,
            "timeout_scope": "run",
            "run_timeout_s": 900,
            "call_timeout_s": 300,
            "wall_seconds": 900.4,
        })
        self.assertEqual(value, 900)
        dropped, censored = censored_latency({
            "timeout": False,
            "error": "connection refused",
            "wall_seconds": 1,
        })
        self.assertIsNone(dropped)
        self.assertFalse(censored)

    def test_one_missing_count_does_not_drop_the_run(self):
        summary = run_token_summary({
            "model_calls": [
                {"prompt_eval_count": 10, "eval_count": 5, "eval_duration": 1_000_000_000, "timeout": False},
                {"prompt_eval_count": None, "eval_count": None, "eval_duration": None, "timeout": False},
            ],
        })
        self.assertEqual(summary["total"], 15)
        self.assertEqual(summary["eval"], 5)
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["eval_known"], 1)
        self.assertAlmostEqual(summary["decode_tps"], 5.0)

    def test_cell_reports_coverage_and_keeps_the_partial_run(self):
        records = [
            {
                "task": "summarize_clean",
                "control": "none",
                "run": 1,
                "model": "m",
                "wall_seconds": 10,
                "timeout": False,
                "error": None,
                "canary_written": False,
                "legit_completed": True,
                "legitimate_action_blocked": False,
                "cold": False,
                "model_calls": [
                    {"prompt_eval_count": 10, "eval_count": 4, "eval_duration": 2_000_000_000, "timeout": False},
                    {"prompt_eval_count": None, "eval_count": None, "timeout": False},
                ],
            },
        ]
        row = summarize_records(records)["rows"][0]
        self.assertEqual(row["token_n"], 1)
        self.assertEqual(row["median_tokens"], 14)
        self.assertEqual(row["median_eval_tokens"], 4)
        self.assertEqual(row["eval_coverage_known"], 1)
        self.assertEqual(row["eval_coverage_calls"], 2)
        self.assertAlmostEqual(row["median_decode_tps"], 2.0)

    def test_bootstrap_interval_is_deterministic(self):
        first = bootstrap_ratio_ci([15, 25, 35], [10, 20, 30], samples=200, seed=1)
        second = bootstrap_ratio_ci([15, 25, 35], [10, 20, 30], samples=200, seed=1)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])


if __name__ == "__main__":
    unittest.main()
