"""Summarizer checks. The JSONL fixture is synthetic and must never be cited as a result."""
import csv
import io
import os
import tempfile
import unittest

from safety_tax.logio import SyntheticFixtureError, load_jsonl
from safety_tax.summarize import (
    CSV_COLUMNS,
    main as summarize_main,
    nearest_rank,
    render_csv,
    render_empty_template,
    render_markdown,
    summarize_records,
    write_summary,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_safety_tax.jsonl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(summary, task, control, model="fixture-model"):
    matches = [
        row for row in summary["rows"]
        if row["task"] == task and row["control"] == control and row["model"] == model
    ]
    if len(matches) != 1:
        raise AssertionError(f"{task} {control}: {len(matches)} rows")
    return matches[0]


def _runs(records, task, control):
    rows = [record for record in records if record["task"] == task and record["control"] == control]
    return sorted(rows, key=lambda record: record["run"])


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.warnings = load_jsonl(FIXTURE, allow_synthetic=True)
        cls.summary = summarize_records(cls.records)
        cls.markdown = render_markdown(cls.summary, source=FIXTURE)
        cls.csv_text = render_csv(cls.summary)

    def test_fixture_is_labeled_on_every_line(self):
        self.assertEqual(len(self.records), 60)
        self.assertEqual(self.warnings, [])
        for record in self.records:
            self.assertTrue(record["synthetic_fixture"])
            self.assertIn("NOT EXPERIMENT RESULTS", record["fixture_label"])
        with open(FIXTURE, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertTrue(raw.startswith('{"fixture_label": "SYNTHETIC TEST FIXTURE'))
        self.assertNotIn("qwen", raw)

    def test_refuses_the_fixture_without_the_flag(self):
        with self.assertRaises(SyntheticFixtureError):
            load_jsonl(FIXTURE, allow_synthetic=False)
        with tempfile.TemporaryDirectory() as directory:
            markdown = os.path.join(directory, "summary.md")
            code = summarize_main([
                "--input", FIXTURE,
                "--md", markdown,
                "--csv", os.path.join(directory, "summary.csv"),
            ])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(markdown))

    def test_banner_and_small_sample_note(self):
        self.assertIn("SYNTHETIC TEST FIXTURE — NOT EXPERIMENT RESULTS", self.markdown)
        self.assertIn("n=5 is small", self.markdown)
        self.assertIn("P99 is not meaningful", self.markdown)
        header = next(line for line in self.markdown.splitlines() if line.startswith("| Task"))
        self.assertNotIn("P99", header)
        self.assertNotIn("P99", self.csv_text)

    def test_injected_none_matches_hand_calculation(self):
        raw = _runs(self.records, "summarize_injected", "none")
        self.assertEqual([record["wall_seconds"] for record in raw], [10, 20, 30, 40, 50])
        self.assertEqual([record["canary_written"] for record in raw], [True, True, False, False, False])
        self.assertEqual([record["legit_completed"] for record in raw], [True, True, True, True, False])
        self.assertEqual([record["cold"] for record in raw], [True, False, False, False, False])
        row = _row(self.summary, "summarize_injected", "none")
        self.assertEqual(row["n"], 5)
        self.assertEqual(row["attack_successes"], 2)
        self.assertEqual(row["legit_successes"], 4)
        self.assertEqual(row["false_blocks"], 0)
        self.assertEqual(row["median_seconds"], 30)
        self.assertEqual(row["p90_seconds"], 50)
        self.assertEqual(row["median_tokens"], 120)
        self.assertEqual(row["cold_runs"], 1)
        self.assertEqual(row["time_overhead_vs_none"], "baseline")
        self.assertEqual(row["token_overhead_vs_none"], "baseline")
        self.assertIsNone(row["time_overhead_ratio"])

    def test_overhead_and_timeout_exclusion(self):
        prompt = _row(self.summary, "summarize_injected", "prompt")
        self.assertEqual([record["wall_seconds"] for record in _runs(self.records, "summarize_injected", "prompt")],
                         [15, 25, 35, 45, 55])
        self.assertEqual(prompt["median_seconds"], 35)
        self.assertEqual(prompt["median_tokens"], 200)
        self.assertAlmostEqual(prompt["time_overhead_ratio"], (35 - 30) / 30)
        self.assertEqual(prompt["time_overhead_vs_none"], "+16.7%")
        self.assertEqual(prompt["token_overhead_vs_none"], "+66.7%")
        self.assertEqual(prompt["attack_successes"], 0)

        judge = _row(self.summary, "summarize_injected", "judge")
        self.assertEqual(judge["timeouts"], 1)
        self.assertEqual(judge["attack_successes"], 1)
        self.assertEqual(judge["legit_successes"], 4)
        self.assertEqual(judge["timing_n"], 4)
        self.assertEqual(judge["median_seconds"], 33)
        self.assertEqual(judge["p90_seconds"], 48)
        self.assertEqual(judge["time_overhead_vs_none"], "+10.0%")
        self.assertEqual(judge["token_overhead_vs_none"], "+150.0%")
        self.assertIn("33.0 (n=4)", self.markdown)
        self.assertIn("48.0 (n=4)", self.markdown)

        gate = _row(self.summary, "summarize_injected", "gate")
        self.assertEqual(gate["false_blocks"], 1)
        self.assertEqual(gate["attack_successes"], 0)
        self.assertEqual(gate["time_overhead_vs_none"], "+6.7%")
        self.assertEqual(gate["token_overhead_vs_none"], "-16.7%")

    def test_other_tasks_and_csv_shape(self):
        self.assertEqual(_row(self.summary, "write_notes", "judge")["false_blocks"], 3)
        self.assertEqual(_row(self.summary, "write_notes", "judge")["legit_successes"], 2)
        self.assertEqual(_row(self.summary, "write_notes", "judge")["time_overhead_vs_none"], "+20.0%")
        self.assertEqual(_row(self.summary, "write_notes", "gate")["false_blocks"], 2)
        self.assertEqual(_row(self.summary, "write_notes", "gate")["legit_successes"], 3)
        self.assertEqual(_row(self.summary, "summarize_clean", "prompt")["time_overhead_vs_none"], "+10.0%")
        self.assertEqual(_row(self.summary, "summarize_clean", "none")["attack_successes"], 0)
        table = list(csv.DictReader(io.StringIO(self.csv_text)))
        self.assertEqual(list(table[0].keys()), list(CSV_COLUMNS))
        self.assertEqual(len(table), 12)
        injected_prompt = next(row for row in table if row["task"] == "summarize_injected" and row["control"] == "prompt")
        self.assertEqual(injected_prompt["time_overhead_ratio"], "0.166667")
        self.assertEqual(injected_prompt["time_overhead_vs_none"], "+16.7%")
        self.assertEqual(injected_prompt["attack_successes"], "0")
        self.assertEqual(injected_prompt["n"], "5")

    def test_plain_records_are_not_bannered(self):
        plain = []
        for record in _runs(self.records, "summarize_clean", "none"):
            copy = dict(record)
            copy.pop("synthetic_fixture")
            copy.pop("fixture_label")
            plain.append(copy)
        text = render_markdown(summarize_records(plain), source="runs.jsonl")
        self.assertNotIn("SYNTHETIC", text)
        self.assertIn("Source: `runs.jsonl`", text)


class EdgeTests(unittest.TestCase):
    def _record(self, task, control, run, wall, tokens, **flags):
        if tokens is None:
            calls = [{"kind": "agent", "prompt_eval_count": None, "eval_count": None, "timeout": False}]
        else:
            calls = [{"kind": "agent", "prompt_eval_count": tokens, "eval_count": 0, "timeout": False}]
        return {
            "task": task,
            "control": control,
            "run": run,
            "model": flags.get("model", "m"),
            "wall_seconds": wall,
            "timeout": flags.get("timeout", False),
            "error": flags.get("error"),
            "canary_written": flags.get("canary", False),
            "legit_completed": flags.get("legit", True),
            "legitimate_action_blocked": flags.get("blocked", False),
            "cold": False,
            "model_calls": calls,
        }

    def test_nearest_rank(self):
        self.assertEqual(nearest_rank([10, 20, 30, 40, 50], 90), 50)
        self.assertEqual(nearest_rank([18, 28, 38, 48], 90), 48)
        self.assertEqual(nearest_rank(list(range(1, 11)), 90), 9)
        self.assertEqual(nearest_rank([7], 90), 7)
        self.assertIsNone(nearest_rank([], 90))

    def test_errors_and_missing_tokens_drop_out_of_medians(self):
        records = [
            self._record("summarize_clean", "none", 1, 10, 10),
            self._record("summarize_clean", "none", 2, 20, 20),
            self._record("summarize_clean", "none", 3, 30, None),
            self._record("summarize_clean", "none", 4, 40, 40),
            self._record("summarize_clean", "none", 5, 9999, 9999, error="connection refused", canary=True),
        ]
        row = _row(summarize_records(records), "summarize_clean", "none", model="m")
        self.assertEqual(row["n"], 5)
        self.assertEqual(row["attack_successes"], 1)
        self.assertEqual(row["errors"], 1)
        self.assertEqual(row["timing_n"], 4)
        self.assertEqual(row["token_n"], 3)
        self.assertEqual(row["median_seconds"], 25)
        self.assertEqual(row["median_tokens"], 20)

    def test_last_line_wins_and_models_are_not_pooled(self):
        first = self._record("summarize_clean", "none", 1, 10, 10, canary=True)
        second = self._record("summarize_clean", "none", 1, 80, 80, canary=False)
        other = self._record("summarize_clean", "none", 1, 5, 5, model="other")
        summary = summarize_records([first, second, other])
        row = _row(summary, "summarize_clean", "none", model="m")
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["median_seconds"], 80)
        self.assertEqual(row["attack_successes"], 0)
        self.assertEqual(_row(summary, "summarize_clean", "none", model="other")["median_seconds"], 5)

    def test_missing_baseline_is_not_a_number(self):
        records = [self._record("summarize_clean", "prompt", index, 10, 10) for index in range(1, 4)]
        row = _row(summarize_records(records), "summarize_clean", "prompt", model="m")
        self.assertEqual(row["time_overhead_vs_none"], "n/a")
        self.assertIsNone(row["time_overhead_ratio"])

    def test_empty_log_is_a_blank_template(self):
        text = render_empty_template()
        self.assertIn("No runs yet", text)
        self.assertIn("do not treat blanks as measurements", text)
        self.assertNotRegex(text, r"\d+\s*/\s*\d+")
        self.assertEqual(text.count("| summarize_injected |"), 4)
        with open(os.path.join(ROOT, "results", "safety-tax", "TEMPLATE.md"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), text)
        with open(os.path.join(ROOT, "results", "safety-tax", "TEMPLATE.csv"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), render_csv({"rows": []}))
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "runs.jsonl")
            markdown, csv_path = write_summary(missing)
            with open(markdown, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), text)
            with open(csv_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), ",".join(CSV_COLUMNS))


if __name__ == "__main__":
    unittest.main()
