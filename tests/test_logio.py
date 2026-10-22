"""JSONL resume. A partial last line must not corrupt the next record."""
import os
import tempfile
import unittest

from safety_tax.logio import append_jsonl, load_jsonl


class AppendTests(unittest.TestCase):
    def test_append_after_a_partial_line_keeps_the_next_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.jsonl")
            with open(path, "wb") as handle:
                handle.write(b'{"task": "partial"')
            append_jsonl(path, {
                "task": "summarize_clean",
                "control": "none",
                "run": 1,
                "model": "m",
            })
            records, warnings = load_jsonl(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task"], "summarize_clean")
            self.assertEqual(records[0]["run"], 1)
            self.assertTrue(any("invalid JSON" in warning for warning in warnings))
            with open(path, "rb") as handle:
                raw = handle.read()
            self.assertTrue(raw.startswith(b'{"task": "partial"\n{'))
            self.assertNotIn(b'{"task": "partial"{"', raw)
            self.assertTrue(raw.endswith(b"\n"))

    def test_append_to_a_terminated_file_does_not_invent_a_blank_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "runs.jsonl")
            append_jsonl(path, {"task": "summarize_clean", "control": "none", "run": 1, "model": "m"})
            append_jsonl(path, {"task": "summarize_clean", "control": "none", "run": 2, "model": "m"})
            records, warnings = load_jsonl(path)
            self.assertEqual(warnings, [])
            self.assertEqual([record["run"] for record in records], [1, 2])


if __name__ == "__main__":
    unittest.main()
