"""The write-up must not contain measured Safety Tax rates. Blanks are not results."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE = re.compile(r"\d+\s*/\s*\d+")


class DocTests(unittest.TestCase):
    def test_results_section_is_empty_of_rates(self):
        with open(os.path.join(ROOT, "docs", "safety-tax.md"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("python -m safety_tax", text)
        self.assertIn("qwen3:4b", text)
        self.assertIn("12 GB", text)
        self.assertIn("n=5", text)
        self.assertIn("not a production", text.lower())
        results = text.split("## Results", 1)[1]
        self.assertIn("Not run", results)
        self.assertIsNone(RATE.search(results), RATE.search(results).group(0) if RATE.search(results) else "")
        self.assertNotIn("+16.7%", text)
        self.assertNotIn("33.0", text)
        self.assertIn("## Pre-registration", text)
        self.assertIn("results/safety-tax/PREREGISTRATION.md", text)
        prereg = text.split("## Pre-registration", 1)[1].split("## ", 1)[0]
        self.assertNotIn("Hypothesis 1", prereg)
        self.assertIn("## Pilot", text)
        self.assertIn("--runs 5", text)
        self.assertIn("--runs 15", text)
        self.assertIn("## Claims you can and cannot make", text)
        self.assertIn("ESTIMATE", text)
        self.assertNotIn("same as `agent.py`", text)
        with open(os.path.join(ROOT, "results", "safety-tax", "PREREGISTRATION.md"), encoding="utf-8") as handle:
            form = handle.read()
        self.assertIn("Hypothesis 1", form)
        self.assertIsNone(RATE.search(form))

    def test_readme_points_at_the_experiment(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        section = readme.split("## Safety Tax", 1)[1].split("## ", 1)[0]
        self.assertIn("docs/safety-tax.md", section)
        self.assertIn("python -m safety_tax", section)
        self.assertIsNone(RATE.search(section))


if __name__ == "__main__":
    unittest.main()
