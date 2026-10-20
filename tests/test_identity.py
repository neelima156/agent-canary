"""The resume hash changes when the experiment changes."""
import unittest
from unittest import mock

from safety_tax.identity import fingerprint, fingerprint_payload
from safety_tax.runner import parse_args


class IdentityTests(unittest.TestCase):
    def test_hash_is_stable_and_covers_prompts_and_seed(self):
        args = parse_args(["--runs", "1", "--tasks", "summarize_clean", "--controls", "none"])
        first, payload = fingerprint(args)
        second, _again = fingerprint(args)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertIn("judge_system", payload)
        self.assertIn("tasks", payload)
        self.assertIn("seed", payload)
        self.assertIn("temperature", payload)
        self.assertIn("num_ctx", payload)
        self.assertIn("git_sha", payload)
        args.temperature = args.temperature + 0.1
        changed, _payload = fingerprint(args)
        self.assertNotEqual(first, changed)
        args.temperature = args.temperature - 0.1
        args.seed = args.seed + 1
        self.assertNotEqual(first, fingerprint(args)[0])

    def test_hash_changes_when_the_judge_prompt_changes(self):
        args = parse_args([])
        original = fingerprint(args)[0]
        with mock.patch("safety_tax.identity.JUDGE_SYSTEM", "reply maybe"):
            self.assertNotEqual(original, fingerprint(args)[0])

    def test_payload_lists_expected_actions_separately_from_the_allowlist(self):
        payload = fingerprint_payload(parse_args([]))
        injected = payload["tasks"]["summarize_injected"]
        self.assertEqual(injected["write_allowlist"], [])
        self.assertEqual(injected["expected_actions"], [{"path": "test-page.html", "tool": "read_file"}])
        riding = payload["tasks"]["write_notes_injected"]
        self.assertIn("NOTES.txt", riding["user_text"])
        self.assertIn("write-notes-injected.html", riding["source_relative"])
        self.assertEqual(riding["write_allowlist"], ["NOTES.txt"])


if __name__ == "__main__":
    unittest.main()
