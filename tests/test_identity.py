"""The resume hash changes when the experiment changes."""
import hashlib
import unittest
from unittest import mock

from safety_tax.identity import fingerprint, fingerprint_payload
from safety_tax.runner import parse_args
from safety_tax.tasks import get_task


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
        self.assertNotIn("git_sha", payload)
        self.assertNotIn("runs", payload)
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

    def test_hash_ignores_git_sha_and_run_count(self):
        args = parse_args(["--runs", "5"])
        with mock.patch("safety_tax.identity.git_sha", return_value="abc"):
            first = fingerprint(args)[0]
        with mock.patch("safety_tax.identity.git_sha", return_value="def"):
            second = fingerprint(args)[0]
        self.assertEqual(first, second)
        longer = parse_args(["--runs", "15"])
        self.assertEqual(first, fingerprint(longer)[0])

    def test_hash_covers_document_bytes_tool_schemas_and_model_digest(self):
        args = parse_args([])
        payload = fingerprint_payload(args)
        self.assertIn("model_digest", payload)
        self.assertIsNone(payload["model_digest"])
        self.assertNotIn("ollama_version", payload)
        self.assertNotIn("git_sha", payload)
        names = {tool["function"]["name"] for tool in payload["tool_schemas"]}
        self.assertEqual(names, {"read_file", "write_file"})
        page = payload["tasks"]["summarize_injected"]["document_sha256"]
        with open(get_task("summarize_injected").source_path(), "rb") as handle:
            raw = handle.read()
        self.assertEqual(page, hashlib.sha256(raw).hexdigest())
        self.assertEqual(fingerprint(args)[0], fingerprint(args, model_digest=None)[0])
        self.assertNotEqual(fingerprint(args)[0], fingerprint(args, model_digest="sha256:abc")[0])
        original = fingerprint(args)[0]
        with mock.patch("safety_tax.identity._sha_file", return_value="ab" * 32):
            self.assertNotEqual(original, fingerprint(args)[0])


if __name__ == "__main__":
    unittest.main()
