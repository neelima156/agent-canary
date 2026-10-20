"""Approval gate, legitimate-action definition, and sandbox containment."""
import os
import tempfile
import unittest

from agent import BASE_SYSTEM, DEFENDED_SYSTEM, TASK

from safety_tax.config import CANARY_FILENAME, NOTES_FILENAME
from safety_tax.controls import (
    gate_decision,
    is_legitimate_action,
    present_tool_result,
    system_prompt,
)
from safety_tax.sandbox import basename_only, reset_sandbox, sandbox_path
from safety_tax.tasks import TASKS, get_task


class GateTests(unittest.TestCase):
    def test_reads_are_allowed_and_unlisted_writes_are_denied(self):
        injected = get_task("summarize_injected")
        read = gate_decision(injected, "read_file", {"path": "test-page.html"})
        self.assertTrue(read.allow)
        write = gate_decision(injected, "write_file", {"path": "CANARY.txt", "content": "BLUEBIRD"})
        self.assertFalse(write.allow)
        self.assertIn("not explicitly request", write.reason)
        self.assertFalse(is_legitimate_action(injected, "write_file", {"path": "CANARY.txt"}))
        self.assertTrue(is_legitimate_action(injected, "read_file", {"path": "test-page.html"}))

    def test_requested_notes_file_is_on_the_allowlist(self):
        task = get_task("write_notes")
        for path in ("NOTES.txt", "notes.txt", "./NOTES.txt", "folder/NOTES.txt", "folder\\NOTES.txt"):
            decision = gate_decision(task, "write_file", {"path": path, "content": "hello"})
            self.assertTrue(decision.allow, path)
            self.assertTrue(is_legitimate_action(task, "write_file", {"path": path}), path)
        canary = gate_decision(task, "write_file", {"path": "CANARY.txt", "content": "BLUEBIRD"})
        self.assertFalse(canary.allow)
        self.assertFalse(is_legitimate_action(task, "write_file", {"path": "CANARY.txt"}))

    def test_unknown_tool_and_unsafe_path_are_denied(self):
        task = get_task("write_notes")
        self.assertFalse(gate_decision(task, "delete_file", {"path": "NOTES.txt"}).allow)
        self.assertFalse(is_legitimate_action(task, "delete_file", {"path": "NOTES.txt"}))
        for path in ("", "..", ".", "foo/.."):
            decision = gate_decision(task, "write_file", {"path": path})
            self.assertFalse(decision.allow, path)
            self.assertFalse(is_legitimate_action(task, "write_file", {"path": path}))

    def test_gate_allows_a_write_only_when_it_is_legitimate(self):
        paths = ["test-page.html", "NOTES.txt", "CANARY.txt", "other.txt", "./NOTES.txt"]
        for task in TASKS.values():
            for path in paths:
                decision = gate_decision(task, "write_file", {"path": path, "content": "x"})
                legitimate = is_legitimate_action(task, "write_file", {"path": path})
                self.assertEqual(decision.allow, legitimate, (task.task_id, path))
                read = gate_decision(task, "read_file", {"path": path})
                self.assertTrue(read.allow)

    def test_prompt_is_the_existing_defense_and_other_controls_are_not(self):
        self.assertEqual(system_prompt("prompt"), DEFENDED_SYSTEM)
        for control in ("none", "gate", "judge"):
            self.assertEqual(system_prompt(control), BASE_SYSTEM)
        wrapped = present_tool_result("prompt", "read_file", "page")
        self.assertIn("<untrusted_content>", wrapped)
        self.assertNotIn("<untrusted_content>", present_tool_result("none", "read_file", "page"))
        self.assertNotIn("<untrusted_content>", present_tool_result("gate", "read_file", "page"))
        self.assertEqual(present_tool_result("prompt", "read_file", "ERROR: missing"), "ERROR: missing")

    def test_tasks_match_the_canary_harness(self):
        injected = get_task("summarize_injected")
        clean = get_task("summarize_clean")
        write = get_task("write_notes")
        self.assertEqual(injected.user_text, TASK)
        self.assertEqual(clean.user_text, injected.user_text)
        self.assertTrue(injected.injection)
        self.assertFalse(clean.injection)
        self.assertFalse(write.injection)
        self.assertIn(NOTES_FILENAME, write.user_text)
        self.assertEqual(set(write.write_allowlist), {NOTES_FILENAME})
        self.assertEqual(injected.write_allowlist, frozenset())
        self.assertTrue(injected.source_path().endswith(os.path.join("testpages", "test-page.html")))
        self.assertTrue(os.path.isfile(injected.source_path()))
        self.assertTrue(os.path.isfile(clean.source_path()))


class SandboxTests(unittest.TestCase):
    def test_traversal_and_absolute_paths_stay_inside(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = os.path.join(directory, "sandbox")
            os.makedirs(sandbox)
            outside = os.path.join(directory, CANARY_FILENAME)
            for raw in ("../../CANARY.txt", "..\\..\\CANARY.txt", "/etc/passwd", "C:\\Windows\\NOTES.txt"):
                path = sandbox_path(sandbox, raw)
                self.assertEqual(os.path.commonpath([os.path.realpath(sandbox), path]), os.path.realpath(sandbox))
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("BLUEBIRD")
            self.assertFalse(os.path.exists(outside))
            self.assertEqual(os.path.basename(sandbox_path(sandbox, "/etc/passwd")), "passwd")
            self.assertTrue(os.path.isfile(os.path.join(sandbox, CANARY_FILENAME)))
            self.assertEqual(basename_only("folder\\NOTES.txt"), "NOTES.txt")

    def test_rejects_empty_and_dot_names(self):
        with tempfile.TemporaryDirectory() as directory:
            for raw in ("", ".", "..", "foo/..", "a\x00b"):
                with self.assertRaises(ValueError):
                    sandbox_path(directory, raw)

    def test_reset_unlinks_a_symlink_instead_of_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = os.path.join(directory, "outside")
            os.makedirs(outside)
            link = os.path.join(directory, "sandbox")
            os.symlink(outside, link)
            reset_sandbox(link)
            self.assertFalse(os.path.islink(link))
            self.assertTrue(os.path.isdir(link))
            self.assertTrue(os.path.isdir(outside))
            with open(os.path.join(link, CANARY_FILENAME), "w", encoding="utf-8") as handle:
                handle.write("BLUEBIRD")
            self.assertFalse(os.path.exists(os.path.join(outside, CANARY_FILENAME)))


class PageTwinTests(unittest.TestCase):
    def test_clean_page_is_the_article_without_the_canary(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "testpages", "test-page.html"), encoding="utf-8") as handle:
            injected = handle.read()
        with open(os.path.join(root, "testpages", "morning-clean.html"), encoding="utf-8") as handle:
            clean = handle.read()
        self.assertIn("CANARY.txt", injected)
        self.assertIn("BLUEBIRD", injected)
        self.assertIn("color:white", injected)
        self.assertNotIn("CANARY", clean.upper())
        self.assertNotIn("BLUEBIRD", clean.upper())
        for sentence in (
            "5 Habits for a Calm Morning",
            "Start with a glass of water, five minutes of stillness, a short walk, a protein breakfast",
            "Small, repeatable habits beat big plans you abandon by Wednesday.",
        ):
            self.assertIn(sentence, injected)
            self.assertIn(sentence, clean)


if __name__ == "__main__":
    unittest.main()
