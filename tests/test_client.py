"""Ollama client with requests mocked. These tests do not open a socket."""
import unittest
from unittest import mock

import requests

from safety_tax.client import CallTimeout, OllamaClient, OllamaError


def _response(status, payload):
    class Response:
        status_code = status
        text = "nope"

        def json(self):
            return payload
    return Response()


class ClientTests(unittest.TestCase):
    def test_http_timeout_caps_connect_and_keeps_the_read_budget(self):
        client = OllamaClient("http://127.0.0.1:11434", connect_timeout=10)
        self.assertEqual(client.http_timeout(30), (10, 30))
        self.assertEqual(client.http_timeout(4), (4, 4))

    def test_chat_posts_timing_fields_and_optional_tools(self):
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _response(200, {
                "message": {"role": "assistant", "content": "hi"},
                "prompt_eval_count": 4,
                "eval_count": 3,
                "prompt_eval_duration": 20,
                "eval_duration": 30,
                "total_duration": 50,
                "load_duration": 7,
            })

        client = OllamaClient("http://127.0.0.1:11434/")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            message, timing = client.chat("qwen3:4b", [{"role": "user", "content": "hi"}], tools=None, timeout=30)
        self.assertEqual(message["content"], "hi")
        self.assertEqual(timing["prompt_eval_count"], 4)
        self.assertEqual(timing["eval_count"], 3)
        self.assertEqual(timing["prompt_eval_duration"], 20)
        self.assertEqual(timing["eval_duration"], 30)
        self.assertEqual(timing["total_duration"], 50)
        self.assertEqual(timing["load_duration"], 7)
        self.assertFalse(timing["timeout"])
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["timeout"], (10, 30))
        self.assertNotIn("tools", captured["json"])
        self.assertFalse(captured["json"]["stream"])

        def fake_post_tools(url, json, timeout):
            captured["tools"] = json.get("tools")
            return _response(200, {"message": {"role": "assistant", "content": ""}})

        with mock.patch("safety_tax.client.requests.post", fake_post_tools):
            client.chat("qwen3:4b", [], tools=[{"type": "function"}], timeout=30)
        self.assertEqual(captured["tools"], [{"type": "function"}])

    def test_requests_timeout_becomes_call_timeout(self):
        def fake_post(url, json, timeout):
            raise requests.Timeout("hung")

        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            with self.assertRaises(CallTimeout) as caught:
                client.chat("qwen3:4b", [], timeout=5)
        self.assertGreaterEqual(caught.exception.wall_seconds, 0)

    def test_error_payload_becomes_ollama_error(self):
        def fake_post(url, json, timeout):
            return _response(404, {"error": "model not found"})

        client = OllamaClient("http://127.0.0.1:11434")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            with self.assertRaises(OllamaError) as caught:
                client.chat("missing", [], timeout=5)
        self.assertIn("model not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
