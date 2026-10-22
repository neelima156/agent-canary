"""Ollama client with requests mocked. These tests do not open a socket."""
import unittest
from unittest import mock

import requests

from safety_tax.client import CallTimeout, OllamaClient, OllamaError, lookup_model_identity


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
        self.assertFalse(captured["json"]["think"])
        self.assertEqual(captured["json"]["keep_alive"], "30m")
        self.assertEqual(captured["json"]["options"]["seed"], 0)
        self.assertNotIn("num_predict", captured["json"]["options"])

        def fake_post_tools(url, json, timeout):
            captured["tools"] = json.get("tools")
            return _response(200, {"message": {"role": "assistant", "content": ""}})

        with mock.patch("safety_tax.client.requests.post", fake_post_tools):
            client.chat("qwen3:4b", [], tools=[{"type": "function"}], timeout=30)
        self.assertEqual(captured["tools"], [{"type": "function"}])

    def test_connect_timeout_is_an_error_not_a_permanent_timeout(self):
        def fake_post(url, json, timeout):
            raise requests.ConnectTimeout("connect timed out")

        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            with self.assertRaises(OllamaError) as caught:
                client.chat("qwen3:4b", [], timeout=30)
        self.assertIn("connect timeout", str(caught.exception))

    def test_connection_error_is_an_ollama_error(self):
        def fake_post(url, json, timeout):
            raise requests.ConnectionError("connection refused")

        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            with self.assertRaises(OllamaError):
                client.chat("qwen3:4b", [], timeout=30)

    def test_read_timeout_is_still_a_call_timeout(self):
        def fake_post(url, json, timeout):
            raise requests.ReadTimeout("read timed out")

        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            with self.assertRaises(CallTimeout):
                client.chat("qwen3:4b", [], timeout=30)

    def test_judge_num_predict_is_sent_when_set(self):
        captured = {}

        def fake_post(url, json, timeout):
            captured["options"] = json["options"]
            return _response(200, {"message": {"role": "assistant", "content": "DENY"}})

        client = OllamaClient("http://127.0.0.1:11434")
        with mock.patch("safety_tax.client.requests.post", fake_post):
            client.chat("qwen3:4b", [], timeout=30, num_predict=32, seed=7, think=False)
        self.assertEqual(captured["options"]["num_predict"], 32)
        self.assertEqual(captured["options"]["seed"], 7)

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

    def test_model_identity_reads_digest_and_version(self):
        def fake_get(url, timeout):
            self.assertIn("/api/version", url)
            return _response(200, {"version": "0.9.0"})

        def fake_post(url, json, timeout):
            self.assertIn("/api/show", url)
            self.assertEqual(json["model"], "qwen3:4b")
            return _response(200, {"digest": "sha256:abc"})

        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.get", fake_get), \
                mock.patch("safety_tax.client.requests.post", fake_post):
            self.assertEqual(client.model_identity("qwen3:4b"), ("sha256:abc", "0.9.0"))

    def test_model_identity_swallows_transport_errors(self):
        client = OllamaClient("http://127.0.0.1:9")
        with mock.patch("safety_tax.client.requests.get", side_effect=requests.ConnectionError("nope")), \
                mock.patch("safety_tax.client.requests.post", side_effect=requests.ConnectionError("nope")):
            self.assertEqual(client.model_identity("qwen3:4b"), (None, None))
        self.assertEqual(lookup_model_identity(object(), "qwen3:4b"), (None, None))


if __name__ == "__main__":
    unittest.main()
