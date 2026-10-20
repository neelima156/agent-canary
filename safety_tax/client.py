"""Ollama /api/chat client.

ConnectTimeout is a subclass of Timeout. It must be caught before Timeout,
or an unreachable host is stored as a permanent timeout row and --retry-errors
skips it. A refused connection is an OllamaError. A read timeout is a CallTimeout.
"""
import time

import requests

from safety_tax.config import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_CTX,
    DEFAULT_SEED,
    TIMING_FIELDS,
)


class CallTimeout(Exception):
    def __init__(self, wall_seconds):
        super().__init__("ollama call timed out")
        self.wall_seconds = wall_seconds


class OllamaError(Exception):
    pass


def empty_timing(kind, wall_seconds, timeout):
    rec = {field: None for field in TIMING_FIELDS}
    rec["kind"] = kind
    rec["wall_seconds"] = round(wall_seconds or 0.0, 3)
    rec["timeout"] = timeout
    return rec


def describe_preflight(message, timing):
    """What one probe call tells us about thinking and token accounting."""
    message = message or {}
    timing = timing or {}
    content = message.get("content") or ""
    thinking = message.get("thinking")
    thinking_present = bool(thinking) or ("<think>" in str(content).lower())
    missing = []
    for field in ("prompt_eval_count", "eval_count"):
        if timing.get(field) is None:
            missing.append(field)
    return {
        "thinking_field_present": thinking_present,
        "eval_count": timing.get("eval_count"),
        "load_duration": timing.get("load_duration"),
        "missing_token_counts": missing,
    }


def format_preflight(report):
    missing = report.get("missing_token_counts") or []
    missing_text = "none" if not missing else ",".join(missing)
    eval_count = report.get("eval_count")
    eval_text = "missing" if eval_count is None else str(eval_count)
    load = report.get("load_duration")
    load_text = "missing" if load is None else str(load)
    thinking = "yes" if report.get("thinking_field_present") else "no"
    return "\n".join((
        f"preflight thinking_field_present={thinking}",
        f"preflight eval_count={eval_text}",
        f"preflight load_duration={load_text}",
        f"preflight missing_token_counts={missing_text}",
    ))


class OllamaClient:
    def __init__(self, host, connect_timeout=DEFAULT_CONNECT_TIMEOUT_S):
        self.host = host.rstrip("/")
        self.connect_timeout = connect_timeout

    def http_timeout(self, call_timeout):
        connect = min(self.connect_timeout, float(call_timeout))
        return (connect, float(call_timeout))

    def chat(self, model, messages, tools=None, temperature=0.7, timeout=300,
             num_ctx=DEFAULT_NUM_CTX, seed=DEFAULT_SEED, keep_alive=DEFAULT_KEEP_ALIVE,
             think=False, num_predict=None):
        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "seed": seed,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": bool(think),
            "keep_alive": keep_alive,
            "options": options,
        }
        if tools is not None:
            body["tools"] = tools
        url = f"{self.host}/api/chat"
        started = time.perf_counter()
        try:
            response = requests.post(url, json=body, timeout=self.http_timeout(timeout))
        except requests.ConnectTimeout as exc:
            raise OllamaError(f"connect timeout: {exc}") from exc
        except requests.ConnectionError as exc:
            raise OllamaError(f"connection error: {exc}") from exc
        except requests.Timeout:
            raise CallTimeout(time.perf_counter() - started) from None
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc
        wall = time.perf_counter() - started
        try:
            data = response.json()
        except ValueError:
            raise OllamaError(f"invalid JSON from Ollama (HTTP {response.status_code})") from None
        if not isinstance(data, dict):
            raise OllamaError(f"unexpected Ollama payload (HTTP {response.status_code})")
        if response.status_code >= 400 or ("error" in data and "message" not in data):
            raise OllamaError(str(data.get("error") or response.text))
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response missing message")
        timing = {field: data.get(field) for field in TIMING_FIELDS}
        timing["wall_seconds"] = round(wall, 3)
        timing["timeout"] = False
        return message, timing
