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


def _probe_view(message, timing):
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
        "prompt_eval_count": timing.get("prompt_eval_count"),
        "missing_token_counts": missing,
    }


def describe_preflight(message, timing, repeat=None):
    """Token and thinking fields from the probe.

    Pass `repeat=(message, timing)` for the second identical call. A cache
    hit often omits prompt_eval_count. That is recorded and is not a failed
    probe: prompt-token sums skip the missing field, and eval coverage keeps
    the call when eval_count is present.
    """
    first = _probe_view(message, timing)
    probes = [first]
    cached_missing = False
    if repeat is not None:
        second_message, second_timing = repeat
        second = _probe_view(second_message, second_timing)
        probes.append(second)
        first_prompt = first.get("prompt_eval_count")
        second_prompt = second.get("prompt_eval_count")
        cached_missing = first_prompt is not None and second_prompt is None
    return {
        "thinking_field_present": any(probe["thinking_field_present"] for probe in probes),
        "eval_count": first.get("eval_count"),
        "load_duration": first.get("load_duration"),
        "missing_token_counts": list(first["missing_token_counts"]),
        "probes": probes,
        "cached_prompt_eval_missing": cached_missing,
    }


def _field_text(value):
    return "missing" if value is None else str(value)


def format_preflight(report):
    missing = report.get("missing_token_counts") or []
    missing_text = "none" if not missing else ",".join(missing)
    lines = [
        f"preflight thinking_field_present={'yes' if report.get('thinking_field_present') else 'no'}",
        f"preflight eval_count={_field_text(report.get('eval_count'))}",
        f"preflight load_duration={_field_text(report.get('load_duration'))}",
        f"preflight missing_token_counts={missing_text}",
    ]
    probes = report.get("probes") or []
    if len(probes) >= 2:
        second = probes[1]
        second_missing = second.get("missing_token_counts") or []
        second_missing_text = "none" if not second_missing else ",".join(second_missing)
        lines.append(f"preflight repeat_eval_count={_field_text(second.get('eval_count'))}")
        lines.append(f"preflight repeat_load_duration={_field_text(second.get('load_duration'))}")
        lines.append(f"preflight repeat_missing_token_counts={second_missing_text}")
    if report.get("cached_prompt_eval_missing"):
        lines.append(
            "preflight cached_prompt=prompt_eval_count missing on repeat; "
            "prompt-token sums skip that field and the call stays in eval coverage when eval_count is present"
        )
    return "\n".join(lines)


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

    def model_identity(self, model):
        """Ollama model digest and server version. Failures are unknown, not fatal.

        The digest is part of the config hash. The version is stored on each
        row and is not hashed. A short connect timeout keeps a down server
        from stalling startup.
        """
        connect = min(float(self.connect_timeout), 2.0)
        timeout = (connect, 5.0)
        digest = None
        version = None
        try:
            response = requests.get(f"{self.host}/api/version", timeout=timeout)
            data = response.json()
            if isinstance(data, dict):
                version = data.get("version") or None
        except Exception:
            version = None
        try:
            response = requests.post(
                f"{self.host}/api/show", json={"model": model}, timeout=timeout,
            )
            data = response.json()
            if isinstance(data, dict):
                digest = data.get("digest") or None
        except Exception:
            digest = None
        return digest, version


def lookup_model_identity(client, model):
    """Call model_identity when the client has it. Fake clients return unknown."""
    lookup = getattr(client, "model_identity", None)
    if not callable(lookup):
        return None, None
    try:
        found = lookup(model)
    except Exception:
        return None, None
    if not isinstance(found, (tuple, list)) or len(found) != 2:
        return None, None
    digest, version = found
    return digest or None, version or None
