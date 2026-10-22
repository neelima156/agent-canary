"""Shared helpers for unit tests. Not experiment data."""
import copy

from safety_tax.client import CallTimeout
from safety_tax.config import TIMING_FIELDS

SUMMARY = (
    "The page describes five habits for a calm morning: water, a few minutes of stillness, "
    "a short walk, a protein breakfast, and one written intention. Small repeatable habits "
    "beat plans that are abandoned by Wednesday."
)


def timing(**overrides):
    record = {field: 0 for field in TIMING_FIELDS}
    record["wall_seconds"] = 0.1
    record["timeout"] = False
    record.update(overrides)
    return record


def tool_call(name, args):
    return {"function": {"name": name, "arguments": args}}


def assistant(content="", calls=None, **timing_overrides):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"message": message, "timing": timing(**timing_overrides)}


class FakeClient:
    """Scripted Ollama stand-in. 'timeout' or an exception instance is raised instead of a reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, messages, tools=None, temperature=0.7, timeout=300, num_ctx=4096,
             seed=None, keep_alive=None, think=None, num_predict=None):
        self.calls.append({
            "model": model,
            "messages": copy.deepcopy(messages),
            "tools": tools,
            "temperature": temperature,
            "timeout": timeout,
            "num_ctx": num_ctx,
            "seed": seed,
            "keep_alive": keep_alive,
            "think": think,
            "num_predict": num_predict,
        })
        if not self.replies:
            raise AssertionError("no scripted replies left")
        item = self.replies.pop(0)
        if item == "timeout":
            raise CallTimeout(0.05)
        if isinstance(item, Exception):
            raise item
        return item["message"], dict(item["timing"])
