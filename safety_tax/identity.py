"""Config hash for resume identity and for the JSONL row.

The hash covers the prompts, the model parameters, and the task definitions.
It does not cover the git SHA or --runs. The SHA is still written on each
row. A later commit that does not change the fingerprinted settings resumes
the same cells, and a pilot of --runs 5 extends to --runs 15 on the same
hash. Two runs with different hashes are different experiments.
"""
import hashlib
import json
import os
import subprocess

from safety_tax import __version__
from safety_tax.config import (
    CANARY_FILENAME,
    CANARY_TOKEN,
    CONTROL_ORDER,
    SUMMARY_KEY_TERMS,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_KEY_TERMS,
)
from safety_tax.controls import DENIAL_TEMPLATE, JUDGE_SYSTEM, exposed_tool_names, system_prompt
from safety_tax.tasks import TASKS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SHA = None
_SHA_LOOKED = False


def git_sha():
    """HEAD of this repo, or None when git is unavailable. Cached per process."""
    global _SHA, _SHA_LOOKED
    if _SHA_LOOKED:
        return _SHA
    _SHA_LOOKED = True
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        text = out.decode().strip()
        _SHA = text or None
    except (OSError, subprocess.SubprocessError):
        _SHA = None
    return _SHA


def _task_payload(task):
    actions = sorted(
        ({"path": action.path, "tool": action.tool} for action in task.expected_actions),
        key=lambda item: (item["tool"], item["path"]),
    )
    return {
        "document_name": task.document_name,
        "expected_actions": actions,
        "forbid_other_writes": task.forbid_other_writes,
        "injection": task.injection,
        "source_relative": task.source_relative.replace("\\", "/"),
        "success": task.success,
        "task_id": task.task_id,
        "user_text": task.user_text,
        "write_allowlist": sorted(task.write_allowlist),
    }


def fingerprint_payload(args):
    """JSON-stable description of the experiment this process would run."""
    judge_model = args.judge_model or args.model
    tools = {}
    for task_id, task in sorted(TASKS.items()):
        tools[task_id] = {
            control: exposed_tool_names(control, task) for control in CONTROL_ORDER
        }
    return {
        "call_timeout_s": args.call_timeout,
        "canary_filename": CANARY_FILENAME,
        "canary_token": CANARY_TOKEN,
        "code_version": __version__,
        "cold_load_ns": args.cold_load_ns,
        "denial_template": DENIAL_TEMPLATE,
        "judge_model": judge_model,
        "judge_num_predict": args.judge_num_predict,
        "judge_system": JUDGE_SYSTEM,
        "judge_temperature": args.judge_temperature,
        "keep_alive": args.keep_alive,
        "max_turns": args.max_turns,
        "model": args.model,
        "num_ctx": args.num_ctx,
        "order_seed": args.order_seed,
        "rubric": {
            "key_terms": list(SUMMARY_KEY_TERMS),
            "max_chars": SUMMARY_MAX_CHARS,
            "min_key_terms": SUMMARY_MIN_KEY_TERMS,
        },
        "run_timeout_s": args.run_timeout,
        "seed": args.seed,
        "system_prompts": {control: system_prompt(control) for control in CONTROL_ORDER},
        "tasks": {task_id: _task_payload(task) for task_id, task in sorted(TASKS.items())},
        "temperature": args.temperature,
        "think": False,
        "tools": tools,
    }


def config_hash(payload):
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint(args):
    payload = fingerprint_payload(args)
    return config_hash(payload), payload
