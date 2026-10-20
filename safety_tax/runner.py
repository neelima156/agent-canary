"""Run the task × control matrix. A failed cell is one JSONL line, then the next cell."""
import os
import sys

from safety_tax.client import OllamaClient
from safety_tax.config import (
    CONTROL_ORDER,
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_COLD_LOAD_NS,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OUT,
    DEFAULT_RUN_TIMEOUT_S,
    DEFAULT_RUNS,
    DEFAULT_TEMPERATURE,
    TASK_ORDER,
    default_host,
)
from safety_tax.logio import SyntheticFixtureError, append_jsonl, cell_key, is_run_record, load_jsonl
from safety_tax.loop import run_task
from safety_tax.summarize import write_summary
from safety_tax.tasks import get_task


def sandbox_for(out_path, task_id, control, run_index):
    root = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".", "sandbox")
    return os.path.join(root, f"{task_id}__{control}__run{run_index}")


def planned_cells(tasks, controls, runs):
    cells = []
    for task_id in tasks:
        for control in controls:
            for run_index in range(1, runs + 1):
                cells.append((task_id, control, run_index))
    return cells


def validation_error(args):
    if args.runs < 1:
        return "--runs must be >= 1"
    if args.call_timeout <= 0 or args.run_timeout <= 0:
        return "--call-timeout and --run-timeout must be > 0"
    if args.max_turns < 1:
        return "--max-turns must be >= 1"
    if args.temperature < 0 or args.judge_temperature < 0:
        return "temperatures must be >= 0"
    if args.cold_load_ns < 0:
        return "--cold-load-ns must be >= 0"
    return None


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Safety Tax: latency, tokens, and false blocks of safety controls on a canary agent.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None, help="Defaults to --model. Kept in RAM if it is the same model.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER), choices=list(TASK_ORDER))
    parser.add_argument("--controls", nargs="+", default=list(CONTROL_ORDER), choices=list(CONTROL_ORDER))
    parser.add_argument("--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT_S,
                        help="Seconds for one Ollama HTTP call.")
    parser.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT_S,
                        help="Seconds for one task, across every model call and tool.")
    parser.add_argument("--ollama-host", default=None, help="Defaults to $OLLAMA_HOST or http://127.0.0.1:11434.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--judge-temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--cold-load-ns", type=float, default=DEFAULT_COLD_LOAD_NS,
                        help="Mark a run cold when any call's load_duration is at least this many nanoseconds.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Redo logged rows that have an error and are not timeouts. Timeouts stay logged.")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args(argv)


def _done_keys(records, retry_errors):
    """Last line for a cell wins, matching the summarizer."""
    last = {}
    for record in records:
        if not is_run_record(record):
            continue
        last[cell_key(record)] = record
    done = set()
    for key, record in last.items():
        if retry_errors and record.get("error") and not record.get("timeout"):
            continue
        done.add(key)
    return done


def run_batch(args, client=None, clock=None):
    host = args.ollama_host or default_host()
    if client is None:
        client = OllamaClient(host)
    judge_model = args.judge_model or args.model
    out = args.out
    existing, warnings = load_jsonl(out, allow_synthetic=False)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    done = _done_keys(existing, args.retry_errors)
    cells = planned_cells(args.tasks, args.controls, args.runs)
    print(
        f"Safety Tax: {len(cells)} cells, model={args.model}, judge={judge_model}, "
        f"call_timeout={args.call_timeout}s, run_timeout={args.run_timeout}s, log={out}",
        flush=True,
    )
    executed = []
    skipped = 0
    for index, (task_id, control, run_index) in enumerate(cells, 1):
        key = (task_id, control, run_index, args.model, judge_model if control == "judge" else "")
        if key in done:
            skipped += 1
            print(f"skip {index}/{len(cells)} {task_id} {control} run {run_index} (already in log)", flush=True)
            continue
        task = get_task(task_id)
        sandbox = sandbox_for(out, task_id, control, run_index)
        try:
            record = run_task(
                task, control, run_index, args.model, client, sandbox,
                judge_model=judge_model,
                call_timeout=args.call_timeout,
                run_timeout=args.run_timeout,
                temperature=args.temperature,
                judge_temperature=args.judge_temperature,
                max_turns=args.max_turns,
                num_ctx=args.num_ctx,
                cold_load_ns=args.cold_load_ns,
                ollama_host=host,
                clock=clock,
            )
        except Exception as exc:
            record = {
                "schema": 1,
                "task": task_id,
                "control": control,
                "run": run_index,
                "model": args.model,
                "judge_model": judge_model if control == "judge" else None,
                "cold": False,
                "cold_load_ns_threshold": args.cold_load_ns,
                "wall_seconds": None,
                "model_calls": [],
                "n_model_calls": 0,
                "judge_gate_seconds": 0.0,
                "tool_seconds": 0.0,
                "canary_written": False,
                "said_bluebird": False,
                "legit_completed": False,
                "legitimate_action_blocked": False,
                "notes_present": False,
                "error": f"{type(exc).__name__}: {exc}",
                "timeout": False,
                "timeout_scope": None,
                "stop_reason": "error",
                "tool_calls": [],
                "final": "",
                "sandbox_files": [],
                "ollama_host": host,
                "call_timeout_s": args.call_timeout,
                "run_timeout_s": args.run_timeout,
                "temperature": args.temperature,
                "judge_temperature": args.judge_temperature,
                "max_turns": args.max_turns,
            }
        append_jsonl(out, record)
        executed.append(record)
        print(
            f"{index}/{len(cells)} {task_id} {control} run {run_index} "
            f"wall={record['wall_seconds']}s cold={record['cold']} "
            f"canary={record['canary_written']} legit={record['legit_completed']} "
            f"blocked={record['legitimate_action_blocked']} "
            f"timeout={record['timeout']} error={record['error']}",
            flush=True,
        )
    print(f"done: executed {len(executed)}, skipped {skipped}, log {out}", flush=True)
    if executed and all(record.get("error") and not record.get("timeout") for record in executed):
        print(
            "Every new run recorded an error. If Ollama is not up, start it and rerun the same command with --retry-errors.",
            flush=True,
        )
    if not args.no_summary:
        md_path, csv_path = write_summary(out)
        print(f"wrote {md_path}", flush=True)
        print(f"wrote {csv_path}", flush=True)
    return executed


def main(argv=None):
    args = parse_args(argv)
    error = validation_error(args)
    if error:
        print(error, file=sys.stderr)
        return 2
    try:
        run_batch(args)
    except SyntheticFixtureError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Finished runs are in the log. Run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    return 0
