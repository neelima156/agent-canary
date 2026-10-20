"""Run the task × control matrix. A failed cell is one JSONL line, then the next cell."""
import os
import random
import sys

from safety_tax.client import (
    CallTimeout,
    OllamaClient,
    OllamaError,
    describe_preflight,
    empty_timing,
    format_preflight,
)
from safety_tax.config import (
    CONTROL_ORDER,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_COLD_LOAD_NS,
    DEFAULT_CONTROLS,
    DEFAULT_JUDGE_NUM_PREDICT,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_ORDER_SEED,
    DEFAULT_OUT,
    DEFAULT_REPLAY_N,
    DEFAULT_RUN_TIMEOUT_S,
    DEFAULT_RUNS,
    DEFAULT_SEED,
    DEFAULT_TASKS,
    DEFAULT_TEMPERATURE,
    REPLAY_CONTROLS,
    SCHEMA,
    SMOKE_CONTROLS,
    SMOKE_TASKS,
    TASK_ORDER,
    default_host,
)
from safety_tax.identity import fingerprint, git_sha
from safety_tax.logio import SyntheticFixtureError, append_jsonl, cell_key, is_run_record, load_jsonl
from safety_tax.loop import error_record, prepare_user, run_task
from safety_tax.summarize import write_summary
from safety_tax.tasks import get_task

WARMUP_USER = "Reply with the single word READY."


def sandbox_for(out_path, task_id, control, run_index):
    root = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".", "sandbox")
    return os.path.join(root, f"{task_id}__{control}__run{run_index}")


def planned_cells(tasks, controls, runs, seed):
    """Every cell, shuffled with a fixed seed so control and run are interleaved.

    The nested order task → control → run puts one control first for the whole
    batch. A seeded shuffle spreads that position effect. The seed is recorded
    on every row and inside the config hash.
    """
    cells = [
        (task_id, control, run_index)
        for task_id in tasks
        for control in controls
        for run_index in range(1, runs + 1)
    ]
    random.Random(seed).shuffle(cells)
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
    if args.replay_n < 1:
        return "--replay-n must be >= 1"
    if args.judge_num_predict < 1:
        return "--judge-num-predict must be >= 1"
    if args.bootstrap_samples < 1:
        return "--bootstrap-samples must be >= 1"
    return None


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Safety Tax: latency, tokens, and false blocks of safety controls on a canary agent.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None, help="Defaults to --model. Kept in RAM if it is the same model.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Runs per cell. Default is the recommended end-to-end size, not a smoke test.")
    parser.add_argument("--tasks", nargs="+", default=None, choices=list(TASK_ORDER),
                        help="Default: the recommended reduced matrix. --catalog selects every task.")
    parser.add_argument("--controls", nargs="+", default=None, choices=list(CONTROL_ORDER),
                        help="Default: the recommended reduced matrix. --catalog selects every control.")
    parser.add_argument("--catalog", action="store_true",
                        help="Use every task and every control. Does not change --runs.")
    parser.add_argument("--smoke", action="store_true",
                        help="Preset: --runs 1 and a two-task, two-control matrix. Overrides --tasks, --controls, and --runs.")
    parser.add_argument("--preflight", action="store_true",
                        help="Probe one call, print thinking and token fields, and exit. Does not run the matrix.")
    parser.add_argument("--cost-replay", action="store_true",
                        help="Replay logged tool calls through controls. Not an effectiveness run.")
    parser.add_argument("--replay-from", default=None,
                        help="JSONL of end-to-end rows to replay. Defaults to --out.")
    parser.add_argument("--replay-n", type=int, default=DEFAULT_REPLAY_N,
                        help="Replays of each logged tool call in --cost-replay.")
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Master sampling seed. Run i sends seed+i. agent.py does not set a seed.")
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED,
                        help="Seed for the cell shuffle. Recorded on every row.")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE,
                        help="Ollama keep_alive so the model stays loaded between calls.")
    parser.add_argument("--judge-num-predict", type=int, default=DEFAULT_JUDGE_NUM_PREDICT,
                        help="Generation cap for the judge only. The agent is not capped.")
    parser.add_argument("--cold-load-ns", type=float, default=DEFAULT_COLD_LOAD_NS,
                        help="Mark a run cold when any call's load_duration is at least this many nanoseconds.")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES,
                        help="Resamples for the overhead interval in the summary.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Redo logged rows that have an error and are not timeouts. Timeouts stay logged.")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args(argv)
    if args.tasks is None:
        args.tasks = list(TASK_ORDER if args.cost_replay else DEFAULT_TASKS)
    if args.controls is None:
        args.controls = list(REPLAY_CONTROLS if args.cost_replay else DEFAULT_CONTROLS)
    if args.catalog:
        args.tasks = list(TASK_ORDER)
        args.controls = list(CONTROL_ORDER)
    if args.smoke and not args.cost_replay and not args.preflight:
        args.runs = 1
        args.tasks = list(SMOKE_TASKS)
        args.controls = list(SMOKE_CONTROLS)
    return args


def _done_keys(records, retry_errors):
    """Last line for a cell wins, matching the summarizer."""
    last = {}
    for record in records:
        if not is_run_record(record):
            continue
        if record.get("record_type") not in (None, "end_to_end"):
            continue
        if record.get("warmup"):
            continue
        last[cell_key(record)] = record
    done = set()
    for key, record in last.items():
        if retry_errors and record.get("error") and not record.get("timeout"):
            continue
        done.add(key)
    return done


def _stub_key(task_id, control, run_index, model, judge_model, config_hash):
    return cell_key({
        "record_type": "end_to_end",
        "warmup": False,
        "task": task_id,
        "control": control,
        "run": run_index,
        "model": model,
        "judge_model": judge_model if control == "judge" else None,
        "config_hash": config_hash,
    })


def probe_model(args, client, config_hash, sha, purpose="warmup"):
    """One discarded call. Prints the preflight lines and appends a warm-up row.

    Returns a process exit code: 0 when the probe returned a body, 2 otherwise.
    The row is marked warmup and is excluded from summaries.
    """
    load_jsonl(args.out, allow_synthetic=False)
    model = args.model
    timeout = False
    error = None
    message = {}
    timing = empty_timing("warmup", 0.0, False)
    try:
        message, timing = client.chat(
            model,
            [
                {"role": "system", "content": "You answer with one word."},
                {"role": "user", "content": prepare_user(model, WARMUP_USER)},
            ],
            tools=None,
            temperature=args.temperature,
            timeout=args.call_timeout,
            num_ctx=args.num_ctx,
            seed=args.seed,
            keep_alive=args.keep_alive,
            think=False,
            num_predict=16,
        )
    except CallTimeout as exc:
        timeout = True
        timing = empty_timing("warmup", exc.wall_seconds, True)
    except OllamaError as exc:
        error = str(exc)
        timing = empty_timing("warmup", 0.0, False)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        timing = empty_timing("warmup", 0.0, False)
    report = describe_preflight(message, timing)
    print(format_preflight(report), flush=True)
    record = {
        "schema": SCHEMA,
        "record_type": "warmup",
        "warmup": True,
        "purpose": purpose,
        "task": "warmup",
        "control": "warmup",
        "run": 0,
        "model": model,
        "judge_model": None,
        "config_hash": config_hash,
        "git_sha": sha,
        "seed": args.seed,
        "keep_alive": args.keep_alive,
        "think": False,
        "order_seed": args.order_seed,
        "preflight": report,
        "model_calls": [timing],
        "wall_seconds": timing.get("wall_seconds"),
        "timeout": timeout,
        "timeout_scope": "call" if timeout else None,
        "timeout_cap_s": args.call_timeout if timeout else None,
        "error": error,
        "call_timeout_s": args.call_timeout,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "ollama_host": args.ollama_host or default_host(),
    }
    append_jsonl(args.out, record)
    if error or timeout:
        print(f"preflight failed: timeout={timeout} error={error}", flush=True)
        return 2
    return 0


def run_batch(args, client=None, clock=None):
    host = args.ollama_host or default_host()
    if client is None:
        client = OllamaClient(host)
    judge_model = args.judge_model or args.model
    config_hash, _payload = fingerprint(args)
    sha = git_sha()
    out = args.out
    existing, warnings = load_jsonl(out, allow_synthetic=False)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    done = _done_keys(existing, args.retry_errors)
    cells = planned_cells(args.tasks, args.controls, args.runs, args.order_seed)
    pending = []
    for index, (task_id, control, run_index) in enumerate(cells, 1):
        key = _stub_key(task_id, control, run_index, args.model, judge_model, config_hash)
        if key in done:
            print(f"skip {index}/{len(cells)} {task_id} {control} run {run_index} (already in log)", flush=True)
            continue
        pending.append((index, task_id, control, run_index))
    print(
        f"Safety Tax: {len(cells)} cells ({len(pending)} pending), model={args.model}, "
        f"judge={judge_model}, order_seed={args.order_seed}, config_hash={config_hash}, "
        f"git_sha={sha or 'unavailable'}, call_timeout={args.call_timeout}s, "
        f"run_timeout={args.run_timeout}s, log={out}",
        flush=True,
    )
    if pending:
        probe_model(args, client, config_hash, sha, purpose="batch warmup")
    executed = []
    skipped = len(cells) - len(pending)
    for index, task_id, control, run_index in pending:
        plan_index = index
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
                config_hash=config_hash,
                git_sha=sha,
                order_seed=args.order_seed,
                order_index=plan_index,
                seed=args.seed,
                keep_alive=args.keep_alive,
                think=False,
                judge_num_predict=args.judge_num_predict,
            )
        except Exception as exc:
            record = error_record(
                task_id, control, run_index,
                model=args.model, judge_model=judge_model, host=host,
                config_hash=config_hash, git_sha=sha, exc=exc,
                order_index=plan_index, order_seed=args.order_seed,
                call_timeout=args.call_timeout, run_timeout=args.run_timeout,
                temperature=args.temperature, judge_temperature=args.judge_temperature,
                max_turns=args.max_turns, num_ctx=args.num_ctx, seed=args.seed,
                keep_alive=args.keep_alive, judge_num_predict=args.judge_num_predict,
                cold_load_ns=args.cold_load_ns,
            )
        append_jsonl(out, record)
        executed.append(record)
        print(
            f"{plan_index}/{len(cells)} {task_id} {control} run {run_index} "
            f"wall={record['wall_seconds']}s cold={record['cold']} "
            f"attack={record['attack_success']} canary={record['canary_written']} "
            f"said={record['said_bluebird']} legit={record['legit_completed']} "
            f"blocked={record['legitimate_action_blocked']} "
            f"constraint={record['constraint_held']} "
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
        md_path, csv_path = write_summary(
            out, bootstrap_samples=args.bootstrap_samples,
        )
        print(f"wrote {md_path}", flush=True)
        print(f"wrote {csv_path}", flush=True)
    return executed


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        host = args.ollama_host or default_host()
        client = OllamaClient(host)
        config_hash, _payload = fingerprint(args)
        try:
            return probe_model(args, client, config_hash, git_sha(), purpose="preflight")
        except SyntheticFixtureError as exc:
            print(exc, file=sys.stderr)
            return 2
    if args.cost_replay:
        error = validation_error(args)
        if error:
            print(error, file=sys.stderr)
            return 2
        from safety_tax.replay import run_replay
        try:
            return run_replay(args)
        except SyntheticFixtureError as exc:
            print(exc, file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nInterrupted. Finished replay rows are in the log.", file=sys.stderr)
            return 130
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
