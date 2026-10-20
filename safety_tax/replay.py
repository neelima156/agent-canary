"""Cost-only replay. Re-decide logged tool calls. Do not rerun the agent.

This measures how long a control takes to allow or deny a call that already
happened. It does not measure whether the control stops the canary. Rows are
marked record_type=cost_replay and the effectiveness table ignores them.
"""
from safety_tax.client import CallTimeout, OllamaClient, OllamaError
from safety_tax.config import SCHEMA, default_host
from safety_tax.controls import review_call
from safety_tax.identity import fingerprint, git_sha
from safety_tax.logio import SyntheticFixtureError, append_jsonl, cell_key, is_effectiveness_record, load_jsonl
from safety_tax.loop import Deadline, call_seed_for, prepare_user
from safety_tax.tasks import get_task


def extract_calls(records, tasks):
    """Tool calls from end-to-end rows. Warm-up and earlier replays are skipped."""
    wanted = set(tasks) if tasks else None
    extracted = []
    for record in records:
        if not is_effectiveness_record(record):
            continue
        if wanted is not None and record["task"] not in wanted:
            continue
        for index, call in enumerate(record.get("tool_calls") or []):
            extracted.append({
                "task": record["task"],
                "source_control": record.get("control"),
                "source_run": record.get("run"),
                "source_model": record.get("model"),
                "call_index": index,
                "tool": call.get("tool") or "",
                "args": call.get("args") or {},
            })
    return extracted


def _done_keys(records, retry_errors):
    last = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("record_type") != "cost_replay":
            continue
        if "task" not in record or "control" not in record or "run" not in record or "model" not in record:
            continue
        last[cell_key(record)] = record
    done = set()
    for key, record in last.items():
        if retry_errors and record.get("error") and not record.get("timeout"):
            continue
        done.add(key)
    return done


def run_replay(args, client=None):
    host = args.ollama_host or default_host()
    if client is None:
        client = OllamaClient(host)
    source = args.replay_from or args.out
    config_hash, _payload = fingerprint(args)
    sha = git_sha()
    try:
        records, warnings = load_jsonl(source, allow_synthetic=False)
    except SyntheticFixtureError:
        raise
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    calls = extract_calls(records, args.tasks)
    if not calls:
        print(
            f"No recorded tool calls to replay in {source}. "
            "Run an end-to-end batch first. This mode does not invent calls.",
            flush=True,
        )
        return 2
    judge_model = args.judge_model or args.model
    needs_model = "judge" in args.controls
    if needs_model:
        from safety_tax.runner import probe_model
        probe_model(args, client, config_hash, sha, purpose="cost-replay warmup")
    out = args.out
    existing, out_warnings = load_jsonl(out, allow_synthetic=False) if out != source else (records, [])
    for warning in out_warnings:
        print(f"warning: {warning}", flush=True)
    done = _done_keys(existing, args.retry_errors)
    n = args.replay_n
    print(
        f"Cost replay: {len(calls)} logged calls × {len(args.controls)} controls × {n} replays, "
        f"config_hash={config_hash}, source={source}, log={out}. "
        "These rows are not effectiveness results.",
        flush=True,
    )
    written = 0
    skipped = 0
    for item in calls:
        task = get_task(item["task"])
        for control in args.controls:
            for replay_index in range(1, n + 1):
                stub = {
                    "record_type": "cost_replay",
                    "task": item["task"],
                    "control": control,
                    "run": replay_index,
                    "model": args.model,
                    "judge_model": judge_model if control == "judge" else None,
                    "config_hash": config_hash,
                    "source_run": item["source_run"],
                    "source_call": item["call_index"],
                    "source_control": item["source_control"],
                }
                if cell_key(stub) in done:
                    skipped += 1
                    continue
                deadline = Deadline(args.run_timeout)
                this_seed = call_seed_for(args.seed, replay_index)
                try:
                    review = review_call(
                        control, task, item["tool"], item["args"],
                        client=client, judge_model=judge_model, deadline=deadline,
                        call_timeout=args.call_timeout,
                        judge_temperature=args.judge_temperature,
                        num_ctx=args.num_ctx, prepare_user=prepare_user,
                        seed=this_seed, keep_alive=args.keep_alive, think=False,
                        num_predict=args.judge_num_predict,
                    )
                    decision = review.decision
                    record = {
                        "schema": SCHEMA,
                        "record_type": "cost_replay",
                        "warmup": False,
                        "task": item["task"],
                        "control": control,
                        "run": replay_index,
                        "model": args.model,
                        "judge_model": judge_model if control == "judge" else None,
                        "config_hash": config_hash,
                        "git_sha": sha,
                        "source_run": item["source_run"],
                        "source_call": item["call_index"],
                        "source_control": item["source_control"],
                        "source_model": item["source_model"],
                        "tool": item["tool"],
                        "seed": args.seed,
                        "call_seed": this_seed,
                        "keep_alive": args.keep_alive,
                        "think": False,
                        "judge_num_predict": args.judge_num_predict if control == "judge" else None,
                        "decision": None if decision is None else ("allow" if decision.allow else "deny"),
                        "reason": None if decision is None else decision.reason,
                        "judge_parse": None if decision is None else decision.judge_parse,
                        "judge_gate_seconds": round(review.seconds, 3),
                        "model_calls": review.model_calls,
                        "timeout": bool(review.timeout),
                        "timeout_scope": review.timeout_scope,
                        "timeout_cap_s": args.call_timeout if review.timeout else None,
                        "call_timeout_s": args.call_timeout,
                        "run_timeout_s": args.run_timeout,
                        "error": review.error,
                        "wall_seconds": round(review.seconds, 3),
                    }
                except (CallTimeout, OllamaError, Exception) as exc:
                    record = {
                        "schema": SCHEMA,
                        "record_type": "cost_replay",
                        "warmup": False,
                        "task": item["task"],
                        "control": control,
                        "run": replay_index,
                        "model": args.model,
                        "judge_model": judge_model if control == "judge" else None,
                        "config_hash": config_hash,
                        "git_sha": sha,
                        "source_run": item["source_run"],
                        "source_call": item["call_index"],
                        "source_control": item["source_control"],
                        "tool": item["tool"],
                        "decision": None,
                        "judge_gate_seconds": None,
                        "model_calls": [],
                        "timeout": isinstance(exc, CallTimeout),
                        "error": str(exc),
                        "wall_seconds": None,
                    }
                append_jsonl(out, record)
                written += 1
    print(f"cost replay done: wrote {written}, skipped {skipped}, log {out}", flush=True)
    if not args.no_summary:
        from safety_tax.summarize import write_summary
        md_path, csv_path = write_summary(out)
        print(f"wrote {md_path}", flush=True)
        print(f"wrote {csv_path}", flush=True)
    return 0


def main(argv=None):
    """Entry point used by runner.main. Kept import-light for tests."""
    from safety_tax.runner import parse_args, validation_error
    import sys
    args = parse_args(argv)
    error = validation_error(args)
    if error:
        print(error, file=sys.stderr)
        return 2
    try:
        return run_replay(args)
    except SyntheticFixtureError as exc:
        print(exc, file=sys.stderr)
        return 2
