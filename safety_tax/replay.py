"""Cost-only replay. Re-decide logged tool calls. Do not rerun the agent.

Calls are deduplicated within one source experiment by
(source model, source config hash, task, tool, normalized arguments).
Different experiments do not collapse into one call. Each unique call's
allow/deny decision is one trial. --replay-n repeats that call only to
time it. Resume identity is the source record (model, config hash, run,
call index) plus a hash of the full arguments, so a later experiment
cannot skip or overwrite an earlier one. Rows are record_type=cost_replay
and the effectiveness table ignores them. This does not measure whether
the control stops the canary.
"""
import hashlib
import json

from safety_tax.client import CallTimeout, OllamaClient, OllamaError, lookup_model_identity
from safety_tax.config import SCHEMA, default_host
from safety_tax.controls import decision_label, is_legitimate_action, review_call
from safety_tax.identity import fingerprint, git_sha
from safety_tax.logio import SyntheticFixtureError, append_jsonl, cell_key, is_effectiveness_record, load_jsonl
from safety_tax.loop import Deadline, call_seed_for, prepare_user
from safety_tax.tasks import get_task


def normalize_args(args):
    """Stable identity for a tool call.

    Path is the base name, case-folded, because the gate compares names that
    way. Other strings are stripped. Key order does not matter.
    """
    if not isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    cleaned = {}
    for key in sorted(args):
        value = args[key]
        if key == "path" and isinstance(value, str):
            cleaned[key] = value.replace("\\", "/").split("/")[-1].casefold()
        elif isinstance(value, str):
            cleaned[key] = value.strip()
        else:
            cleaned[key] = value
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=str)


def hash_args(args):
    """SHA-256 of the full arguments. Key order does not matter. Not normalized."""
    blob = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_calls(records, tasks):
    """Unique tool calls from end-to-end rows.

    Warm-up and earlier replays are skipped. Within one source experiment
    (source model and source config hash), the same task, tool, and
    normalized arguments are one call no matter how many runs logged it.
    A different source config hash is a different call even when the run
    index and the arguments match. The first occurrence supplies the
    source identity, including a hash of that call's full arguments.
    """
    wanted = set(tasks) if tasks else None
    order = []
    seen = {}
    raw_count = 0
    for record in records:
        if not is_effectiveness_record(record):
            continue
        if wanted is not None and record["task"] not in wanted:
            continue
        source_model = record.get("model") or ""
        source_config = record.get("config_hash") or ""
        for index, call in enumerate(record.get("tool_calls") or []):
            raw_count += 1
            tool = call.get("tool") or ""
            args = call.get("args") or {}
            normalized = normalize_args(args)
            key = (source_model, source_config, record["task"], tool, normalized)
            if key in seen:
                seen[key]["duplicates"] += 1
                continue
            item = {
                "task": record["task"],
                "source_control": record.get("control"),
                "source_run": record.get("run"),
                "source_model": source_model,
                "source_config_hash": source_config,
                "call_index": index,
                "tool": tool,
                "args": args,
                "args_norm": normalized,
                "args_hash": hash_args(args),
                "duplicates": 1,
            }
            seen[key] = item
            order.append(key)
    return [seen[key] for key in order], raw_count


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
    sha = git_sha()
    try:
        records, warnings = load_jsonl(source, allow_synthetic=False)
    except SyntheticFixtureError:
        raise
    model_digest, ollama_version = lookup_model_identity(client, args.model)
    config_hash, _payload = fingerprint(args, model_digest=model_digest)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    calls, raw_count = extract_calls(records, args.tasks)
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
        probe_model(
            args, client, config_hash, sha, purpose="cost-replay warmup",
            model_digest=model_digest, ollama_version=ollama_version,
        )
    out = args.out
    existing, out_warnings = load_jsonl(out, allow_synthetic=False) if out != source else (records, [])
    for warning in out_warnings:
        print(f"warning: {warning}", flush=True)
    done = _done_keys(existing, args.retry_errors)
    n = args.replay_n
    print(
        f"Cost replay: {len(calls)} unique calls "
        f"(collapsed within each source experiment from {raw_count} logged calls) "
        f"× {len(args.controls)} controls "
        f"× {n} latency repeats, config_hash={config_hash}, source={source}, log={out}. "
        "Allow/deny is counted once per unique call. These rows are not effectiveness results.",
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
                    "source_model": item["source_model"],
                    "source_config_hash": item["source_config_hash"],
                    "source_run": item["source_run"],
                    "source_call": item["call_index"],
                    "source_control": item["source_control"],
                    "args_hash": item["args_hash"],
                }
                if cell_key(stub) in done:
                    skipped += 1
                    continue
                deadline = Deadline(args.run_timeout)
                this_seed = call_seed_for(args.seed, replay_index)
                try:
                    legitimate = is_legitimate_action(task, item["tool"], item["args"])
                except Exception:
                    legitimate = False
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
                    label = decision_label(decision)
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
                        "model_digest": model_digest,
                        "ollama_version": ollama_version,
                        "source_run": item["source_run"],
                        "source_call": item["call_index"],
                        "source_control": item["source_control"],
                        "source_model": item["source_model"],
                        "source_config_hash": item["source_config_hash"],
                        "tool": item["tool"],
                        "args_norm": item["args_norm"],
                        "args_hash": item["args_hash"],
                        "legitimate": legitimate,
                        "duplicates_collapsed": item["duplicates"],
                        "seed": args.seed,
                        "call_seed": this_seed,
                        "keep_alive": args.keep_alive,
                        "think": False,
                        "judge_num_predict": args.judge_num_predict if control == "judge" else None,
                        "decision": label,
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
                        "model_digest": model_digest,
                        "ollama_version": ollama_version,
                        "source_run": item["source_run"],
                        "source_call": item["call_index"],
                        "source_control": item["source_control"],
                        "source_model": item["source_model"],
                        "source_config_hash": item["source_config_hash"],
                        "tool": item["tool"],
                        "args_norm": item["args_norm"],
                        "args_hash": item["args_hash"],
                        "legitimate": legitimate,
                        "decision": None,
                        "judge_gate_seconds": None,
                        "model_calls": [],
                        "timeout": isinstance(exc, CallTimeout),
                        "error": str(exc),
                        "wall_seconds": None,
                    }
                append_jsonl(out, record)
                done.add(cell_key(record))
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
