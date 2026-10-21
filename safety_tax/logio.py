"""JSONL log shared by the runner and the summarizer."""
import json
import os


class SyntheticFixtureError(RuntimeError):
    pass


def is_run_record(record):
    if not isinstance(record, dict):
        return False
    needed = ("task", "control", "run", "model")
    return all(key in record for key in needed)


def record_kind(record):
    if record.get("record_type"):
        return record["record_type"]
    if record.get("warmup"):
        return "warmup"
    return "end_to_end"


def is_effectiveness_record(record):
    """End-to-end task rows. Warm-up and cost-replay rows are not outcomes."""
    if not is_run_record(record):
        return False
    return record_kind(record) == "end_to_end"


def cell_key(record):
    """Resume identity.

    Includes the config hash so a change of prompts, model parameters, task
    definitions, or code version does not skip a cell logged under an older
    experiment. Judge model matters only for the judge control. Cost-replay
    rows key on the source experiment (model and config hash), the source
    run and call index, and a hash of the full arguments, so two experiments
    that logged the same call index do not overwrite or skip each other.
    """
    control = record["control"]
    judge = record.get("judge_model") or "" if control == "judge" else ""
    kind = record_kind(record)
    base = (
        kind,
        record["task"],
        control,
        int(record["run"]),
        record["model"],
        judge,
        record.get("config_hash") or "",
    )
    if kind == "cost_replay":
        return base + (
            record.get("source_model") or "",
            record.get("source_config_hash") or "",
            record.get("source_run"),
            record.get("source_call"),
            record.get("args_hash") or "",
        )
    return base


def synthetic_message(path):
    return (
        f"Refusing {path}: it is marked synthetic_fixture. "
        "That file is test data, not experiment results. "
        "Point --out at results/safety-tax/runs.jsonl for a real run."
    )


def load_jsonl(path, allow_synthetic=False):
    records = []
    warnings = []
    if not path or not os.path.exists(path):
        return records, warnings
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"{path}:{lineno}: invalid JSON, skipped")
                continue
            if not isinstance(record, dict):
                warnings.append(f"{path}:{lineno}: not an object, skipped")
                continue
            records.append(record)
    if any(record.get("synthetic_fixture") for record in records) and not allow_synthetic:
        raise SyntheticFixtureError(synthetic_message(path))
    return records, warnings


def append_jsonl(path, record):
    """Append one JSON object as its own line.

    A crashed writer can leave a partial last line with no newline. Opening
    in append mode and writing the next object would glue that fragment to
    the new record, so both lines fail to parse and resume reruns a cell
    whose previous attempt was actually saved. If the file does not end in
    a newline, write one before the new record. The partial fragment stays
    a skipped bad line; the new record is intact.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with open(path, "a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(payload + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def dedupe_runs(records):
    """Last complete line for a cell wins. Non-run objects are dropped."""
    order = []
    by_key = {}
    for record in records:
        if not is_run_record(record):
            continue
        key = cell_key(record)
        if key not in by_key:
            order.append(key)
        by_key[key] = record
    return [by_key[key] for key in order]
