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


def cell_key(record):
    """Resume identity. Judge model matters only for the judge control."""
    control = record["control"]
    judge = record.get("judge_model") or "" if control == "judge" else ""
    return (record["task"], control, int(record["run"]), record["model"], judge)


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
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
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
