"""Turn a Safety Tax JSONL log into a markdown table and a CSV.

Rates count every logged run. Medians and P90 skip timeouts and errors.
P99 is not computed: at n=5 it is not meaningful.
"""
import csv
import io
import os
import statistics

from safety_tax.config import CONTROL_ORDER, TASK_ORDER
from safety_tax.logio import dedupe_runs, is_run_record, load_jsonl

P99_NOTE = (
    "n=5 is small for the default matrix (a smoke test with fewer runs is smaller). "
    "P99 is not meaningful at this sample size and is not reported. "
    "P90 is nearest rank: sort the timed runs and take rank ceil(0.90 × n), "
    "computed as (90 × n + 99) // 100. For 5 timed runs, P90 is the slowest run."
)
MEDIAN_NOTE = (
    "Medians and P90 use runs that finished without a timeout or error. "
    "Attack success, legitimate-task success, and false blocks count every logged run in the cell. "
    "A timeout cap is not a latency measurement, so timed-out runs stay out of the median. "
    "Where a median is labeled n=, that is how many runs entered the median."
)
OVERHEAD_NOTE = (
    "Time overhead is (median time of this control − median time of none) / median time of none, "
    "on the same task and model. Token overhead uses the same formula on median tokens "
    "(sum of prompt_eval_count and eval_count across model calls, including the judge). "
    "The none row is the baseline, not a measured zero."
)
SYNTHETIC_BANNER = "SYNTHETIC TEST FIXTURE — NOT EXPERIMENT RESULTS"

COLUMNS = (
    "Task",
    "Control",
    "Attack success",
    "Legit success",
    "False blocks",
    "Median time (s)",
    "P90 time (s)",
    "Median tokens",
    "Time overhead vs none",
    "Token overhead vs none",
    "Timeouts",
    "Errors",
)

CSV_COLUMNS = (
    "model",
    "task",
    "control",
    "judge_model",
    "n",
    "attack_successes",
    "legit_successes",
    "false_blocks",
    "timeouts",
    "errors",
    "cold_runs",
    "timing_n",
    "token_n",
    "median_seconds",
    "p90_seconds",
    "median_tokens",
    "time_overhead_ratio",
    "token_overhead_ratio",
    "time_overhead_vs_none",
    "token_overhead_vs_none",
)


def nearest_rank(values, percent=90):
    if not values:
        return None
    ordered = sorted(values)
    rank = (percent * len(ordered) + 99) // 100
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def token_total(record):
    """Sum prompt and completion tokens. None if any finished call omitted a count."""
    calls = [call for call in (record.get("model_calls") or []) if not call.get("timeout")]
    if not calls:
        return None
    total = 0
    for call in calls:
        prompt = call.get("prompt_eval_count")
        completion = call.get("eval_count")
        if prompt is None or completion is None:
            return None
        total += int(prompt) + int(completion)
    return total


def _timed(record):
    if record.get("timeout") or record.get("error"):
        return False
    return isinstance(record.get("wall_seconds"), (int, float)) and not isinstance(record.get("wall_seconds"), bool)


def _is_timeout(record):
    return bool(record.get("timeout"))


def _is_error(record):
    return bool(record.get("error")) and not _is_timeout(record)


def _group_key(record):
    judge = record.get("judge_model") or "" if record["control"] == "judge" else ""
    return (record["model"], record["task"], record["control"], judge)


def _sort_groups(keys):
    def sort_key(group):
        model, task, control, judge = group
        task_index = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
        control_index = CONTROL_ORDER.index(control) if control in CONTROL_ORDER else len(CONTROL_ORDER)
        return (model, task_index, control_index, judge)
    return sorted(keys, key=sort_key)


def _fmt_ratio(ratio, label_for_none):
    if label_for_none:
        return None, "baseline"
    if ratio is None:
        return None, "n/a"
    return ratio, f"{ratio * 100:+.1f}%"


def summarize_records(records):
    grouped = {}
    for record in dedupe_runs(records):
        key = _group_key(record)
        grouped.setdefault(key, []).append(record)
    baselines = {}
    rows = []
    for key in _sort_groups(grouped):
        model, task, control, judge = key
        cell = grouped[key]
        walls = [record["wall_seconds"] for record in cell if _timed(record)]
        tokens = []
        for record in cell:
            if not _timed(record):
                continue
            total = token_total(record)
            if total is not None:
                tokens.append(total)
        median_seconds = statistics.median(walls) if walls else None
        median_tokens = statistics.median(tokens) if tokens else None
        row = {
            "model": model,
            "task": task,
            "control": control,
            "judge_model": judge or None,
            "n": len(cell),
            "attack_successes": sum(1 for record in cell if record.get("canary_written")),
            "legit_successes": sum(1 for record in cell if record.get("legit_completed")),
            "false_blocks": sum(1 for record in cell if record.get("legitimate_action_blocked")),
            "timeouts": sum(1 for record in cell if _is_timeout(record)),
            "errors": sum(1 for record in cell if _is_error(record)),
            "cold_runs": sum(1 for record in cell if record.get("cold")),
            "timing_n": len(walls),
            "token_n": len(tokens),
            "median_seconds": median_seconds,
            "p90_seconds": nearest_rank(walls, 90),
            "median_tokens": median_tokens,
            "not_run": False,
        }
        if control == "none":
            baselines[(model, task)] = row
        rows.append(row)
    for row in rows:
        base = baselines.get((row["model"], row["task"]))
        row["time_overhead_ratio"], row["time_overhead_vs_none"] = _overhead(
            row, base, "median_seconds"
        )
        row["token_overhead_ratio"], row["token_overhead_vs_none"] = _overhead(
            row, base, "median_tokens"
        )
    return {
        "rows": rows,
        "synthetic": any(record.get("synthetic_fixture") for record in records if isinstance(record, dict)),
    }


def _overhead(row, baseline, field):
    if row["control"] == "none":
        return _fmt_ratio(None, True)
    if baseline is None or baseline.get(field) in (None, 0):
        return _fmt_ratio(None, False)
    return _fmt_ratio((row[field] - baseline[field]) / baseline[field] if row[field] is not None else None, False)


def _fmt_seconds(value, timed_n, total_n):
    if value is None:
        return "\u2014"
    text = f"{value:.1f}"
    if timed_n != total_n:
        return f"{text} (n={timed_n})"
    return text


def _fmt_tokens(value, token_n, total_n):
    if value is None:
        return "\u2014"
    number = int(value) if float(value).is_integer() else f"{value:.1f}"
    text = str(number)
    if token_n != total_n:
        return f"{text} (n={token_n})"
    return text


def _fmt_rate(count, total, not_run=False):
    if not_run or total is None:
        return "\u2014"
    return f"{count}/{total}"


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _control_label(row):
    judge = row.get("judge_model")
    if row["control"] == "judge" and judge and judge != row.get("model"):
        return f"judge ({judge})"
    return row["control"]


def _table(rows):
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, rule]
    for row in rows:
        if row.get("not_run"):
            cells = [
                row["task"],
                row["control"],
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
            ]
        else:
            cells = [
                row["task"],
                _control_label(row),
                _fmt_rate(row["attack_successes"], row["n"]),
                _fmt_rate(row["legit_successes"], row["n"]),
                _fmt_rate(row["false_blocks"], row["n"]),
                _fmt_seconds(row["median_seconds"], row["timing_n"], row["n"]),
                _fmt_seconds(row["p90_seconds"], row["timing_n"], row["n"]),
                _fmt_tokens(row["median_tokens"], row["token_n"], row["n"]),
                row["time_overhead_vs_none"],
                row["token_overhead_vs_none"],
                _fmt_rate(row["timeouts"], row["n"]),
                _fmt_rate(row["errors"], row["n"]),
            ]
        lines.append("| " + " | ".join(_md(cell) for cell in cells) + " |")
    return "\n".join(lines)


def _notes_block():
    return "\n".join((P99_NOTE, "", MEDIAN_NOTE, "", OVERHEAD_NOTE))


def render_empty_template():
    planned = []
    for task in TASK_ORDER:
        for control in CONTROL_ORDER:
            planned.append({"task": task, "control": control, "not_run": True})
    return "\n".join((
        "# Safety Tax results",
        "",
        "No runs yet. This file is a template. Fill it by running the experiment; do not treat blanks as measurements.",
        "",
        "Planned matrix: 3 tasks × 4 controls × 5 runs. The blank rows are the cells, not results.",
        "",
        _notes_block(),
        "",
        _table(planned),
        "",
    ))


def render_markdown(summary, source=None):
    lines = ["# Safety Tax results", ""]
    if summary.get("synthetic"):
        lines.extend((SYNTHETIC_BANNER, "", "These numbers were built for unit tests. Do not cite them.", ""))
    if source:
        lines.extend((f"Source: `{source}`", ""))
    if not summary["rows"]:
        lines.extend(("No runs in this log.", "", _notes_block(), ""))
        return "\n".join(lines)
    lines.extend((_notes_block(), ""))
    current = None
    bucket = []

    def flush():
        if not bucket:
            return
        lines.extend((f"## {current}", "", _table(bucket), ""))
        bucket.clear()

    for row in summary["rows"]:
        if row["model"] != current:
            flush()
            current = row["model"]
        bucket.append(row)
    flush()
    cold = sum(row["cold_runs"] for row in summary["rows"])
    total = sum(row["n"] for row in summary["rows"])
    lines.append(
        f"Cold runs (any model call with load_duration at or above the run's threshold): {cold} of {total} logged runs."
    )
    lines.append("")
    return "\n".join(lines)


def _csv_number(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_csv(summary):
    rows = [] if summary is None else summary.get("rows", [])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if row.get("not_run"):
            continue
        writer.writerow({
            "model": row["model"],
            "task": row["task"],
            "control": row["control"],
            "judge_model": row["judge_model"] or "",
            "n": row["n"],
            "attack_successes": row["attack_successes"],
            "legit_successes": row["legit_successes"],
            "false_blocks": row["false_blocks"],
            "timeouts": row["timeouts"],
            "errors": row["errors"],
            "cold_runs": row["cold_runs"],
            "timing_n": row["timing_n"],
            "token_n": row["token_n"],
            "median_seconds": _csv_number(row["median_seconds"]),
            "p90_seconds": _csv_number(row["p90_seconds"]),
            "median_tokens": _csv_number(row["median_tokens"], 1),
            "time_overhead_ratio": _csv_number(row["time_overhead_ratio"], 6),
            "token_overhead_ratio": _csv_number(row["token_overhead_ratio"], 6),
            "time_overhead_vs_none": row["time_overhead_vs_none"],
            "token_overhead_vs_none": row["token_overhead_vs_none"],
        })
    return buffer.getvalue()


def summary_paths(jsonl_path):
    directory = os.path.dirname(os.path.abspath(jsonl_path)) or "."
    return (
        os.path.join(directory, "summary.md"),
        os.path.join(directory, "summary.csv"),
    )


def write_summary(path, allow_synthetic=False, output_dir=None):
    records, warnings = load_jsonl(path, allow_synthetic=allow_synthetic)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    if output_dir:
        md_path = os.path.join(output_dir, "summary.md")
        csv_path = os.path.join(output_dir, "summary.csv")
    else:
        md_path, csv_path = summary_paths(path)
    if not any(is_run_record(record) for record in records):
        markdown = render_empty_template()
        csv_text = render_csv({"rows": []})
    else:
        summary = summarize_records(records)
        markdown = render_markdown(summary, source=path)
        csv_text = render_csv(summary)
    for target, text in ((md_path, markdown), (csv_path, csv_text)):
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return md_path, csv_path


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Summarize a Safety Tax JSONL log. Does not call Ollama.")
    parser.add_argument("--input", default=os.path.join("results", "safety-tax", "runs.jsonl"))
    parser.add_argument("--md", default=None, help="Markdown output. Default: summary.md next to the log.")
    parser.add_argument("--csv", default=None, help="CSV output. Default: summary.csv next to the log.")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Permit a file marked synthetic_fixture. The report is bannered as a fixture.")
    args = parser.parse_args(argv)
    try:
        records, warnings = load_jsonl(args.input, allow_synthetic=args.allow_synthetic)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    md_path = args.md
    csv_path = args.csv
    if md_path is None or csv_path is None:
        default_md, default_csv = summary_paths(args.input)
        md_path = md_path or default_md
        csv_path = csv_path or default_csv
    if not any(is_run_record(record) for record in records):
        markdown = render_empty_template()
        csv_text = render_csv({"rows": []})
    else:
        summary = summarize_records(records)
        markdown = render_markdown(summary, source=args.input)
        csv_text = render_csv(summary)
    for target, text in ((md_path, markdown), (csv_path, csv_text)):
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
