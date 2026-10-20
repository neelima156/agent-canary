"""Rate intervals, censored latencies, token coverage, and overhead bootstrap.

These functions summarize a log. They do not invent measurements.
"""
import math
import random
import statistics

from safety_tax.config import DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_BOOTSTRAP_SEED

# Standard normal quantile for a 95% two-sided interval.
WILSON_Z = 1.959963984540054


def number(value):
    """Finite int or float. Booleans are not counts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def wilson_interval(successes, total, z=WILSON_Z):
    """Wilson score interval for k successes in n trials. None when n is 0."""
    if total is None or int(total) <= 0:
        return None
    k = int(successes)
    n = int(total)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def format_rate(successes, total):
    """k/n plus a Wilson 95% interval. An empty cell is an em dash."""
    if total is None:
        return "\u2014"
    interval = wilson_interval(successes, total)
    if interval is None:
        return "\u2014"
    low, high = interval
    return f"{int(successes)}/{int(total)} [{low:.2f}, {high:.2f}]"


def censored_latency(record):
    """Return (seconds, censored).

    Errors that are not timeouts are excluded (None, False).
    A timeout is included at its cap. Call timeouts use call_timeout_s,
    run timeouts use run_timeout_s, and timeout_cap_s wins when the row
    recorded it. A row with no cap falls back to wall_seconds, which is
    how long the run lasted before the timeout was declared.
    """
    if record.get("error") and not record.get("timeout"):
        return None, False
    if record.get("timeout"):
        scope = record.get("timeout_scope")
        cap = number(record.get("timeout_cap_s"))
        if cap is None and scope == "run":
            cap = number(record.get("run_timeout_s"))
        if cap is None and scope == "call":
            cap = number(record.get("call_timeout_s"))
        if cap is None:
            cap = number(record.get("call_timeout_s"))
        if cap is None:
            cap = number(record.get("run_timeout_s"))
        if cap is None:
            cap = number(record.get("wall_seconds"))
        if cap is None:
            return None, False
        return cap, True
    wall = number(record.get("wall_seconds"))
    if wall is None:
        return None, False
    return wall, False


def percentile(sorted_values, fraction):
    """Linear percentile on a sorted non-empty list. fraction is in [0, 1]."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def bootstrap_ratio_ci(control, baseline, samples=DEFAULT_BOOTSTRAP_SAMPLES,
                       seed=DEFAULT_BOOTSTRAP_SEED, alpha=0.05):
    """Percentile bootstrap 95% CI for (median control - median baseline) / median baseline.

    Returns (low, high) or None when the ratio is undefined. The point
    estimate is computed separately from the full samples; this interval
    describes resampling variation of that median ratio.
    """
    if not control or not baseline:
        return None
    rng = random.Random(seed)
    ratios = []
    n_control = len(control)
    n_base = len(baseline)
    for _ in range(int(samples)):
        control_median = statistics.median(control[rng.randrange(n_control)] for _ in range(n_control))
        base_median = statistics.median(baseline[rng.randrange(n_base)] for _ in range(n_base))
        if not base_median:
            continue
        ratios.append((control_median - base_median) / base_median)
    if not ratios:
        return None
    ratios.sort()
    return percentile(ratios, alpha / 2.0), percentile(ratios, 1.0 - alpha / 2.0)


def run_token_summary(record):
    """Sum the counts that exist. One missing call does not drop the run.

    Timeout calls have no body and are not expected to carry a count.
    decode_tps is sum(eval_count) / sum(eval_duration) over calls where
    the duration is a positive nanosecond count. It is None when no such
    call exists, including when every duration is zero.
    """
    calls = [call for call in (record.get("model_calls") or []) if not call.get("timeout")]
    prompt_sum = 0
    eval_sum = 0
    prompt_known = 0
    eval_known = 0
    eval_for_rate = 0
    duration_ns = 0
    for call in calls:
        prompt = number(call.get("prompt_eval_count"))
        completion = number(call.get("eval_count"))
        duration = number(call.get("eval_duration"))
        if prompt is not None:
            prompt_sum += prompt
            prompt_known += 1
        if completion is not None:
            eval_sum += completion
            eval_known += 1
        if completion is not None and duration is not None and duration > 0:
            eval_for_rate += completion
            duration_ns += duration
    has_any = prompt_known or eval_known
    decode = None
    if duration_ns:
        decode = eval_for_rate / (duration_ns / 1e9)
    return {
        "total": (prompt_sum + eval_sum) if has_any else None,
        "eval": eval_sum if eval_known else None,
        "decode_tps": decode,
        "calls": len(calls),
        "eval_known": eval_known,
        "prompt_known": prompt_known,
    }
