# Safety Tax results

No runs yet. This file is a template. Fill it by running the experiment; do not treat blanks as measurements.

Planned matrix: 3 tasks × 4 controls × 5 runs. The blank rows are the cells, not results.

n=5 is small for the default matrix (a smoke test with fewer runs is smaller). P99 is not meaningful at this sample size and is not reported. P90 is nearest rank: sort the timed runs and take rank ceil(0.90 × n), computed as (90 × n + 99) // 100. For 5 timed runs, P90 is the slowest run.

Medians and P90 use runs that finished without a timeout or error. Attack success, legitimate-task success, and false blocks count every logged run in the cell. A timeout cap is not a latency measurement, so timed-out runs stay out of the median. Where a median is labeled n=, that is how many runs entered the median.

Time overhead is (median time of this control − median time of none) / median time of none, on the same task and model. Token overhead uses the same formula on median tokens (sum of prompt_eval_count and eval_count across model calls, including the judge). The none row is the baseline, not a measured zero.

| Task | Control | Attack success | Legit success | False blocks | Median time (s) | P90 time (s) | Median tokens | Time overhead vs none | Token overhead vs none | Timeouts | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| summarize_injected | none | — | — | — | — | — | — | — | — | — | — |
| summarize_injected | prompt | — | — | — | — | — | — | — | — | — | — |
| summarize_injected | gate | — | — | — | — | — | — | — | — | — | — |
| summarize_injected | judge | — | — | — | — | — | — | — | — | — | — |
| summarize_clean | none | — | — | — | — | — | — | — | — | — | — |
| summarize_clean | prompt | — | — | — | — | — | — | — | — | — | — |
| summarize_clean | gate | — | — | — | — | — | — | — | — | — | — |
| summarize_clean | judge | — | — | — | — | — | — | — | — | — | — |
| write_notes | none | — | — | — | — | — | — | — | — | — | — |
| write_notes | prompt | — | — | — | — | — | — | — | — | — | — |
| write_notes | gate | — | — | — | — | — | — | — | — | — | — |
| write_notes | judge | — | — | — | — | — | — | — | — | — | — |
