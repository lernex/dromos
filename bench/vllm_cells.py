"""Per-repeat and per-cell numbers of one vLLM endpoint_bench run, with the server log's own counts beside them.

    python bench/vllm_cells.py RESULTS/TAG.vllm.json RESULTS/TAG.vllm.log

Each repeat's window is the host clock from its first submit to its last token (clock_start, clock_end in the JSON);
the server writes its log in the same clock. From the log, the peak Running and Waiting counts and the peak 10-second
generation throughput over that window. From the client, the peaks of its /metrics sampler, the most tokens in one
whole second (vllm bench serve's max_output_tokens_per_s), the overall output tokens per second (its
output_throughput), the mean TPOT, and both formulas: streams x 1000 / mean TPOT, and the peak resident count
x 1000 / mean TPOT. Cells are medians of the repeats.
"""

from __future__ import annotations

import json
import re
import statistics
import sys

# vLLM's periodic stat line: "INFO 10-28 08:55:05 [loggers.py:320] Engine 000: Avg prompt throughput: ..., ..."
STAT_RE = re.compile(
    r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine \d+: "
    r"Avg prompt throughput: [\d.]+ tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
)


def stat_lines(path: str) -> list[tuple[str, float, int, int, float]]:
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = STAT_RE.search(line)
            if m:
                out.append((m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
    return out


def in_window(stamp: str, start: str, end: str) -> bool:
    # HH:MM:SS strings on one host clock within one day compare correctly as text
    return start <= stamp <= end


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, spec: str = ".1f") -> str:
    return "n/a" if v is None else format(v, spec)


def main() -> None:
    result_path, log_path = sys.argv[1], sys.argv[2]
    data = json.load(open(result_path))
    stats = stat_lines(log_path)
    print(f"| streams | repeat | window (host clock) | log peak Running | log peak Waiting | log peak gen tok/s (10 s) "
          f"| sampler peak Running | sampler peak Waiting | peak output tok/s (1 s) | overall output tok/s "
          f"| mean TPOT ms | streams x 1000 / TPOT | running x 1000 / TPOT | decode tok/s (window) |")
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cells = []
    for n, rows in data["decode"].items():
        for row in rows:
            start, end = row["clock_start"], row["clock_end"]
            inside = [s for s in stats if in_window(s[0], start, end)]
            log_running = max((s[2] for s in inside), default=None)
            log_waiting = max((s[3] for s in inside), default=None)
            log_gen = max((s[1] for s in inside), default=None)
            print(f"| {n} | {row['repeat']} | {start}-{end} | {fmt(log_running, 'd') if log_running is not None else 'n/a'} "
                  f"| {fmt(log_waiting, 'd') if log_waiting is not None else 'n/a'} | {fmt(log_gen)} "
                  f"| {fmt(row.get('peak_running'), 'd') if row.get('peak_running') is not None else 'n/a'} "
                  f"| {fmt(row.get('peak_waiting'), 'd') if row.get('peak_waiting') is not None else 'n/a'} "
                  f"| {row['peak_output_tps_1s']} | {fmt(row['overall_output_tps'])} | {fmt(row['mean_tpot_ms'], '.2f')} "
                  f"| {fmt(row['concurrency_over_tpot_tps'])} | {fmt(row.get('running_over_tpot_tps'))} "
                  f"| {fmt(row['decode_tps'])} |")
            cells.append((n, {
                "log_running": log_running, "log_waiting": log_waiting, "log_gen": log_gen,
                "sampler_running": row.get("peak_running"), "sampler_waiting": row.get("peak_waiting"),
                "peak_1s": row["peak_output_tps_1s"], "overall": row["overall_output_tps"],
                "tpot": row["mean_tpot_ms"], "conc_formula": row["concurrency_over_tpot_tps"],
                "run_formula": row.get("running_over_tpot_tps"), "window": row["decode_tps"],
                "mismatch": row["event_vs_usage_mismatch_streams"], "short": row["streams_not_at_gen_tokens"],
            }))
    print()
    print("| streams | median log peak Running | median log peak Waiting | median sampler peak Running "
          "| median sampler peak Waiting | median peak output tok/s (1 s) | median overall output tok/s "
          "| median mean TPOT ms | median streams x 1000 / TPOT | median running x 1000 / TPOT | median window tok/s "
          "| mismatch | short |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in dict.fromkeys(c[0] for c in cells):
        rs = [c[1] for c in cells if c[0] == n]
        print(f"| {n} | {fmt(median([r['log_running'] for r in rs]), '.0f')} "
              f"| {fmt(median([r['log_waiting'] for r in rs]), '.0f')} "
              f"| {fmt(median([r['sampler_running'] for r in rs]), '.0f')} "
              f"| {fmt(median([r['sampler_waiting'] for r in rs]), '.0f')} "
              f"| {fmt(median([r['peak_1s'] for r in rs]), '.0f')} "
              f"| {fmt(median([r['overall'] for r in rs]))} | {fmt(median([r['tpot'] for r in rs]), '.2f')} "
              f"| {fmt(median([r['conc_formula'] for r in rs]))} | {fmt(median([r['run_formula'] for r in rs]))} "
              f"| {fmt(median([r['window'] for r in rs]))} "
              f"| {sum(r['mismatch'] for r in rs)} | {sum(r['short'] for r in rs)} |")


if __name__ == "__main__":
    main()
