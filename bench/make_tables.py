"""Collects one results directory (written by bench/run_bench.sh on the server) into a CSV and markdown tables.

    python bench/make_tables.py RESULTS_DIR            writes RESULTS_DIR/summary.csv and prints the markdown

Engine rows come from `lernex bench` (three runs per format, in process) and `lernex prefill`; the two client views
come from endpoint_bench.py JSON: `lernex serve` (TAG.serve.json) and vLLM (TAG.vllm.json). Each cell is the median
of the repeats, with the min-max range beside it.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
import sys

STREAMS = ["1", "8", "32", "64"]
PREFILL_T = ["512", "2048", "8192"]

# (tag, label, has a vLLM equivalent) for each model; the labels are the table rows
FORMATS = {
    "praxis": [
        ("praxis-bf16", "Praxis BF16 (BF16X decode copies)", True),
        ("praxis-bf16-nobf16x", "Praxis BF16, plain (`--no-bf16x`)", True),
        ("praxis-int8", "Praxis INT8, engine's own 8-bit (`--weights int8`)", False),
        ("praxis-fp8", "Praxis FP8 checkpoint (compressed-tensors)", True),
        ("praxis-nvfp4", "Praxis NVFP4 (Metis GPTQ folder)", False),
    ],
    "logos": [
        ("logos-fp8", "Logos FP8 checkpoint (compressed-tensors)", True),
    ],
}

TIME_UNITS = {"ms": 1e-3, "s": 1.0, "min": 60.0}


def seconds(value: str, unit: str) -> float:
    return float(value) * TIME_UNITS[unit]


def parse_bench(path: str) -> dict:
    """One `lernex bench` output: {streams: {decode, per_stream, ttft_p50, ttft_p90, prefill}}."""
    rows = {}
    for line in open(path):
        tok = line.split()
        if len(tok) < 8 or not tok[0].isdigit():
            continue
        try:
            n = tok[0]
            decode, per = float(tok[1]), float(tok[2])
            ttft50 = seconds(tok[3], tok[4])
            ttft90 = seconds(tok[5], tok[6])
            prefill = float(tok[7])
        except (ValueError, KeyError):
            continue
        rows[n] = {"decode": decode, "per_stream": per, "ttft_p50": ttft50, "ttft_p90": ttft90, "prefill": prefill}
    return rows


def parse_prefill_txt(path: str):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"prefill (\d+) tokens: ([\d.]+) ms median of (\d+) \(min ([\d.]+)\), ([\d.]+) tok/s", text)
    if not m:
        return None
    return {"ms": float(m.group(2)), "tps": float(m.group(5))}


def load_client(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def med_range(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:,.1f}"
    return f"{statistics.median(values):,.1f} ({min(values):,.1f}-{max(values):,.1f})"


def main() -> None:
    root = sys.argv[1]
    csv_rows = []
    md = []
    for model, formats in FORMATS.items():
        md.append(f"\n### {model.capitalize()}\n")
        md.append("Decode tok/s, median of 3 (min-max). `bench` is `lernex bench` in process; `serve` is "
                  "`lernex serve` through the client; `vLLM` is vLLM through the same client.\n")
        md.append("| Format | View | 1 stream | 8 streams | 32 streams | 64 streams |")
        md.append("|---|---|---:|---:|---:|---:|")
        ttft_md = []
        prefill_md = []
        for tag, label, _has_vllm in formats:
            bench_files = sorted(glob.glob(os.path.join(root, f"{tag}.bench.r*.txt")))
            bench = [parse_bench(p) for p in bench_files]
            serve = load_client(os.path.join(root, f"{tag}.serve.json"))
            vllm = load_client(os.path.join(root, f"{tag}.vllm.json"))
            views = []
            if bench:
                views.append(("bench", lambda n, b=bench: [r[n]["decode"] for r in b if n in r], "in process"))
            if serve:
                views.append(("serve", lambda n, s=serve: [r["decode_tps"] for r in s["decode"].get(n, [])], "client"))
            if vllm:
                views.append(("vllm", lambda n, v=vllm: [r["decode_tps"] for r in v["decode"].get(n, [])], "client"))
            for view, getter, how in views:
                cells = [med_range(getter(n)) for n in STREAMS]
                md.append(f"| {label} | {view} | " + " | ".join(cells) + " |")
                for n in STREAMS:
                    vals = getter(n)
                    if vals:
                        csv_rows.append([model, tag, view, n, "decode_tps", statistics.median(vals),
                                         min(vals), max(vals), len(vals)])
            # per-stream and TTFT
            for view, getter_ps, getter_ttft in (
                ("bench", lambda n, b=bench: [r[n]["per_stream"] for r in b if n in r],
                 lambda n, b=bench: [r[n]["ttft_p50"] for r in b if n in r]),
                ("serve", lambda n, s=serve: [r["per_stream_tps"] for r in s["decode"].get(n, [])] if s else [],
                 lambda n, s=serve: [r["ttft_p50"] for r in s["decode"].get(n, [])] if s else []),
                ("vllm", lambda n, v=vllm: [r["per_stream_tps"] for r in v["decode"].get(n, [])] if v else [],
                 lambda n, v=vllm: [r["ttft_p50"] for r in v["decode"].get(n, [])] if v else []),
            ):
                if not getter_ps(STREAMS[0]):
                    continue
                ps = [med_range(getter_ps(n)) for n in STREAMS]
                tt = []
                for n in STREAMS:
                    vals = getter_ttft(n)
                    tt.append("n/a" if not vals else f"{statistics.median(vals) * 1e3:,.0f} ms")
                ttft_md.append(f"| {label} | {view} | " + " | ".join(ps) + " | " + " / ".join(tt) + " |")
                for n in STREAMS:
                    for v in getter_ttft(n):
                        csv_rows.append([model, tag, view, n, "ttft_p50_s", v, "", "", 1])
            # prefill
            ptxt = {}
            for T in PREFILL_T:
                p = parse_prefill_txt(os.path.join(root, f"{tag}.prefill.T{T}.txt"))
                if p:
                    ptxt[T] = p["tps"]
                    csv_rows.append([model, tag, "prefill-inprocess", T, "prefill_tps", p["tps"], "", "", 1])
            if ptxt:
                prefill_md.append(f"| {label} | lernex prefill (in process) | " + " | ".join(
                    f"{ptxt[T]:,.0f}" if T in ptxt else "n/a" for T in PREFILL_T) + " |")
            for view, data in (("serve", serve), ("vllm", vllm)):
                if not data:
                    continue
                cells = []
                for T in PREFILL_T:
                    rec = data.get("prefill", {}).get(T)
                    if rec:
                        cells.append(f"{rec['prefill_tps']:,.0f}")
                        csv_rows.append([model, tag, view + "-prefill", T, "prefill_tps", rec["prefill_tps"], "", "", 1])
                    else:
                        cells.append("n/a")
                prefill_md.append(f"| {label} | {view} (client TTFT, max_tokens 1) | " + " | ".join(cells) + " |")
        md.append("\nPer-stream decode tok/s, and TTFT p50 (1 / 8 / 32 / 64 streams, in the order of the table).\n")
        md.append("| Format | View | per stream 1 | 8 | 32 | 64 | TTFT p50 (ms) 1 / 8 / 32 / 64 |")
        md.append("|---|---|---:|---:|---:|---:|---|")
        md.extend(ttft_md)
        md.append("\nPrefill tok/s by prompt length T.\n")
        md.append("| Format | View | T=512 | T=2048 | T=8192 |")
        md.append("|---|---|---:|---:|---:|")
        md.extend(prefill_md)
    out_csv = os.path.join(root, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tag", "view", "streams_or_T", "metric", "value", "min", "max", "n"])
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\nwrote {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
