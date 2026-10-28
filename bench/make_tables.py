"""Collects one results directory (written by bench/run_bench.sh on the server) into summary.csv and markdown tables.

    python bench/make_tables.py RESULTS_DIR            writes RESULTS_DIR/summary.csv and prints the markdown

Two throughput definitions, side by side, for every engine view:
  per-token     concurrency x 1000 / mean TPOT, where a stream's TPOT is its mean time per token after its first.
                THIS IS THE ONE THE PUBLISHED TABLE USES, for lernex and vLLM alike.
  window        cli/bench.cpp's decode tok/s: tokens generated after every stream has its first token, over the
                time to the last stream's last token.
Views: `bench` is `lernex bench` in process (three runs per format); `serve` is `lernex serve` through
endpoint_bench.py; `vllm` is vLLM through the same client (TAG.vllm.json). Each cell is the median of the repeats,
with the min-max range beside it.
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
    """One `lernex bench` output: {streams: {decode, per_stream, ttft_p50, ttft_p90, prefill, per_token}}.

    Row: streams, decode, per stream, TTFT p50 (value, unit), TTFT p90 (value, unit), prefill, then the roofline
    cells (roofline, % of spec, % of peak) and the per-token column, which is the last cell.
    """
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        tok = line.split()
        if len(tok) < 8 or not tok[0].isdigit():
            continue
        try:
            n = tok[0]
            rec = {
                "decode": float(tok[1]),
                "per_stream": float(tok[2]),
                "ttft_p50": seconds(tok[3], tok[4]),
                "ttft_p90": seconds(tok[5], tok[6]),
                "prefill": float(tok[7]),
            }
            if len(tok) >= 12:
                rec["per_token"] = float(tok[-1])
        except (ValueError, KeyError):
            continue
        rows[n] = rec
    return rows


def parse_prefill_txt(path: str):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"prefill (\d+) tokens: ([\d.]+) ms median of (\d+) \(min ([\d.]+)\), ([\d.]+) tok/s", text)
    if not m:
        return None
    return {"ms": float(m.group(2)), "tps": float(m.group(5))}


def parse_spec(path: str):
    """`lernex spec-check` lines: every line that reports rows bit-identical, as (name, identical, of)."""
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        m = re.search(r"(\d+) of (\d+) rows bit-identical", line)
        if m:
            out.append((line.split()[0], int(m.group(1)), int(m.group(2))))
    return out


def load_client(path: str):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def med_range(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:,.1f}"
    return f"{statistics.median(values):,.1f} ({min(values):,.1f}-{max(values):,.1f})"


def view_getters(bench, serve, vllm):
    """{view: {metric: f(streams) -> [values]}} for the decode metrics, for each view that exists."""
    out = {}
    if bench:
        out["bench"] = {
            "per_token": lambda n, b=bench: [r[n]["per_token"] for r in b if n in r and "per_token" in r[n]],
            "window": lambda n, b=bench: [r[n]["decode"] for r in b if n in r],
            "per_stream": lambda n, b=bench: [r[n]["per_token"] / int(n) for r in b if n in r and "per_token" in r[n]],
            "ttft_p50": lambda n, b=bench: [r[n]["ttft_p50"] for r in b if n in r],
        }
    for name, data in (("serve", serve), ("vllm", vllm)):
        if not data:
            continue
        out[name] = {
            "per_token": lambda n, d=data: [r["concurrency_over_tpot_tps"] for r in d["decode"].get(n, [])],
            "window": lambda n, d=data: [r["decode_tps"] for r in d["decode"].get(n, [])],
            "per_stream": lambda n, d=data: [r["concurrency_over_tpot_tps"] / int(n) for r in d["decode"].get(n, [])],
            "ttft_p50": lambda n, d=data: [r["ttft_p50"] for r in d["decode"].get(n, [])],
        }
    return out


def main() -> None:
    root = sys.argv[1]
    csv_rows = []
    md = []
    view_name = {"bench": "lernex bench (in process)", "serve": "lernex serve (client)", "vllm": "vLLM (client)"}
    for model, formats in FORMATS.items():
        loaded = {}
        for tag, label, _v in formats:
            bench = [parse_bench(p) for p in sorted(glob.glob(os.path.join(root, f"{tag}.bench.r*.txt")))]
            loaded[tag] = {
                "label": label,
                "getters": view_getters(bench, load_client(os.path.join(root, f"{tag}.serve.json")),
                                        load_client(os.path.join(root, f"{tag}.vllm.json"))),
                "serve": load_client(os.path.join(root, f"{tag}.serve.json")),
                "vllm": load_client(os.path.join(root, f"{tag}.vllm.json")),
                "prefill": {T: parse_prefill_txt(os.path.join(root, f"{tag}.prefill.T{T}.txt")) for T in PREFILL_T},
                "spec": parse_spec(os.path.join(root, f"{tag}.spec-check.txt")),
                "cache": parse_bench(os.path.join(root, f"{tag}.bench-prefixcache.txt")),
                "nocache": parse_bench(os.path.join(root, f"{tag}.bench-nocache.txt")),
            }
        md.append(f"\n### {model.capitalize()}\n")

        for metric, heading, note in (
            ("per_token", "Decode tok/s, per-token formula (published)",
             "concurrency x 1000 / mean TPOT, each stream's TPOT over its tokens after the first. Same definition for "
             "every row."),
            ("window", "Decode tok/s, window definition (cli/bench.cpp)",
             "tokens generated after every stream has its first token, over the time to the last stream's last token."),
        ):
            md.append(f"\n**{heading}.** Median of 3 (min-max). {note}\n")
            md.append("| Format | View | 1 stream | 8 streams | 32 streams | 64 streams |")
            md.append("|---|---|---:|---:|---:|---:|")
            for tag, label, _v in formats:
                info = loaded[tag]
                for view, getters in info["getters"].items():
                    cells = [med_range(getters[metric](n)) for n in STREAMS]
                    md.append(f"| {label} | {view_name[view]} | " + " | ".join(cells) + " |")
                    for n in STREAMS:
                        vals = getters[metric](n)
                        if vals:
                            csv_rows.append([model, tag, view, n, f"decode_{metric}", statistics.median(vals),
                                             min(vals), max(vals), len(vals)])

        md.append("\n**Per stream tok/s (per-token formula) and TTFT p50.**\n")
        md.append("| Format | View | per stream 1 | 8 | 32 | 64 | TTFT p50, 1 / 8 / 32 / 64 |")
        md.append("|---|---|---:|---:|---:|---:|---|")
        for tag, label, _v in formats:
            info = loaded[tag]
            for view, getters in info["getters"].items():
                ps = [med_range(getters["per_stream"](n)) for n in STREAMS]
                tt = []
                for n in STREAMS:
                    vals = getters["ttft_p50"](n)
                    tt.append("n/a" if not vals else f"{statistics.median(vals) * 1e3:,.0f} ms")
                md.append(f"| {label} | {view_name[view]} | " + " | ".join(ps) + " | " + " / ".join(tt) + " |")

        md.append("\n**Prefill tok/s by prompt length T.** In-process `lernex prefill` (one forward of T tokens), "
                  "and the client view as T / TTFT of a max_tokens-1 request.\n")
        md.append("| Format | View | T=512 | T=2048 | T=8192 |")
        md.append("|---|---|---:|---:|---:|")
        for tag, label, _v in formats:
            info = loaded[tag]
            cells = [f"{info['prefill'][T]['tps']:,.0f}" if info["prefill"][T] else "n/a" for T in PREFILL_T]
            md.append(f"| {label} | lernex prefill (in process) | " + " | ".join(cells) + " |")
            for view in ("serve", "vllm"):
                data = info[view]
                if not data:
                    continue
                cells = []
                for T in PREFILL_T:
                    rec = data.get("prefill", {}).get(T)
                    cells.append(f"{rec['prefill_tps']:,.0f}" if rec else "n/a")
                    if rec:
                        csv_rows.append([model, tag, view + "-prefill", T, "prefill_tps", rec["prefill_tps"], "", "", 1])
                md.append(f"| {label} | {view_name[view]}, TTFT | " + " | ".join(cells) + " |")
            for T in PREFILL_T:
                if info["prefill"][T]:
                    csv_rows.append([model, tag, "prefill-inprocess", T, "prefill_tps", info["prefill"][T]["tps"], "", "", 1])

        md.append("\n**Prefix cache check** (`lernex bench` default, prefix cache on, against `--no-prefix-cache`, "
                  "decode tok/s by view as above).\n")
        md.append("| Format | streams | prefill tok/s, cache on | prefill tok/s, cache off | TTFT p50, cache on | "
                  "TTFT p50, cache off | window decode tok/s, cache on | window decode tok/s, cache off |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for tag, label, _v in formats:
            info = loaded[tag]
            cache = info["cache"]
            nocache = info["nocache"]
            for n in STREAMS:
                if not cache or n not in cache or not nocache or n not in nocache:
                    continue
                md.append(f"| {label} | {n} | {cache[n]['prefill']:,.0f} | {nocache[n]['prefill']:,.0f} | "
                          f"{cache[n]['ttft_p50'] * 1e3:,.0f} ms | {nocache[n]['ttft_p50'] * 1e3:,.0f} ms | "
                          f"{cache[n]['decode']:,.1f} | {nocache[n]['decode']:,.1f} |")

        md.append("\n**Correctness** (`lernex spec-check`, verify steps bit-identical to one-token steps).\n")
        md.append("| Format | prompt | rows bit-identical |")
        md.append("|---|---|---:|")
        for tag, label, _v in formats:
            for name, ident, of in loaded[tag]["spec"] or []:
                md.append(f"| {label} | {name} | {ident} of {of} |")

    out_csv = os.path.join(root, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tag", "view", "streams_or_T", "metric", "value", "min", "max", "n"])
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\nwrote {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
