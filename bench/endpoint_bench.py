"""Decode and prefill throughput of an OpenAI-compatible /v1/completions server, measured the way cli/bench.cpp
measures the engine in process, so `lernex serve` and vLLM are timed by the same client, with the same prompts.

    python bench/endpoint_bench.py --url http://127.0.0.1:8811 --model NAME --vocab 65536 \
        --streams 1,8,32,64 --prompt-tokens 512 --gen-tokens 256 --repeats 3 \
        --prefill 512,2048,8192 --prefill-runs 5 --prompts PROMPTS_JSON --metrics --out OUT.json

Decode (per stream count, per repeat), with the definitions of cli/bench.cpp:
  every stream gets the prompt cli/bench.cpp builds for it (xorshift64 seeded per stream, ids in
  [1000, vocab - 1000)), is sent as token ids, asks for exactly --gen-tokens greedy tokens with ignore_eos,
  and streams them. A warm-up batch (min(prompt, 64) prompt tokens, 8 generated) runs first, as the bench does.
  decode tok/s    tokens whose time is after every stream's first token, over the time from that point to the
                  last stream's last token
  per stream      decode tok/s divided by the streams
  TTFT p50/p90    from submit to the first token, over the streams
  prefill tok/s   prompt tokens of every stream, over the time from submit until the last stream's first token
Each streamed chunk that carries a token is one token event; the count per stream is checked against the
server's usage.completion_tokens (mismatch counts are reported, and must be zero for the run to be valid).

Output, in vllm bench serve's definitions, from the same token events:
  peak_output_tps_1s   the most tokens in one whole second counted from the first submit (its max_output_tokens_per_s)
  overall_output_tps   every output token over the time from the first submit to the last token (its output_throughput)
  running_over_tpot_tps  the formula with the count actually resident: peak_running x 1000 / mean TPOT
  --metrics polls the server's /metrics every 0.25 s for vllm:num_requests_running and _waiting while the cells run;
  peak_running and peak_waiting are the largest values seen between a repeat's submit and its last token.

Prefill (per T): one request of exactly T prompt tokens with max_tokens 1, the tokens cli/prefill.cpp builds
(the prompts JSON's sequences concatenated in key order, repeated to T). Runs after one warm-up; the median of
--prefill-runs is reported as T / TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time

import aiohttp

MASK64 = (1 << 64) - 1
# vLLM's gauges, one line per engine label set (summed over engines); a server without them yields no samples.
METRIC_RE = re.compile(r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})? +([0-9.eE+-]+)", re.M)


def bench_prompt(stream: int, n: int, vocab: int) -> list[int]:
    """The prompt cli/bench.cpp run_batch gives stream `stream`."""
    x = (0x9E3779B97F4A7C15 * (stream + 1)) & MASK64
    out = []
    for _ in range(n):
        x ^= (x << 13) & MASK64
        x ^= x >> 7
        x ^= (x << 17) & MASK64
        out.append(1000 + x % (vocab - 2000))
    return out


def prefill_tokens(prompts_path: str, T: int) -> list[int]:
    """The tokens cli/prefill.cpp builds: the JSON's sequences, in key order (a std::map), repeated to T."""
    with open(prompts_path) as f:
        data = json.load(f)
    text: list[int] = []
    for key in sorted(data):
        text.extend(int(t) for t in data[key])
    if not text:
        raise SystemExit("no tokens in the prompts")
    return [text[i % len(text)] for i in range(T)]


async def stream_one(session: aiohttp.ClientSession, url: str, body: dict) -> dict:
    """POST one streamed completion; returns token event times (perf_counter), usage and the finish reason."""
    events: list[float] = []
    empty_finish: list[float] = []  # a chunk with the finish reason and no text: a separate marker, or the last token
    usage = None
    finish = None
    async with session.post(url, json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            now = time.perf_counter()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                reason = choice.get("finish_reason")
                text = choice.get("text", "")
                if reason is not None:
                    finish = reason
                    if text == "":
                        empty_finish.append(now)
                        continue
                events.append(now)
    if usage is None:
        raise RuntimeError("the stream carried no usage (stream_options.include_usage is needed)")
    if finish is None:
        raise RuntimeError("the stream ended without a finish reason")
    # lernex serve ends a stream with an empty chunk that carries the reason, after its tokens; vLLM can put the
    # reason on the last token's chunk, and that token's text can be empty. The count decides which it was.
    if empty_finish and len(events) < usage["completion_tokens"]:
        events.append(empty_finish[0])
    return {"events": events, "usage": usage, "finish": finish}


def completion_body(name: str, prompt: list[int], max_tokens: int) -> dict:
    return {
        "model": name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def wait_ready(session: aiohttp.ClientSession, base: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            async with session.get(base + "/v1/models") as resp:
                if resp.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        if time.time() > deadline:
            raise SystemExit(f"the server at {base} did not come up within {timeout:.0f} s")
        await asyncio.sleep(2.0)


async def sample_metrics(session: aiohttp.ClientSession, base: str, samples: list, stop: asyncio.Event,
                         every: float = 0.25) -> None:
    """Polls the server's gauges until `stop` is set. Each sample is (perf_counter, running, waiting)."""
    while not stop.is_set():
        try:
            async with session.get(base + "/metrics") as resp:
                text = await resp.text()
            totals = {"running": 0.0, "waiting": 0.0}
            found = set()
            for kind, value in METRIC_RE.findall(text):
                totals[kind] += float(value)
                found.add(kind)
            if found == {"running", "waiting"}:
                samples.append((time.perf_counter(), int(totals["running"]), int(totals["waiting"])))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), every)
        except asyncio.TimeoutError:
            pass


def cell_metrics(results: list[dict], submitted: float, samples: list) -> dict:
    """The output and residency numbers of one repeat, from its token events and the sampled gauges."""
    stamps = [t for r in results for t in r["events"]]
    span = max(r["events"][-1] for r in results) - submitted
    buckets: dict[int, int] = {}
    for t in stamps:
        second = int(t - submitted)
        buckets[second] = buckets.get(second, 0) + 1
    inside = [s for s in samples if submitted <= s[0] <= submitted + span]
    return {
        "span_s": span,
        "output_tokens": len(stamps),
        "overall_output_tps": len(stamps) / max(span, 1e-9),
        "peak_output_tps_1s": max(buckets.values()),
        "metric_samples": len(inside),
        "peak_running": max((s[1] for s in inside), default=None),
        "peak_waiting": max((s[2] for s in inside), default=None),
    }


async def run_streams(session, url, name, streams, prompt_tokens, gen_tokens, vocab, samples=None) -> dict:
    prompts = [bench_prompt(i, prompt_tokens, vocab) for i in range(streams)]
    submitted = time.perf_counter()
    wall = time.time()
    results = await asyncio.gather(
        *(stream_one(session, url, completion_body(name, p, gen_tokens)) for p in prompts)
    )
    row = summarize(results, prompt_tokens, submitted, gen_tokens)
    row.update(cell_metrics(results, submitted, samples or []))
    row["running_over_tpot_tps"] = (
        row["peak_running"] * 1000.0 / row["mean_tpot_ms"] if row["peak_running"] else None
    )
    # Wall clock of this host, which is the clock the server writes its log in, for matching its stat lines.
    row["clock_start"] = time.strftime("%H:%M:%S", time.localtime(wall))
    row["clock_end"] = time.strftime("%H:%M:%S", time.localtime(wall + row["span_s"]))
    return row


def summarize(results: list[dict], prompt_tokens: int, submitted: float, gen_tokens: int) -> dict:
    n = len(results)
    if any(not r["events"] for r in results):
        raise RuntimeError("a stream produced no tokens")
    all_decoding = max(r["events"][0] for r in results)
    end = max(r["events"][-1] for r in results)
    ttft = sorted(r["events"][0] - submitted for r in results)
    tokens = sum(1 for r in results for t in r["events"] if t > all_decoding)
    window = max(end - all_decoding, 1e-9)
    decode = tokens / window
    mismatches = sum(1 for r in results if len(r["events"]) != r["usage"]["completion_tokens"])
    short = sum(1 for r in results if r["usage"]["completion_tokens"] != gen_tokens)
    # Mean time per output token after the first, per stream (vllm bench serve's TPOT), and the formula it is
    # often quoted with: concurrency x 1000 / mean TPOT. Reported beside decode_tps so the two can be compared.
    tpots = [(r["events"][-1] - r["events"][0]) / max(len(r["events"]) - 1, 1) for r in results]
    mean_tpot_ms = 1e3 * statistics.fmean(tpots)
    return {
        "streams": n,
        "decode_tps": decode,
        "per_stream_tps": decode / n,
        "ttft_p50": ttft[n // 2],
        "ttft_p90": ttft[min(n - 1, n * 9 // 10)],
        "prefill_tps": n * prompt_tokens / max(all_decoding - submitted, 1e-9),
        "window_s": window,
        "tokens_in_window": tokens,
        "mean_tpot_ms": mean_tpot_ms,
        "concurrency_over_tpot_tps": n * 1000.0 / mean_tpot_ms,
        "event_vs_usage_mismatch_streams": mismatches,
        "streams_not_at_gen_tokens": short,
        "finish": sorted({r["finish"] for r in results}),
    }


async def prefill_once(session, url, name, tokens: list[int]) -> float:
    submitted = time.perf_counter()
    result = await stream_one(session, url, completion_body(name, tokens, 1))
    return result["events"][0] - submitted


def save_partial(path: str, out: dict) -> None:
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


async def main_async(args) -> dict:
    base = args.url.rstrip("/")
    url = base + "/v1/completions"
    streams_list = [int(s) for s in args.streams.split(",") if s]
    prefill_list = [int(t) for t in args.prefill.split(",") if t] if args.prefill else []
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    connector = aiohttp.TCPConnector(limit=0)
    out = {
        "label": args.label,
        "url": base,
        "model": args.model,
        "vocab": args.vocab,
        "prompt_tokens": args.prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "repeats": args.repeats,
        "decode": {},
        "prefill": {},
    }
    samples: list = []
    stop = asyncio.Event()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await wait_ready(session, base, args.ready_timeout)
        sampler = asyncio.create_task(sample_metrics(session, base, samples, stop)) if args.metrics else None
        try:
            for n in streams_list:
                # Warm-up, as cli/bench.cpp does: min(prompt, 64) prompt tokens, 8 generated tokens.
                await run_streams(session, url, args.model, n, min(args.prompt_tokens, 64), 8, args.vocab)
                rows = []
                out["decode"][str(n)] = rows  # saved as each repeat finishes, so a later failure keeps them
                for r in range(args.repeats):
                    row = await run_streams(
                        session, url, args.model, n, args.prompt_tokens, args.gen_tokens, args.vocab, samples
                    )
                    row["repeat"] = r
                    rows.append(row)
                    save_partial(args.out, out)
                    print(
                        f"{args.label} streams={n:>3} rep={r} decode={row['decode_tps']:8.1f} tok/s "
                        f"per-stream={row['per_stream_tps']:7.1f} TTFT p50={row['ttft_p50']:.3f}s "
                        f"prefill={row['prefill_tps']:8.0f} tok/s tpot-formula={row['concurrency_over_tpot_tps']:8.1f} "
                        f"overall={row['overall_output_tps']:8.1f} peak1s={row['peak_output_tps_1s']} "
                        f"running={row['peak_running']} waiting={row['peak_waiting']} "
                        f"mismatch={row['event_vs_usage_mismatch_streams']} "
                        f"short={row['streams_not_at_gen_tokens']}",
                        flush=True,
                    )
            for T in prefill_list:
                tokens = prefill_tokens(args.prompts, T)
                await prefill_once(session, url, args.model, tokens)  # warm-up
                ms = []
                for _ in range(args.prefill_runs):
                    ms.append(await prefill_once(session, url, args.model, tokens))
                med = statistics.median(ms)
                out["prefill"][str(T)] = {
                    "ttft_median_s": med,
                    "runs_s": ms,
                    "prefill_tps": T / med,
                }
                print(f"{args.label} prefill T={T}: TTFT median {med * 1e3:.1f} ms, {T / med:.0f} tok/s", flush=True)
        finally:
            stop.set()
            if sampler is not None:
                await sampler
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="server origin, e.g. http://127.0.0.1:8811")
    ap.add_argument("--model", required=True, help="the served model name the server answers to")
    ap.add_argument("--vocab", type=int, required=True, help="vocab size of the model (config.json)")
    ap.add_argument("--streams", default="1,8,32,64")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prefill", default="", help="comma list of T for the prefill test (empty: none)")
    ap.add_argument("--prefill-runs", type=int, default=5)
    ap.add_argument("--prompts", default="", help="prompts JSON for the prefill tokens (cli/prefill.cpp)")
    ap.add_argument("--metrics", action="store_true",
                    help="poll the server's /metrics for the running and waiting request counts")
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.prefill and not args.prompts:
        ap.error("--prefill needs --prompts")
    result = asyncio.run(main_async(args))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
