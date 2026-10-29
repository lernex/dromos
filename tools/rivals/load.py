#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-work load generator for docs/RIVALS.md.

Sends identical token-id prompts to any OpenAI-compatible /v1/completions server (`dromos serve`,
`sglang.launch_server`) and reports, per wave, the numbers the comparison uses:

  e2e tok/s     completion tokens of the whole wave over the time from sending the first request to
                the last one finishing (prefill and ramp included: what a client waits for)
  TTFT p50/p90  time from sending a request to its first streamed chunk
  stream tok/s  per-request decode rate after the first token, median across the wave
  tokens        completion tokens per request (min..max), so every engine is seen doing the same work

Prompts are token ids cut from tools/rivals/corpus.txt with the model's own tokenizer.json, so the
engines never see different text, and they are identical for a given --seed.

  --shared-prefix N   every request is the same N-token prefix plus its own --suffix window (the
                      system-prompt case: prefix caches should hit)
  --shared-prefix 0   every request is an independent --prompt-token window (plain batched decode)

Every request is greedy (temperature 0), repetition penalty 1, no penalties, streamed, with usage.
`ignore_eos` asks the engine to run the full --gen tokens (SGLang honours it; dromos serve ignores
it, so check the per-wave "finish" counts: "length" for every request means the same work).

Examples (run on the GPU server, from the synced checkout; one server at a time):

  python tools/rivals/load.py --url http://127.0.0.1:8080 --model Praxis \\
      --tokenizer ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis/tokenizer.json \\
      --shared-prefix 2048 --suffix 64 --gen 256 --concurrency 64,128 --waves 2 --out out/praxis-64-128.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def corpus_ids(tokenizer_json: Path) -> list[int]:
    from tokenizers import Tokenizer  # the HF tokenizers package; same file the engines read

    tok = Tokenizer.from_file(str(tokenizer_json))
    text = CORPUS.read_text(encoding="utf-8")
    return tok.encode(text, add_special_tokens=False).ids


def build_prompts(ids: list[int], n: int, args: argparse.Namespace, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    if args.shared_prefix:
        need = args.shared_prefix + args.suffix + 1
        if len(ids) < need + args.suffix:
            raise SystemExit(f"corpus has {len(ids)} tokens; need at least {need + args.suffix}")
        prefix = ids[: args.shared_prefix]
        starts = [rng.randrange(args.shared_prefix, len(ids) - args.suffix) for _ in range(n)]
        return [prefix + ids[s : s + args.suffix] for s in starts]
    width = args.prompt_tokens
    if len(ids) <= width:
        raise SystemExit(f"corpus has {len(ids)} tokens; need more than {width}")
    if args.random_windows:
        starts = [rng.randrange(0, len(ids) - width) for _ in range(n)]
    else:  # the bench's rule: window i starts at (N - width) * i // (n - 1), so both read the same tokens
        span = len(ids) - width
        starts = [0] if n == 1 else [span * i // (n - 1) for i in range(n)]
    return [ids[s : s + width] for s in starts]


async def one_request(session, base: str, model: str, prompt: list[int], gen: int, t0: float) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    rec: dict = {"sent": time.perf_counter() - t0}
    try:
        async with session.post(base + "/v1/completions", json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return rec
            first = None
            end = None
            usage = None
            finish = None
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                now = time.perf_counter() - t0
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    if first is None:
                        first = now
                    end = now
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
            if usage is None or first is None:
                rec["error"] = "stream ended without usage or tokens"
                return rec
            rec.update(first=first, end=end, tokens=usage["completion_tokens"], prompt_tokens=usage["prompt_tokens"],
                       finish=finish)
            # Usage can count the prompt as cached (SGLang and Dromos both report it when a prefix hits).
            details = usage.get("prompt_tokens_details") or {}
            rec["cached_tokens"] = details.get("cached_tokens", 0)
    except Exception as exc:  # the wave records the failure and goes on
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


async def wave(session, base: str, model: str, prompts: list[list[int]], gen: int) -> dict:
    t0 = time.perf_counter()
    recs = await asyncio.gather(*(one_request(session, base, model, p, gen, t0) for p in prompts))
    ok = [r for r in recs if "error" not in r]
    errors = [r["error"] for r in recs if "error" in r]
    if not ok:
        return {"requests": len(recs), "errors": errors[:5], "error_count": len(errors)}
    total = sum(r["tokens"] for r in ok)
    wall = max(r["end"] for r in ok)
    ttfts = [r["first"] - r["sent"] for r in ok]
    per_stream = [(r["tokens"] - 1) / (r["end"] - r["first"]) for r in ok if r["tokens"] > 1 and r["end"] > r["first"]]
    return {
        "requests": len(recs),
        "errors": errors[:5],
        "error_count": len(errors),
        "output_tokens": total,
        "wall_s": round(wall, 3),
        "e2e_tok_s": round(total / wall, 2),
        "ttft_p50_ms": round(1000 * percentile(ttfts, 0.5), 1),
        "ttft_p90_ms": round(1000 * percentile(ttfts, 0.9), 1),
        "stream_tok_s_median": round(statistics.median(per_stream), 2) if per_stream else None,
        "tokens_min": min(r["tokens"] for r in ok),
        "tokens_max": max(r["tokens"] for r in ok),
        "finish": dict(Counter(r["finish"] for r in ok)),
        "cached_tokens_mean": round(statistics.mean(r["cached_tokens"] for r in ok), 1),
    }


async def run(args: argparse.Namespace) -> dict:
    import aiohttp

    ids = corpus_ids(Path(args.tokenizer).expanduser())
    base = args.url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    connector = aiohttp.TCPConnector(limit=0)
    results: dict = {
        "url": base,
        "model": args.model,
        "shared_prefix": args.shared_prefix,
        "suffix": args.suffix,
        "prompt_tokens": args.prompt_tokens if not args.shared_prefix else args.shared_prefix + args.suffix,
        "gen": args.gen,
        "seed": args.seed,
        "waves": [],
    }
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # One request first: it fills the prefix cache (shared mode) and the engine's first-step caches.
        warm = build_prompts(ids, 1, args, args.seed - 1)
        warm_rec = await one_request(session, base, args.model, warm[0], min(args.gen, 16), time.perf_counter())
        if "error" in warm_rec:
            raise SystemExit(f"warm-up failed: {warm_rec['error']}")
        results["warmup_tokens"] = warm_rec["tokens"]
        for conc in args.concurrency:
            for w in range(args.waves):
                prompts = build_prompts(ids, conc, args, args.seed + 1000 * conc + w)
                stats = await wave(session, base, args.model, prompts, args.gen)
                stats.update(concurrency=conc, wave=w)
                results["waves"].append(stats)
                print(json.dumps(stats), file=sys.stderr, flush=True)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="server root, e.g. http://127.0.0.1:8080")
    p.add_argument("--model", required=True, help="model name the server answers to")
    p.add_argument("--tokenizer", required=True, help="the model's tokenizer.json")
    p.add_argument("--shared-prefix", type=int, default=2048, help="shared prefix tokens (0: unique prompts)")
    p.add_argument("--suffix", type=int, default=64, help="unique suffix tokens after the shared prefix")
    p.add_argument("--prompt-tokens", type=int, default=512, help="prompt tokens in unique mode")
    p.add_argument("--random-windows", action="store_true",
                   help="unique mode: random window starts (the old behaviour); default is the bench's spread rule")
    p.add_argument("--gen", type=int, default=256, help="generated tokens per request")
    p.add_argument("--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[64, 128])
    p.add_argument("--waves", type=int, default=2, help="measured waves per concurrency level")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--out", help="write the full JSON here (default: stdout)")
    args = p.parse_args()
    results = asyncio.run(run(args))
    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
