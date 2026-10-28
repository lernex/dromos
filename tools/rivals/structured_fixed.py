#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Equal-length control for the structured-output cell in docs/RIVALS.md.

tools/rivals/structured.py reports the e2e rate of constrained and free replies. A free reply runs longer than
a constrained one (about 55 tokens against about 40 on this schema), so those two rates measure different
amounts of work. This client sends the same excerpts as free chat requests that each generate exactly --gen
tokens (ignore_eos), at the same concurrency. Its aggregate tok/s is the decode rate at the length the
constrained replies actually produce. Run it with --gen set to the constrained replies' mean length, and
compare its tok/s with the constrained wave of structured.py from the same server session: the ratio is the
cost of the constrained masks at equal length.

  python tools/rivals/structured_fixed.py --url http://127.0.0.1:8080 --model NAME \
      --tokenizer MODEL/tokenizer.json --concurrency 32 --requests 64 --gen 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aiohttp  # noqa: E402

from structured import excerpts, messages  # noqa: E402


async def one(session, base: str, model: str, text: str, gen: int) -> int:
    body = {
        "model": model,
        "messages": messages(text),
        "max_tokens": gen,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
    }
    async with session.post(base + "/v1/chat/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        data = await resp.json()
    return data["usage"]["completion_tokens"]


async def run(args) -> dict:
    base = args.url.rstrip("/")
    texts = excerpts(Path(args.tokenizer).expanduser(), args.requests, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as session:

        async def guarded(text):
            async with sem:
                return await one(session, base, args.model, text, args.gen)

        t0 = time.perf_counter()
        counts = await asyncio.gather(*(guarded(t) for t in texts))
        wall = time.perf_counter() - t0
    tokens = sum(counts)
    return {
        "url": base,
        "model": args.model,
        "mode": "free_fixed_length",
        "gen": args.gen,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "output_tokens": tokens,
        "min_tokens_per_reply": min(counts),
        "max_tokens_per_reply": max(counts),
        "wall_s": round(wall, 2),
        "e2e_tok_s": round(tokens / wall, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--gen", type=int, required=True, help="tokens every reply generates (ignore_eos)")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--out")
    args = p.parse_args()
    result = asyncio.run(run(args))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
