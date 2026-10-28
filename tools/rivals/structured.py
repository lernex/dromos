#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structured-output cell for docs/RIVALS.md: JSON-schema constrained decoding against unconstrained.

Sends the same chat requests (thinking off, greedy, the same corpus excerpts) to an OpenAI-compatible
server, once with response_format json_schema ("constrained") and once without ("free"), and reports
the aggregate output tok/s, how many replies parse and satisfy the schema, and any HTTP error. An
engine that refuses response_format (lernex serve answers 400 "not supported yet") shows that as the
error count: the cell is a finding, not a number.

  python tools/rivals/structured.py --url http://127.0.0.1:30000 --model NAME \
      --tokenizer MODEL/tokenizer.json --concurrency 32 --requests 64 --out out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": ["title", "year", "tags"],
    "additionalProperties": False,
}


def excerpts(tokenizer_json: Path, n: int, seed: int) -> list[str]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    ids = tok.encode(CORPUS.read_text(encoding="utf-8"), add_special_tokens=False).ids
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = rng.randrange(0, len(ids) - 200)
        out.append(tok.decode(ids[s : s + 160]))
    return out


def messages(excerpt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Describe this excerpt as JSON with a title, the year it was written (0 if unknown), "
            "and up to five tags.\n\n" + excerpt,
        }
    ]


def valid(text: str) -> bool:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict) or set(obj) != {"title", "year", "tags"}:
        return False
    return (
        isinstance(obj["title"], str)
        and isinstance(obj["year"], int)
        and not isinstance(obj["year"], bool)
        and isinstance(obj["tags"], list)
        and len(obj["tags"]) <= 5
        and all(isinstance(t, str) for t in obj["tags"])
    )


async def one(session, base: str, model: str, text: str, constrained: bool, gen: int) -> dict:
    body = {
        "model": model,
        "messages": messages(text),
        "max_tokens": gen,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": False,
    }
    if constrained:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "excerpt", "schema": SCHEMA}}
    t0 = time.perf_counter()
    try:
        async with session.post(base + "/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
            data = await resp.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    content = data["choices"][0]["message"].get("content") or ""
    return {
        "seconds": time.perf_counter() - t0,
        "tokens": data["usage"]["completion_tokens"],
        "valid": valid(content),
        "finish": data["choices"][0].get("finish_reason"),
    }


async def wave(session, base, model, texts, constrained, gen, conc) -> dict:
    sem = asyncio.Semaphore(conc)

    async def guarded(text):
        async with sem:
            return await one(session, base, model, text, constrained, gen)

    t0 = time.perf_counter()
    recs = await asyncio.gather(*(guarded(t) for t in texts))
    wall = time.perf_counter() - t0
    ok = [r for r in recs if "error" not in r]
    errors = [r["error"] for r in recs if "error" in r]
    tokens = sum(r["tokens"] for r in ok)
    return {
        "mode": "constrained" if constrained else "free",
        "concurrency": conc,
        "requests": len(recs),
        "error_count": len(errors),
        "errors": errors[:2],
        "output_tokens": tokens,
        "wall_s": round(wall, 2),
        "e2e_tok_s": round(tokens / wall, 2) if ok else None,
        "valid": sum(r["valid"] for r in ok),
        "finish": sorted({r["finish"] for r in ok if r.get("finish")}),
    }


async def run(args) -> dict:
    import aiohttp

    base = args.url.rstrip("/")
    texts = excerpts(Path(args.tokenizer).expanduser(), args.requests, args.seed)
    results = {"url": base, "model": args.model, "gen": args.gen, "requests": args.requests, "waves": []}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as session:
        for constrained in (False, True):
            stats = await wave(session, base, args.model, texts, constrained, args.gen, args.concurrency)
            results["waves"].append(stats)
            print(json.dumps(stats), file=sys.stderr, flush=True)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--gen", type=int, default=160)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--out")
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
