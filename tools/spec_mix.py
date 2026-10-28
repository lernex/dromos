#!/usr/bin/env python3
"""The chat-mix measurement behind docs/SCOREBOARD.md's prompt-lookup rows.

Runs `lernex spec` on the chat mix with plain decoding and prompt-lookup drafts (ngram), then prints each drafter's
aggregate tokens per second over the mix and the lookup-to-plain ratio. The mix is seven prompts from the scoreboard's
prompt file: the five spec prompts, the chat think prompt and the direct tools prompt. Greedy by default; --temp
samples with the scoreboard's settings (top-p 0.95, top-k 20, seed 1).

    python3 tools/spec_mix.py MODEL_DIR PROMPTS_JSON --weights int8

The prompt-lookup rows in docs/SCOREBOARD.md are this command's output for their format. Quote the ratio this prints,
never a `lernex bench --spec` row: greedy continuation of a corpus repeats, and prompt lookup then matches every step.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# The chat mix. direct_prose is deliberately not in it: its repetition inflates lookup, and the scoreboard leaves it out
# of every headline.
MIX = {
    "spec_code_edit",
    "spec_doc_extract",
    "spec_doc_summary",
    "spec_multi_turn",
    "spec_prose",
    "think_chat",
    "direct_tools",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the model folder")
    ap.add_argument("prompts", help="the prompt file the mix is named in (a JSON object of token id lists)")
    ap.add_argument("--lernex", default=os.environ.get("LERNEX", "build/lernex"), help="the lernex binary")
    ap.add_argument("--weights", help="weight format (bf16, int8, fp8, nvfp4, bf16x); the folder's own by default")
    ap.add_argument("--tokens", type=int, default=512, help="tokens per prompt (512, the scoreboard's)")
    ap.add_argument("--repeat", type=int, default=2, help="runs per prompt, median reported (2)")
    ap.add_argument("--temp", type=float, help="sample at this temperature (greedy by default)")
    ap.add_argument("--rep", type=float, help="repetition penalty for both drafters (none by default)")
    ap.add_argument("--presence", type=float, help="presence penalty (none by default)")
    ap.add_argument("--frequency", type=float, help="frequency penalty (none by default)")
    ap.add_argument("--only", help="comma list: aggregate only these mix prompts (the scoreboard's think row is think_chat)")
    ap.add_argument("--json", help="keep lernex spec's per-prompt numbers in this file")
    args = ap.parse_args()
    mix = MIX if not args.only else set(args.only.split(","))
    if not mix <= MIX:
        sys.exit(f"--only names outside the chat mix: {sorted(mix - MIX)}")

    out = args.json or tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    cmd = [args.lernex, "spec", args.model, args.prompts, "--drafters", "plain,ngram", "--tokens", str(args.tokens),
           "--repeat", str(args.repeat), "--json", out]
    if args.weights:
        cmd += ["--weights", args.weights]
    if args.temp is not None:
        cmd += ["--temp", str(args.temp), "--top-p", "0.95", "--top-k", "20", "--seed", "1"]
    for flag, value in (("--rep", args.rep), ("--presence", args.presence), ("--frequency", args.frequency)):
        if value is not None:
            cmd += [flag, str(value)]
    subprocess.run(cmd, check=True)

    with open(out) as f:
        rows = [r for r in json.load(f) if r["prompt"] in mix]
    if not rows:
        sys.exit(f"no chat-mix prompts in {args.prompts}: the mix is {sorted(MIX)}")

    rates = {}
    for drafter in ("plain", "ngram"):
        rs = [r for r in rows if r["drafter"] == drafter]
        if not rs:
            sys.exit(f"no {drafter} rows for the mix")
        rates[drafter] = sum(r["emitted"] for r in rs) / sum(r["decode_s"] for r in rs)
    label = f"{args.weights or 'folder format'}, {'temp ' + str(args.temp) if args.temp is not None else 'greedy'}"
    what = "chat mix" if mix == MIX else "+".join(sorted(mix))
    print(f"{what} ({label}, {len(rows) // 2} prompts): plain {rates['plain']:.1f} tok/s, "
          f"prompt lookup {rates['ngram']:.1f} tok/s, ratio {rates['ngram'] / rates['plain']:.2f}x")


if __name__ == "__main__":
    main()
