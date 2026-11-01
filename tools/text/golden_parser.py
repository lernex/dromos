#!/usr/bin/env python3
"""OutputParser goldens from the vLLM reference parser (metis tool and reasoning parsers).

    $VLLM_PY golden_parser.py OUT_DIR [--model DIR] [--fuzz N] [--seed S]
        with VLLM_TESTS pointing at vllm-metis/tests (its test_parser.py supplies CASES, TOOLS and the
        output renderer, so the goldens use exactly the reference cases).

Writes OUT_DIR/parser.jsonl: a header line {"vllm", "tools"} and one case per line:
  {"name", "thinking", "text", "ids"?, "expected": {"reasoning", "content", "tool_calls": [[name, args]]}}
or {"name", "thinking", "text", "error"} when the reference raises (the C++ side skips those).
"""
import argparse
import json
import os
import random
import sys
import types
from pathlib import Path


def load_reference(tests_dir, model_dir):
    """Imports vllm-metis's test_parser with its conftest stubbed to the given model folder."""
    sys.path.insert(0, str(tests_dir))
    conftest = types.ModuleType("conftest")
    conftest.model_dir = lambda: Path(model_dir)
    sys.modules["conftest"] = conftest
    import test_parser  # noqa: E402  (registers the metis parsers as a side effect)
    return test_parser


PIECES = [
    "<think>", "</think>", "\n", "\n\n", " ", "  ", "\t", "<|tool_call|>", "<function=", "<function=get_weather>",
    "<function=lookup>", "<function=unknown_fn>", "<parameter=", "<parameter=city>", "<parameter=days>",
    "<parameter=term>", "</parameter>", "</function>", ">", "Paris", "3", "hello", "é", "名", "<|fim_suffix|>",
    "<|endoftext|>", "<|padding|>", "<|tool_result|>", "{", "}", "\"", "'", "{\"name\": \"lookup\", \"arguments\": "
    "{\"term\": \"x\"}}", "<", "</", "<think", "</think", "<|tool_call", "<function", "</function", "<parameter",
    "=", "\\n", "true", "null", "[1, 2]", "1.5", "  \n", "x y", "🚀", "<|fim_middle|>", "\n</parameter>\n",
    "\n<parameter=city>\nParis\n</parameter>\n", "</function>\n</tool_call>", "<tool_call>",
]


def fuzz_texts(rng, count):
    for _ in range(count):
        parts = [rng.choice(PIECES) for _ in range(rng.randint(1, 12))]
        yield "".join(parts)


def run_parser(ref, tokenizer, thinking, text, ids):
    case = {"thinking": thinking, "reasoning": "", "content": "", "calls": []}
    parser, request = ref.make_parser(tokenizer, case)
    reasoning, content, calls = parser.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)
    got_calls = [[c.name, c.arguments] for c in (calls or [])]
    return {"reasoning": reasoning or "", "content": content or "", "tool_calls": got_calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=os.path.join(os.environ.get("LX_WORK", os.path.expanduser("~/lx-work")), "logos/engines/praxis/hf"))
    ap.add_argument("--tests", default=os.path.join(os.environ.get("LX_WORK", os.path.expanduser("~/lx-work")), "logos/engines/vllm-metis/tests"))
    ap.add_argument("--fuzz", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    ref = load_reference(args.tests, args.model)
    import vllm
    from vllm.tokenizers import get_tokenizer
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"cases": 0, "errors": 0}
    with open(out_dir / "parser.jsonl", "w", encoding="utf-8") as out:
        out.write(json.dumps({"vllm": vllm.__version__, "tools": ref.TOOLS}, ensure_ascii=True) + "\n")

        # 1. The reference's own cases: the rendered output, as the model would produce it.
        for name, case in ref.CASES.items():
            _, text, ids = ref.render(tokenizer, case, multi_turn=False)
            row = {"name": name, "thinking": case["thinking"], "text": text}
            try:
                row["expected"] = run_parser(ref, tokenizer, case["thinking"], text, ids)
            except Exception as exc:  # recorded; the C++ side skips these
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["errors"] += 1
            out.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            counts["cases"] += 1

        # 2. Fuzzed fragments of the markup, in both reasoning modes.
        rng = random.Random(args.seed)
        for i, text in enumerate(fuzz_texts(rng, args.fuzz)):
            thinking = rng.random() < 0.6
            ids = tokenizer.encode(text, add_special_tokens=False)
            row = {"name": f"fuzz{i}", "thinking": thinking, "text": text}
            try:
                row["expected"] = run_parser(ref, tokenizer, thinking, text, ids)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["errors"] += 1
            out.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            counts["cases"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    sys.exit(main())
