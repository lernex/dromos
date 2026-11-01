#!/usr/bin/env python3
"""Tokenizer goldens from Transformers, for the C++ parity tests.

    $PY golden_tokenizer.py MODEL_DIR OUT_DIR [--seed N] [--random N]

Writes into OUT_DIR:
  encode.jsonl  one case per line: {"text", "ids", "ids_nospecial"} (and "hex" for invalid bytes:
                the raw bytes, whose lossy text is "text")
  decode.jsonl  one case per line: {"ids", "text", "text_skip"}
  meta.json     vocabulary facts: size, added tokens, stop ids
  speed.txt     a mixed ~13 MB text for the encode speed measurement

Every line is ASCII (ensure_ascii), so the C++ side reads them with the same JSON parser it ships.
Transformers is the reference: ids come from AutoTokenizer, and the no-special-tokens ids come
from the same tokenizer.json with its added tokens removed, which is what parse_special=false means.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import tokenizers
import transformers
from transformers import AutoTokenizer
from tokenizers import Tokenizer as RawTokenizer

SEPARATOR = "\n\x1e\n"  # tok_texts.txt: entries are joined by this (cat -A shows the 0x1E as ^^)
WHITESPACE = [
    " ", "\t", "\n", "\r", "\v", "\f", "\u0085", " ", " ", " ", " ", " ",
    " ", " ", " ", " ", " ", " ", " ", " ", " ",
    " ", " ", " ", "　", "\u001c", "\u001d", "\u001e", "\u001f", "​",
    "﻿",
]
SCRIPTS = [
    (0x4E00, 0x9FFF), (0x3041, 0x3096), (0x30A1, 0x30FA), (0xAC00, 0xD7A3), (0x0600, 0x06FF),
    (0x0900, 0x097F), (0x0400, 0x04FF), (0x0370, 0x03FF), (0x0E01, 0x0E5B), (0x05D0, 0x05EA),
    (0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x0980, 0x09FF), (0x0B80, 0x0BFF), (0x1100, 0x11FF),
]
CJK_PUNCT = ["、", "。", "，", "！", "？", "「", "」", "：", "；", "（", "）"]
EMOJI = [
    "😀", "🚀", "🦙", "👍🏽", "👩‍👩‍👧‍👦", "🏳️‍🌈", "🇺🇸", "🇯🇵", "1️⃣", "❤️", "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "👨🏻‍💻", "🤦🏼‍♂️", "✅", "⚠️", "©", "™", "☀", "❤️", "🧑‍🤝‍🧑", "🫠", "🫨",
    "🇦", "\U0001F9EC", "\U0001FAE8",
]
MATH = [
    "\\frac{a}{b}", "x^2 + y^2 = z^2", "\\sum_{i=1}^{n} x_i", "∫_0^∞ e^{-x} dx", "α+β=γ", "√2 ≈ 1.414",
    "∀x∈ℝ: x²≥0", "ℕ⊂ℤ⊂ℚ⊂ℝ", "𝔸𝐱𝛼𝛽", "10⁻³ × 10⁶", "x₁ + x₂ ≠ 0", "∂f/∂x", "∇·F = 0", "∮ B·dl",
    "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}", "\\int_{0}^{\\pi} \\sin x\\,dx = 2", "n! / (k!(n-k)!)",
]
IDENTS = ["x", "value", "idx", "total_sum", "_tmp", "Ünïcode", "名前", "données", "count2", "fib"]
CODE_LINES = [
    "def {f}({a}, {b}):", "    return {a} if {a} < {b} else {f}({b} - 1) + {f}({a} - 2)",
    "{v} = [{a} for {a} in range({b}) if {a} % 3 == 0]", "class {f}(Base):", "    async def run(self, *args, **kw):",
    "print(f\"{{{v}}}: {a!r:>{b}}\")", "\tif ({a} >= 0x1F && {b} != 'é') {{ return; }}",
    "SELECT u.id, COUNT(*) FROM users u WHERE u.name LIKE '%{v}%' GROUP BY 1;",
    "const re = /^[\\w-]+@\\d{{2,}}$/i; // regex with unicode \\u00e9", "    }}", "x = {a} ** 2 // {b} >>= 3 ^ ~{a}",
    "<div class=\"{v}\">&amp; {a}</div>", "#include <vector>", "// TODO({v}): handle 名前 edge cases",
]
SPECIALS = [
    "<|endoftext|>", "<|padding|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|tool_call|>",
    "<|tool_result|>", "<think>", "</think>", "<think>\n", "</think>\n", "<function=", "</function>",
    "<parameter=", "</parameter>", "<|fim_suffix", "<|fim_suffix|", "<|tool_call", "<|endoftext|><|endoftext|>",
]
ALL_SPECIAL_LIKE = SPECIALS + ["<", ">", "|", "<|", "|>"]


def valid_cp(rng, lo, hi):
    while True:
        cp = rng.randrange(lo, hi)
        if not 0xD800 <= cp <= 0xDFFF:
            return chr(cp)


def multilingual(rng):
    parts = []
    for _ in range(rng.randint(1, 9)):
        r = rng.random()
        if r < 0.45:
            lo, hi = rng.choice(SCRIPTS)
            word = "".join(valid_cp(rng, lo, hi + 1) for _ in range(rng.randint(1, 9)))
            if rng.random() < 0.2:
                word += "́"  # combining acute on a letter
            parts.append(word)
        elif r < 0.6:
            parts.append(rng.choice(CJK_PUNCT))
        elif r < 0.75:
            parts.append(rng.choice(IDENTS))
        elif r < 0.85:
            parts.append(rng.choice(EMOJI))
        else:
            parts.append(rng.choice(WHITESPACE + [" ", "  "]))
    return rng.choice(["", " ", "\n"]).join(parts) + rng.choice(["", ".", "?", "!"])


def code(rng):
    lines = []
    ctx = {
        "f": rng.choice(IDENTS), "a": rng.choice(IDENTS), "b": rng.choice(["0", "1", "42", "7"]),
        "v": rng.choice(IDENTS),
    }
    for _ in range(rng.randint(1, 6)):
        lines.append(rng.choice(CODE_LINES).format(**ctx))
    return rng.choice(["\n", "\r\n"]).join(lines) + rng.choice(["", "\n", "\n\n"])


def math_text(rng):
    pieces = []
    for _ in range(rng.randint(1, 5)):
        pieces.append(rng.choice(["$", "$$", "\\(", "\\["]) + rng.choice(MATH) + rng.choice(["$", "$$", "\\)", "\\]"]))
        pieces.append(rng.choice([" ", "", ", ", "\n"]))
    return "".join(pieces)


def emoji_text(rng):
    return "".join(rng.choice(EMOJI) + rng.choice(["", " ", "‍", "️", "🏽"]) for _ in range(rng.randint(1, 12)))


def whitespace_text(rng):
    out = []
    for _ in range(rng.randint(2, 7)):
        out.append(rng.choice(["a", "b", ".", "1", "é", "名", "!"]))
        run = rng.randint(1, 40)
        out.append("".join(rng.choice(WHITESPACE) for _ in range(run)))
    out.append(rng.choice(["x", "", "\n"]))
    return "".join(out)


def long_word(rng):
    lo, hi = rng.choice(SCRIPTS + [(0x61, 0x7A)])
    return "".join(valid_cp(rng, lo, hi + 1) for _ in range(rng.randint(64, 2000)))


def random_cps(rng):
    return "".join(valid_cp(rng, 0, 0x110000) for _ in range(rng.randint(1, 24)))


def specials_text(rng):
    out = []
    for _ in range(rng.randint(1, 8)):
        out.append(rng.choice(ALL_SPECIAL_LIKE))
        out.append(rng.choice(["", " ", "\n", "hi", "x y", "名"]))
    return "".join(out)


def invalid_bytes(rng):
    raw = rng.randbytes(rng.randint(1, 48))
    return raw


GENERATORS = [multilingual, code, math_text, emoji_text, whitespace_text, long_word, random_cps, specials_text]


def mixed(rng):
    return "".join(rng.choice(GENERATORS)(rng) for _ in range(rng.randint(1, 4)))


def sweep_cases():
    """Every BMP code point (surrogates excluded) in four contexts that set its pre-token class."""
    contexts = [("a", "a"), (" ", ""), ("1", "1"), ("!", "!")]
    for cp in range(0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        for pre, post in contexts:
            yield pre + c + post


def supplementary_cases(rng, n):
    for _ in range(n):
        yield "a" + valid_cp(rng, 0x10000, 0x110000) + "a"


def build_texts(rng, corpus, random_cases):
    texts = []
    if corpus is not None and Path(corpus).exists():
        texts.extend(Path(corpus).read_text(encoding="utf-8").split(SEPARATOR))
    texts.extend([
        "", " ", "a", "\n", "\t", "é", "名", "😀", "<", ">", "<|", "|>",
        "Hello world", "  two spaces\tand a tab\n\nnewlines\r\n",
    ])
    texts.extend(SPECIALS)
    texts.extend(ALL_SPECIAL_LIKE)
    for _ in range(random_cases):
        r = rng.random()
        if r < 0.18:
            texts.append(rng.choice(GENERATORS)(rng))
        elif r < 0.9:
            texts.append(mixed(rng))
        else:
            raw = invalid_bytes(rng)
            texts.append(raw.decode("utf-8", errors="replace"))
    return texts


def encode_lines(hf, raw_nospecial, texts, hex_texts, out):
    batch = 4096
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        ids = hf(chunk, add_special_tokens=False)["input_ids"]
        ids_ns = [e.ids for e in raw_nospecial.encode_batch(chunk, add_special_tokens=False)]
        for text, a, b in zip(chunk, ids, ids_ns):
            out.write(json.dumps({"text": text, "ids": a, "ids_nospecial": b}, ensure_ascii=True, separators=(",", ":")) + "\n")
    for raw in hex_texts:
        text = raw.decode("utf-8", errors="replace")
        a = hf(text, add_special_tokens=False)["input_ids"]
        b = raw_nospecial.encode(text, add_special_tokens=False).ids
        out.write(json.dumps({"text": text, "hex": raw.hex(), "ids": a, "ids_nospecial": b},
                             ensure_ascii=True, separators=(",", ":")) + "\n")


def decode_lines(hf, rng, encoded_texts, out, count):
    vocab = len(hf)
    for i in range(count):
        r = i % 3
        if r == 0:
            ids = [rng.randrange(vocab) for _ in range(rng.randint(0, 40))]
        elif r == 1:
            text = rng.choice(encoded_texts)
            ids = hf(text, add_special_tokens=False)["input_ids"]
            if ids:
                lo = rng.randrange(len(ids))
                ids = ids[lo:lo + rng.randint(1, 30)]
        else:
            pool = list(range(0, 7)) + [rng.randrange(vocab) for _ in range(6)]
            ids = [rng.choice(pool) for _ in range(rng.randint(1, 16))]
        text = hf.decode(ids, skip_special_tokens=False)
        text_skip = hf.decode(ids, skip_special_tokens=True)
        out.write(json.dumps({"ids": ids, "text": text, "text_skip": text_skip}, ensure_ascii=True,
                             separators=(",", ":")) + "\n")


def speed_text(rng, model_dir):
    parts = []
    for name in ["README.md", "modeling_metis16.py", "conversion_receipt.json", "tokenizer.json"]:
        path = model_dir / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    mixed_parts = []
    total = 0
    while total < 8_000_000:
        piece = mixed(rng)
        mixed_parts.append(piece)
        total += len(piece.encode("utf-8"))
    parts.append("\n".join(mixed_parts))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--random", type=int, default=30000)
    ap.add_argument("--supplementary", type=int, default=20000)
    ap.add_argument("--corpus", default=os.path.join(os.environ.get("LX_WORK", os.path.expanduser("~/lx-work")), "logos/engines/tok_texts.txt"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    hf = AutoTokenizer.from_pretrained(str(model_dir))
    spec = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    added_spec = spec["added_tokens"]
    spec["added_tokens"] = []
    raw_nospecial = RawTokenizer.from_str(json.dumps(spec))
    spec["added_tokens"] = added_spec

    texts = build_texts(rng, args.corpus, args.random)
    texts.extend(sweep_cases())
    texts.extend(supplementary_cases(rng, args.supplementary))
    hex_texts = [invalid_bytes(rng) for _ in range(2000)]
    with open(out_dir / "encode.jsonl", "w", encoding="utf-8") as out:
        encode_lines(hf, raw_nospecial, texts, hex_texts, out)
    with open(out_dir / "decode.jsonl", "w", encoding="utf-8") as out:
        decode_lines(hf, rng, texts[: 60000], out, 30000)

    added = [
        {"id": int(entry["id"]), "content": entry["content"], "special": bool(entry["special"])}
        for entry in added_spec
    ]
    cfg = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    eos = cfg.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    meta = {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "vocab_size": len(hf),
        "added_tokens": added,
        "generation_eos_token_ids": eos,
        "tokenizer_eos_token_id": hf.eos_token_id,
        "encode_cases": len(texts) + len(hex_texts),
        "decode_cases": 30000,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")

    (out_dir / "speed.txt").write_text(speed_text(random.Random(args.seed + 1), model_dir), encoding="utf-8")
    print(json.dumps({"encode": len(texts) + len(hex_texts), "decode": 30000, "out": str(out_dir)}))


if __name__ == "__main__":
    sys.exit(main())
