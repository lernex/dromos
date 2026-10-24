"""Token chunks for routing statistics, in the prompts.json format the engine's eval commands read.

    python tools/make_calib.py TOKENIZER_JSON TEXT OUT_JSON [CHUNKS] [TOKENS] [SKIP]

TEXT is tokenized whole and cut into CHUNKS chunks of TOKENS tokens, after skipping the first SKIP
chunks (so a held-out set can be cut from the same text). The calibration text the GGUF imatrix
used (prose, math in thinking mode, code and tool calls in the model's chat template) is the
natural input: routing statistics should come from the traffic the model will see.
"""

import json
import sys

from tokenizers import Tokenizer


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    text = open(sys.argv[2], encoding="utf-8").read()
    chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    size = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    skip = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    ids = tok.encode(text, add_special_tokens=False).ids
    out = {}
    for c in range(chunks):
        start = (skip + c) * size
        if start + size > len(ids):
            break
        out[f"calib_{skip + c:03d}"] = ids[start:start + size]
    json.dump(out, open(sys.argv[3], "w"))
    print(f"{len(ids)} tokens in the text; wrote {len(out)} chunks of {size}")


if __name__ == "__main__":
    main()
