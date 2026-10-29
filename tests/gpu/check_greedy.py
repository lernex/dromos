"""The engine's greedy output is the reference path's argmax.

  check_greedy.py MODEL_DIR BUILD_DIR [PROMPT_TOKENS] [GENERATED]

dromos_gpu_check runs the engine on a random prompt and generates GENERATED greedy tokens. dromos
eval then runs prompt + generated through the model in the reference way (the prompt in one
prefill, the rest one token at a time) and writes every position's logits. Each generated token
must be the argmax of the logits at the position before it. Exact ties may flip under float noise,
so the check reports every mismatch with its logit gap and fails only on a gap above 1e-3.
"""

import array
import json
import os
import random
import subprocess
import sys
import tempfile

model, build = sys.argv[1], sys.argv[2]
prompt_len = int(sys.argv[3]) if len(sys.argv) > 3 else 24
generated_len = int(sys.argv[4]) if len(sys.argv) > 4 else 8

rng = random.Random(7)
prompt = [rng.randrange(1000, 60000) for _ in range(prompt_len)]
ids = ",".join(str(t) for t in prompt)

out = subprocess.run(
    [os.path.join(build, "dromos_gpu_check"), model, ids, str(generated_len)],
    capture_output=True, text=True, check=True,
).stdout.strip()
generated = [int(t) for t in out.split(",")]
assert len(generated) == generated_len, out

sequence = prompt + generated
with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    with open(prompts, "w") as f:
        json.dump({"check": sequence}, f)
    subprocess.run(
        [os.path.join(build, "dromos"), "eval", model, prompts, tmp, str(prompt_len)],
        capture_output=True, text=True, check=True,
    )
    raw = open(os.path.join(tmp, "check.f32"), "rb").read()
    logits = array.array("f")
    logits.frombytes(raw)

vocab = len(logits) // len(sequence)
assert vocab * len(sequence) == len(logits)
mismatches = []
for k, token in enumerate(generated):
    t = prompt_len - 1 + k  # the row that predicts generated[k]
    row = logits[t * vocab : (t + 1) * vocab]
    best = max(range(vocab), key=lambda i: row[i])
    if best != token:
        gap = row[best] - row[token]
        mismatches.append((k, token, best, gap))

print(f"prompt {prompt_len} tokens, {generated_len} greedy tokens: {out}")
for k, token, best, gap in mismatches:
    print(f"  step {k}: engine {token}, reference argmax {best}, logit gap {gap:.5f}")
bad = [m for m in mismatches if m[3] > 1e-3]
print("PASS" if not bad else f"FAIL: {len(bad)} steps disagree beyond float noise")
sys.exit(1 if bad else 0)
