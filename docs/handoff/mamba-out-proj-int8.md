# Mamba out_proj at one byte per weight: measured, not wired

Question: can the Mamba out_proj (254 MB of BF16 per token, about 19% of a Praxis INT8 step) drop to one
byte per weight in the INT8 format, which today keeps it BF16? Answer: not at the group sizes tried.

## Method

- Load-time simulation, measurement only: each out_proj is rounded to a symmetric int8 grid, one f32 scale per
  G columns (G = 0 means one per row, today's per-row INT8 scheme), and held as the bf16 of the rounded values.
  This is the numerics a group-scaled int8 out_proj would have, so no kernel had to exist to measure it.
- The simulation reproduces the known per-row result (the 0.026 tools KL in the engine notes), so it is trusted.
- `dromos eval` on the Praxis BF16 folder with `--weights int8`, every prompt in `gates-prompts.json`
  (chat_think 16 tokens, chat_direct_tools 176, prose 300). Prefill is the whole prompt; decode is the first
  8 tokens as prefill, the rest one token at a time. `tools/compare.py` against the FP32 reference logits
  (`praxisrefs`), top-20 KL.

## Result

Current INT8, out_proj wide (BF16): think 0.00445 prefill / 0.00412 decode, tools 0.01559 / 0.01687,
prose 0.00310 / 0.00281.

Mean of the three prompts, as a ratio to wide (prefill / decode):

| out_proj | mean | think | tools | prose |
|---|---|---|---|---|
| per row (today's INT8 kernel scheme) | 1.66x / 1.56x | 1.07x / 1.13x | 1.89x / 1.70x | 1.38x / 1.35x |
| group 128 | 1.00x / 1.15x | 0.70x / 0.88x | 1.07x / 1.20x | 1.06x / 1.20x |
| group 64 | 1.41x / 1.40x | 0.99x / 1.12x | 1.51x / 1.46x | 1.51x / 1.42x |
| group 32 | 0.99x / 1.01x | 0.76x / 0.86x | 0.96x / 0.96x | 1.43x / 1.52x |

## Why it was dropped

Group 32 is at noise on the mean, but it moves prose (300 tokens, the only long prompt) 43-52% above wide, and
the top-token agreement falls from 99.0% to 98.3%. That is a regression, not noise, so out_proj stays BF16 in
INT8. Group 64 and 128 and per row fail on the mean as well. An FP8 block-scale variant was not run: at the
same byte budget it carries less precision per element than group-32 int8.

## Do not rerun without

A longer evaluation set than these three prompts (at least a few prose-length prompts), and a decision that
the 1.0-1.1 KL ratio applies per prompt rather than on the mean.
