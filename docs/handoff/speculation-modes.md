# Speculation modes (0.1.1)

Prompt-lookup speculation runs on a lone decoding stream. A verify step drafts up to six tokens from the stream's own history, checks them in one pass of the target, and keeps the accepted prefix. This file records which sampling modes, outputs and constraints speculate in 0.1.1, what still falls back to plain decoding, and where coverage is still thin.

## What speculates

**Sampling modes.** Each of these speculates on a lone stream, measured as tokens per decode step over the eight prompts of `all8.json` (plain decoding is 1.00):

| Sampling mode | Praxis BF16 | Praxis INT8 | Logos FP8 |
|---|---|---|---|
| Greedy | 1.79 | 1.95 | 1.56 |
| Temperature 0.7, top-p 0.9 | 1.77 | 1.86 | 1.46 |
| Think card: temperature 0.8, top-p 0.95, top-k 20, repetition 1.05 | 1.27 | 1.35 | 1.28 |
| Direct card: temperature 0.3 | 1.76 | 1.88 | 1.65 |
| min-p 0.05 at temperature 0.8 | 1.70 | 1.78 | 1.41 |
| top-k 40 at temperature 0.8 | 1.70 | 1.67 | 1.62 |

No prompt in any mode fell back to plain decoding. This shows that speculation engages; the speed gain is in the SCOREBOARD.

**Penalties.** Repetition, presence and frequency penalties are judged row by row, so a verify row sees the counts a plain step at its position would see. Exactness: a 2000-draw check with presence 2.0 and frequency 1.0 on Praxis BF16, every sample set within the reference's chi-square limit.

**Log-probabilities.** A verify step reports the numbers the plain step reports. The report reads the raw logits before penalties and masks, through the same reduction the plain sampler uses (`lx_spec_logprobs`). Checked bit for bit: greedy requests with logprobs and top-3 lists, speculation on against `--no-spec`, identical text and identical log-probabilities.

**JSON and tool grammars.** Each verify row is masked by the grammar's state after the drafts before it, on a copy of the grammar, and drafts stop before the first token the grammar rules out. Checked: a 2000-draw JSON exactness run, and tool-call requests whose greedy replies are identical with speculation on and off (call ids aside, which the server makes random per request).

## What still falls back to plain decoding

- **Experts offloaded.** A model that streams experts from host or disk runs without speculation. `serve` prints `speculation off: experts are offloaded` once at startup, and `doctor` says the same.
- **More than one stream.** Speculation runs for a lone stream. While several streams decode together, each step is plain. Multi-stream speculation is out of scope for 0.1.1. `serve` prints the lone-stream rule once at startup.
- **Models without a verify step.** `serve` prints `speculation off: this model has no verify step`.
- **`--no-spec`.** `serve` prints `speculation off: --no-spec`.
- **Steps without a draft.** A step where prompt lookup finds no match is a plain step. The drafter's adaptive length learns from it.
- **Near the request's end.** Drafts stop when the request's remaining tokens or the cache capacity run out.
- **Stop tokens in a draft.** A drafted stop token is cut from the draft; the verify row samples it instead.

## Coverage gaps

1. **Sampled tool-grammar exactness is unmeasured.** The exactness harness (`dromos spec-exactness`) takes a JSON schema only. Tool grammars are covered by the greedy server test, which shows identical replies with speculation on and off, but not by a distribution check under sampling.
2. **JSON exactness is checked on one enum schema.** The 2000-draw run uses `{"answer": enum ["yes","no","maybe"]}` after the prefix `{"answer": "`, which leaves 9 of 65,536 tokens allowed. That exercises the masking path hard, but says nothing about free strings, nested objects, arrays or numbers.

## Exactness notes

Each check is 2000 draws from one context on Praxis BF16 (prompt `spec_doc_extract`, temperature 0.8 unless stated). A movement figure shows how far the mode changes the reference distribution; a check is only informative when the mode moves it well past the noise between two plain sample sets (about 0.025 here).

| Mode | Movement | Result |
|---|---|---|
| min-p 0.05 (top-p 1.0, top-k off) | 0.033 | exact, but close to noise |
| min-p 0.3 (top-p 1.0, top-k off) | 0.190 | exact |
| top-k 40 (top-p 1.0, min-p off) | 0.0005 | exact, but the mode barely moves the distribution, so the run proves little |
| top-k 5 (top-p 1.0, min-p off) | 0.021 | exact, near noise |
| top-k 2 (top-p 1.0, min-p off) | 0.190 | exact |
| Repetition 1.0, presence 2.0, frequency 1.0, output-top 2 | 0.743 | exact |
| JSON enum schema after the prefix | 9 of 65,536 tokens kept | exact |

The informative checks are min-p 0.3, top-k 2, the penalty run and the JSON run. top-k 40 is not established by its own run; top-k 2 covers the top-k path.

## One-stream speed, for reference

Think mode at temperature 0.8, plain decoding against prompt lookup, from the SCOREBOARD:

| Model | Repetition | Plain tok/s | Prompt lookup tok/s | Ratio |
|---|---|---|---|---|
| Praxis BF16 | 1.0 | 319.8 | 385.5 | 1.21x |
| Praxis BF16 | 1.05 | 316.3 | 350.5 | 1.11x |
| Praxis INT8 | 1.0 | 366.1 | 423.7 | 1.16x |
| Praxis INT8 | 1.05 | 362.1 | 415.9 | 1.15x |
| Logos FP8 | 1.0 | 170.4 | 209.1 | 1.23x |
| Logos FP8 | 1.05 | 169.3 | 180.7 | 1.07x |

## Gates and commits

Each mode and output above cleared spec-check bit-identity on all five formats (Praxis BF16, INT8, FP8, NVFP4 GPTQ, Logos FP8) and the kernel suite on its commit.

- Penalties: `d952868`, with the sampling header at `a9a0e94`.
- Speculation tables in the SCOREBOARD: `e7299c5`.
- Exactness harness flags (`--temp`, `--top-p`, `--top-k`, `--min-p`, later `--json-schema` and `--json-prefix`): `c4194fa`.
- Log-probabilities: `a1bc78d` on main (`ef90c06` before the cherry-pick).
- Grammar masking: `2b67198`, stacked on main with the router change and Fold B.
- Startup note for `serve`: `e65c360`, stacked on the grammar commit.
