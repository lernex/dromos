# Rivals: Dromos against each engine on its home turf

Same RTX 5070 Ti (16 GB), same Metis-1.6 weights, same client, same prompts. Each table cell reads
`Dromos X vs Rival Y (ratio)`. The rival side of every cell was measured here, pinned as listed
below. **The Dromos column is pending final main**: Dromos is measured once, at the end, on final
main, with the commands in the last section. Nothing in a Dromos column below was measured on this
doc's commit.

Status: rival numbers measured 2026-10-27/28 on the benchmark machine (Intel i5-14500, 14 physical cores,
30 GB RAM, RTX 5070 Ti 16 GB, Ubuntu). Raw results are in `tools/rivals/results/` and the server
logs stay in `~/lernex-engine-dev/rivals/out/rivals/` on the GPU server.

## Engines and versions

| Engine | Version and commit | Where it ran | Launch |
|---|---|---|---|
| SGLang | 0.5.22, with `sglang-metis` 0.1.0 (`sglang.srt.plugins`), xgrammar 0.2.7, torch 2.14.1, transformers 5.19.0, flashinfer 0.7.1 | `~/venvs/sglang` | `python -m sglang.launch_server --reasoning-parser metis --tool-call-parser metis` plus the flags in each table |
| llama.cpp | `metis16` branch at `07cacf7fa` (model: Add support for Metis-1.6 models), CUDA 13.4, sm_120, `GGML_CUDA_FA=ON` | `~/lernex-engine-dev/rivals-llama` (a `git archive` of that commit, built with `cmake -S . -B build-rivals -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF`, target `llama-bench`) | `llama-bench -p 512 -n 128,256 -r 3` (`-r 2` for the BF16 expert cells) |
| Dromos | main at `6b8731f` for the preliminary runs (not in the tables); final main is measured at the end | `~/lernex-engine-dev/rivals`, built with `scripts/remote.sh rivals build` | `build/dromos serve` |

A note on the llama.cpp build: the `build-cuda` directory already on the server was built from an
older commit (`065905776`), so the numbers here come from a fresh build at the pinned commit.

## Models and bits per weight

Bits per weight are the file size over every element in the file, so keep-wide tensors count. The
GGUF element counts come from the file headers.

| Model | File | Bytes | Bits per weight |
|---|---|---|---|
| Praxis (2.94 B weights) | GGUF Q4_K_M | 2.02 GB | 5.49 |
| Praxis | GGUF Q8_0 | 3.22 GB | 8.76 |
| Praxis | GGUF BF16 | 5.90 GB | 16.03 |
| Praxis | Dromos NVFP4 (`praxis/nvfp4`) | 2.01 GB | about 5.46 (same tensor set as the GGUF) |
| Praxis | FP8 checkpoint (`praxis/fp8`, both engines) | 3.53 GB | 9.6 (several tensors are kept BF16) |
| Logos (10.23 B weights) | GGUF Q4_K_M | 6.74 GB | 5.28 |
| Logos | GGUF Q8_0 | 11.11 GB | 8.69 |
| Logos | GGUF BF16 | 20.47 GB | 16.02 (does not fit the card) |
| Logos | FP8 checkpoint (`hfport/fp8`, both engines) | 11.66 GB | about 9.1 |

Dromos has no 4-bit Logos format: the closest Dromos format to Logos Q4_K_M is FP8, so the Logos
row compares FP8 with Q8_0 (8.7 bits) and says so. Dromos NVFP4 Praxis and llama.cpp Q4_K_M are
both about 5.5 bits per weight in the file, which is the fair pairing for 4-bit.

## Method

- `tools/rivals/load.py`: asyncio streaming client on `/v1/completions` with token-id prompts cut
  from `tools/rivals/corpus.txt` by each model's own `tokenizer.json`, so both engines see identical
  token ids. Greedy (temperature 0, repetition penalty 1, no penalties), `ignore_eos` set. A
  warm-up request fills the prefix cache first. Each wave sends all requests at once.
  - **e2e tok/s**: completion tokens of the wave over the time from sending the first request to the
    last finishing. Prefill and ramp are included: this is what a client waits for.
  - **TTFT p50 / p90**: send to first streamed chunk.
  - **stream tok/s**: per-request decode rate after the first token, median across the wave.
  - Two measured waves per level; the tables show both where they differ.
- Shared-prefix cells: 2,048-token prefix plus a 64-token unique suffix, 256 generated, 64 and 128
  requests in flight. Unique cells: independent 512-token prompts, 256 generated, 1, 8 and 32 in
  flight.
- `tools/rivals/structured.py`: chat requests with thinking off, 160 generated, 64 requests with 32
  in flight. Constrained replies use `response_format` `json_schema`, free replies do not. A reply is
  valid when it parses and matches the schema (title string, year integer, up to five tags).
- `tools/rivals/structured_fixed.py`: the equal-length control for that cell. The same excerpts as free
  requests that each generate exactly `--gen` tokens (`ignore_eos`), at the same concurrency.
- `tools/rivals/serve.sh`: starts one server, waits for `/v1/models`, runs the cells, stops the
  whole process group. Each GPU window holds the lock for one engine's cells, back to back.
- `tools/rivals/single.sh`: llama-bench and `dromos bench` single-stream and CPU cells.
- `tools/rivals/summarize.py`: prints the waves of any result file.

Dromos ignores `ignore_eos`, so a Dromos request can stop at an end token. The client records
`tokens_min` and `tokens_max` per wave. The final Dromos run must show 256 for every request or
say which waves did not, because a short request lowers Dromos's e2e number.

## 1. SGLang home turf

### 1a. Shared prefix (2,048 tokens shared, 64 unique, 256 generated)

SGLang flags, from the tuning below: Praxis (BF16 and FP8) `--max-mamba-cache-size 160
--max-total-tokens 131072 --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.8` (53 running
requests, a 16,838-token KV pool); Logos FP8 `--max-mamba-cache-size 24 --max-total-tokens 16384
--cuda-graph-max-bs-decode 8 --mem-fraction-static 0.85` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(8 running requests).

| Model and format | Streams | Dromos | SGLang e2e tok/s (waves) | SGLang TTFT p50 / p90 (ms, wave 0) | Ratio |
|---|---|---|---|---|---|
| Praxis BF16 | 64 | pending final main | 1607, 1609 | 258 / 6136 | pending |
| Praxis BF16 | 128 | pending final main | 1896, 1900 | 6122 / 12068 | pending |
| Praxis FP8 | 64 | pending final main | 1734, 1839 | 379 / 6258 | pending |
| Praxis FP8 | 128 | pending final main | 2134, 2161 | 5903 / 11481 | pending |
| Logos FP8 | 64 | pending final main | 434, 436 | 16717 / 33141 | pending |
| Logos FP8 | 128 | pending final main | 436, 436 | 35286 / 65837 | pending |

SGLang's radix cache is on in all of these. Its log shows `#cached-token: 2048` on the shared
requests, so the prefix is reused. Praxis BF16 held 53 running requests and the rest queued, which
is where the high p90 TTFT comes from. Per-stream decode at 64 streams is 44 tok/s for Praxis and
55 tok/s for Logos FP8.

### 1b. Structured output (JSON schema, 32 in flight, 64 requests, 160 generated)

Measured on commit `05d1855` (rebased onto `c7342e0`, which carries the lone-stream speculation), not final main. The server ran with `--sequences 32`, so all 32 requests are resident; the default plan held 31 on this GPU and queued the 32nd.

| Mode | SGLang e2e tok/s | Valid replies | Dromos |
|---|---|---|---|
| Free (no `response_format`) | 789 | 49 of 64 | 1,303 and 1,519 (two waves), 49 and 47 of 64 valid |
| `json_schema` (xgrammar) | 957 | 64 of 64 | 1,400 and 1,422 (two waves), 64 of 64 valid, 1.46 to 1.49 times SGLang |

Dromos enforces `response_format` `json_object` and `json_schema` on chat and completions. A schema
compiles to a byte-level pushdown automaton, every decode step masks the logits of the constrained
rows through `lx_mask_logits`, and the reply stops when its grammar completes. Compiling this schema
takes 0.015 milliseconds.

Constrained rows never speculate. Prompt-lookup speculation runs only for a lone stream, and
`spec_eligible` excludes a JSON-constrained row. In both json_schema waves the engine verified zero
drafts. The same server verified 7 to 9 drafts during each free wave's lone-stream tail, so the guard
is exercised, not idle.

Free and constrained e2e rates are not comparable as they stand: a free reply runs longer (54 to 56
tokens) than a constrained one (about 42). `tools/rivals/structured_fixed.py` is the equal-length
control: the same excerpts as free requests, each generating exactly 42 tokens. On this build the
json_schema waves run at 89.6 and 91.1 percent of that control (1,400 against 1,563 tok/s, and 1,422
against 1,561). **The 95 percent bar is not met on this e2e definition.**

The gap comes from steps, not the mask. Each wave was run alone with the engine's step counter read
around it: the json_schema waves took 115 steps for 2,681 tokens (23.3 tokens per step, 16.8 ms per
step), and the control took 89 steps for 2,688 tokens (30.2 per step, 19.4 ms per step). A json_schema
wave's replies vary in length, so the batch drains at the tail, where the control has none. The mask
build costs 0.15 milliseconds at p50 per step (0.43 at p99), about 1 percent of a step, measured
before its overlap with the forward pass; the accept side is not timed separately.

Before the rebase, the same control gave 95.9 percent (1,384 against 1,443 tok/s, second round, on
the default 31 slots). On the rebased build the free control rose to about 1,560 tok/s while the
constrained wave stayed near 1,400. The cause of the free rise was not isolated.

Each wave's first free request after server start ran at about half speed (725 tok/s against 1,501
warm), so the first wave is a warm-up wave. Of the constrained rows, 4.7 percent have exactly one
allowed token, two per reply. Strict jump-forward is not implemented: those steps are a small share,
and a jump would only merge adjacent forced tokens.

Logits and KL on the unconstrained path were checked before the rebase: prefill and decode logits
match the `orch` baseline byte for byte on all three reference prompts. `dromos spec-check` is
bit-identical on the rebased build on every row (40 of 40).

### 1c. Plain batched decode (independent 512-token prompts, 256 generated)

| Model and format | Streams | Dromos | SGLang e2e tok/s (waves) | SGLang stream tok/s | Ratio |
|---|---|---|---|---|---|
| Praxis BF16 | 1 | pending final main | 139 | 142 | pending |
| Praxis BF16 | 8 | pending final main | 674 (669, 679) | 88 to 89 | pending |
| Praxis BF16 | 32 | pending final main | 980 (979, 981) | 58 | pending |
| Praxis FP8 | 1 | pending final main | 140 | 142 | pending |
| Praxis FP8 | 8 | pending final main | 740 | 97 | pending |
| Praxis FP8 | 32 | pending final main | 1596 (1583, 1608) | 54 | pending |
| Logos FP8 | 1 | pending final main | 80 | 82 | pending |
| Logos FP8 | 8 | pending final main | 401 | 53 | pending |
| Logos FP8 | 32 | pending final main | 403 | 54 | pending |

At 32 streams, Praxis BF16 can only run about 21 requests at a time in the 16,838-token KV pool
(each request holds 768 tokens), so the 32-stream cell is memory-limited for SGLang.

### SGLang configuration on this card (what had to change, and why)

These are the runs that failed before the table flags were reached, so the next person does not
repeat them.

| Model | Attempt | Result |
|---|---|---|
| Praxis BF16 | `--mem-fraction-static` left at auto (0.748) | refused at start: weights leave no room for the KV pool |
| Praxis BF16 | mem 0.9, mamba 160, graph 32 | shared-prefix cell dies with CUDA out of memory in `copy_mamba_state` (1.25 GiB temporary) |
| Praxis BF16 | mem 0.85, same flags | same out-of-memory in the copy |
| Praxis BF16 | mem 0.8, graph 32 | runs, but 35 running requests exceed the graph cap, so decode runs without CUDA graphs: 813 to 896 tok/s at 64 streams, 17 tok/s per stream |
| Praxis BF16 | mem 0.8, graph 64 | **final**: 1608 and 1898 tok/s at 64 and 128 streams |
| Logos FP8 | flags from `sglang-metis/README.md` (mem 0.92, KV 32768) | out of memory in the Mamba copy (596 MiB) after the first shared-prefix requests |
| Logos FP8 | mem 0.85, KV 32768 | same out-of-memory (the pool is capped by `--max-total-tokens`, so the fraction changes nothing here) |
| Logos FP8 | mem 0.85, KV 16384, expandable segments | **final**: runs with 8 running requests |

The Mamba copy-on-write that the shared-prefix cells trigger needs roughly 0.6 to 1.3 GiB of free
memory at the moment of the copy. On a 16 GB card the defaults in the sglang-metis README leave less
than that, so the copy fails under shared-prefix load. These numbers are SGLang's best this
session found on this card, not its ceiling on a bigger one.

## 2. llama.cpp home turf

### 2a. Single stream on the GPU (full offload)

llama-bench, `-p 512 -n 128,256`, 3 repetitions, tok/s.

| Format (bits per weight) | pp512 | tg128 | tg256 | Dromos, same weight class | Ratio |
|---|---|---|---|---|---|
| Praxis Q4_K_M (5.49) | 3945 (+/- 197) | 252.4 | 253.3 | pending final main: Praxis NVFP4 (about 5.5) | pending |
| Praxis Q8_0 (8.76) | 4104 | 235.1 | 235.8 | pending final main: Praxis FP8 (9.6) | pending |
| Praxis BF16 (16.03) | 3471 | 184.5 | 185.2 | pending final main: Praxis BF16 | pending |
| Logos Q4_K_M (5.28) | 1868 | 124.4 | 124.9 | pending final main: no Dromos 4-bit Logos; FP8 is the nearest | pending |
| Logos Q8_0 (8.69) | 1897 | 115.0 | 115.5 | pending final main: Logos FP8 (about 9.1) | pending |

The Dromos side is `dromos bench` at one stream, 512 prompt and 256 generated tokens, through the
same card. Its earlier numbers (at 6b8731f) are in `tools/rivals/results/dromos-6b8731f-preliminary/`
and are not in these tables.

### 2b. CPU only (14 threads, no GPU)

llama-bench with `CUDA_VISIBLE_DEVICES=` and `-ngl 0 -t 14`, tok/s. Dromos's CPU backend is
selected with `LX_BACKEND=cpu` and `LX_THREADS=14`. The CPU backend implements BF16, INT8, FP8 and
NVFP4 (`engine/cpu/gemm.cpp`).

| Model | Format (bits per weight) | pp512 | tg128 | tg256 | Dromos CPU, same class | Ratio |
|---|---|---|---|---|---|---|
| Praxis | Q4_K_M (5.49) | 170.1 | 27.4 | 25.8 | pending final main: NVFP4 (about 5.5) | pending |
| Praxis | Q8_0 (8.76) | 127.3 | 22.4 | 23.6 | pending final main: INT8 (`--weights int8`, 8 bits plus a scale per row) | pending |
| Logos | Q4_K_M (5.28) | 75.4 | 12.8 | 12.6 | pending final main: no Logos NVFP4 file | pending |
| Logos | Q8_0 (8.69) | 45.6 | 12.3 | 12.6 | pending final main: Logos FP8 (about 9.1) | pending |

Caveat: these CPU cells ran through the no-lock `run` verb while other agents' jobs were starting.
The load average went from 2.9 to 10.6 during the window, and the Logos Q8_0 prefill spread is
large (45.6 +/- 23). A quiet-machine rerun belongs in the final pass.

### 2c. Logos BF16 partial offload (20.5 GB does not fit 16 GB)

No llama.cpp result. Every configuration tried failed while creating the context. The device
allocation that fails is the CUDA weight buffer, which the log reports at the start of each load:

| `-ngl` | Device buffer requested (MiB) | Result |
|---|---|---|
| 16 | 19318 | fails (`cudaMalloc`, then the compute reservation) |
| 20 | 19318 | fails |
| 22 | 18386 | fails |
| 24 | 16462 | fails |
| 28 | 12640 | fails |
| 30 | 10706 | fails |
| 32 | 8812 | fails |
| 34 | 6879 | fails |
| 36 | 5010 | fails (`failed to allocate compute pp buffers`, 5.25 GB) |

The requested device buffer shrinks as `-ngl` grows, which is the opposite of partial offload: in
this build the layer-to-device mapping for `metis16` looks inverted, so a larger `-ngl` puts fewer
layers on the card. The prompt compute reservation is 5 to 20 GB across the same range. Routing the
routed experts to the host (`-ngl 99 -ot "ffn_.*_exps=CPU"`, 18.1 GB of the 20.5 GB file) still
asks for a 17.3 GB device buffer at micro-batch 512, 128 and 64, so the micro-batch does not help.
Values from 37 to 40 and a host-side layer split were not tried; the upstream check comes first.

The logs are in `tools/rivals/results/single-logos-bf16-*.log`. This is a llama.cpp limit in this
build, not a Dromos loss. The Dromos side (routed experts offloaded automatically) is pending final
main.

## Cells we lose

Pending final main. The only losses known before the Dromos run, from the code and the rival side:

1. **Concurrency on 16 GB.** Dromos's default plan holds 37 BF16 sequences at 8,192 context, and
   21 Logos FP8 sequences at 2,560 context. SGLang holds 53 Praxis and 8 Logos. The Dromos final
   run must use the tuned settings below, or it under-reports the engine.

## Commands and reproduction

Lock windows, one engine each. Every GPU window has a `timeout`. The rivals slot is
`~/lernex-engine-dev/rivals`.

```bash
# rival side (already run; repeat only if the rival or the card changes)
scripts/remote.sh rivals x 'PORT=30000 timeout 1500 tools/rivals/serve.sh sglang sglang-praxis-bf16 \
  ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis \
  --max-mamba-cache-size 160 --max-total-tokens 131072 --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.8'
scripts/remote.sh rivals x 'PORT=30000 timeout 1500 tools/rivals/serve.sh sglang sglang-praxis-fp8 \
  $LX_WORK/logos/engines/praxis/fp8 --max-mamba-cache-size 160 --max-total-tokens 131072 \
  --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.8'
scripts/remote.sh rivals x 'PORT=30000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 1500 \
  tools/rivals/serve.sh sglang sglang-logos-fp8 ~/.cache/dromos/models/Lernex/Metis-1.6-Logos-FP8 \
  --max-mamba-cache-size 24 --max-total-tokens 16384 --cuda-graph-max-bs-decode 8 --mem-fraction-static 0.85'
scripts/remote.sh rivals x 'CELLS=structured PORT=30000 timeout 1500 tools/rivals/serve.sh sglang \
  sglang-praxis-bf16-structured ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis \
  --max-mamba-cache-size 160 --max-total-tokens 131072 --cuda-graph-max-bs-decode 64 --mem-fraction-static 0.8'
scripts/remote.sh rivals x 'timeout 900 tools/rivals/single.sh llama-gpu'
scripts/remote.sh rivals x 'timeout 900 tools/rivals/single.sh llama-gpu-bf16'
scripts/remote.sh rivals run 'timeout 2700 tools/rivals/single.sh llama-cpu'   # no GPU, no lock
scripts/remote.sh rivals x 'NGL="24 22 20 16" timeout 1800 tools/rivals/single.sh llama-offload'
scripts/remote.sh rivals x 'timeout 1800 tools/rivals/single.sh llama-experts'
```

Dromos, final main (to run once, on the lead's final commit; each window is one model and one
format, with the same client):

```bash
# BF16, the default plan and the tuned plans (the tuned ones lift the slot count for 64 and 128 streams)
scripts/remote.sh rivals x 'PORT=8080 timeout 1500 tools/rivals/serve.sh dromos dromos-praxis-bf16 \
  ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis'
scripts/remote.sh rivals x 'PORT=8080 timeout 1500 tools/rivals/serve.sh dromos dromos-praxis-bf16-nobf16x \
  ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis --sequences 128 --context 2560 --no-bf16x'
# FP8 and Logos FP8 (same client, tuned context)
scripts/remote.sh rivals x 'PORT=8080 timeout 1500 tools/rivals/serve.sh dromos dromos-praxis-fp8 \
  $LX_WORK/logos/engines/praxis/fp8 --sequences 128 --context 2560'
scripts/remote.sh rivals x 'PORT=8080 timeout 1500 tools/rivals/serve.sh dromos dromos-logos-fp8 \
  ~/.cache/dromos/models/Lernex/Metis-1.6-Logos-FP8 --sequences 128 --context 2560'
# structured output: json_schema on dromos serve, 64 of 64 valid expected
scripts/remote.sh rivals x 'CELLS=structured PORT=8080 timeout 600 tools/rivals/serve.sh dromos dromos-structured \
  ~/.cache/dromos/models/Lernex/Metis-1.6-Praxis'
# single stream and CPU
scripts/remote.sh rivals x 'timeout 1200 tools/rivals/single.sh dromos-gpu'
scripts/remote.sh rivals run 'timeout 2700 tools/rivals/single.sh dromos-cpu'
scripts/remote.sh rivals x 'timeout 1200 tools/rivals/single.sh dromos-offload'
```

Each Dromos window also records the plan lines (`Up to N sequences`) in
`out/rivals/<name>.server.log`, so the final table can say how many sequences each run held.
