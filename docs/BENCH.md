# Batched serving and prefill: Lernex against vLLM

Status: partial. The vLLM cells and the prefix-cache check are measured. The Logos FP8 cell at 0.96 GPU memory is queued
behind other jobs on the GPU lock and is not in this file yet. The Lernex format sweep (every format, the serve view,
prefill, spec-check and the profile) waits for final main.

Measured 2026-10-28 on `ssh lernex`. Every GPU run went through the lock (`scripts/remote.sh bench x`).

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, sm_120. Spec 896 GB/s; 851 to 853 GB/s measured across the cache-check runs (the roofline line of the bench output) |
| Driver, CUDA | driver 595.91.07; CUDA 13.4 toolkit and runtime (driver API 13.2) |
| Host | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, 30 GB RAM. The server also runs CI runners, so each vLLM run is under a memory cap |
| Lernex | engine commit `71aa511`. Local main is 8 commits above origin/main `dd0fed5`: four engine and CLI changes (`88b182b`, `156341d`, `b1f9ff6`, `71aa511`) and four bench-driver commits. None is pushed. CMake's default build is `LX_ARCHS=120a-real` |
| vLLM | 0.31.0 in `~/venvs/vllm-metis`, with the vllm-metis plugin |
| Client | `bench/endpoint_bench.py` (aiohttp), on the same host; one streamed `/v1/completions` per stream |

vLLM is started with the same flags for every format:

    vllm serve MODEL --served-model-name TAG --trust-remote-code --generation-config vllm \
      --no-enable-prefix-caching --max-model-len 9216 --max-num-seqs 64 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90

It runs in a transient systemd user scope with `MemoryMax=20G` and `MAX_JOBS=2`. Extra flags are appended after these.

| Format | Folder | Notes |
|---|---|---|
| Praxis BF16 | `~/kronos-scratch/logos/engines/praxis/hf` (model.safetensors, 6,162,552,620 bytes) | the folder Lernex loads too |
| Praxis FP8 | `~/kronos-scratch/logos/engines/praxis/fp8` | compressed-tensors. Its conversion receipt names source sha256 `9654d9c8...`, and the BF16 folder's config names `861af5db...`. The FP8 weights may not be the BF16 weights rounded. Speed is not affected; a quality comparison would be |
| Logos FP8 | `~/kronos-scratch/logos/hfport/fp8` | compressed-tensors, 11 GB |

Workload, the same for both engines:

- **Decode.** Each stream gets its own prompt from the generator in `cli/bench.cpp` (xorshift64 seeded per stream,
  ids in [1000, vocab - 1000)). 512 prompt tokens, 256 greedy tokens, `ignore_eos`. Each stream count runs a warm-up
  (min(prompt, 64) prompt tokens, 8 generated), then three measured repeats. Stream counts 1, 8, 32 and 64.
- **Prefill.** One request with exactly T prompt tokens and `max_tokens` 1, for T = 512, 2048 and 8192. The prompt is
  the token sequences from the prompts file, concatenated and repeated to T. Median of five.
- **Correctness.** Every streamed token is counted. Every vLLM decode run had zero mismatched and zero short streams.

## Two definitions of decode throughput

Both come from the same per-token timestamps, for both engines.

- **Per-token formula (published).** concurrency x 1000 / mean TPOT. A stream's TPOT is (t_last - t_first) / (tokens - 1),
  averaged over the streams.
- **Window (`cli/bench.cpp`).** Tokens generated after every stream has its first token, divided by the time from that
  point to the last stream's last token.

At one stream they agree. With concurrency they can split for two reasons. The ramp: when prefill is chunked, early
streams start decoding while later ones are still prefilling, and the formula counts that slower interval. Queueing:
when the server runs fewer requests than the stream count, the formula still multiplies by the full count. Logos FP8 at
0.90 shows the second case.

The tables publish the per-token formula for both engines, with the window beside it.

## Lernex's decode format

Lernex's default BF16 decodes from BF16X copies: a lossless encoding of the BF16 tensors, 4.47 GB more memory, with the
same logits to the bit (`lernex doctor` says so). vLLM has no BF16X, so the format-matched Lernex row is
`--no-bf16x`, which is pending. The Lernex numbers below are BF16X and are not set against vLLM's plain BF16 rows.

## Prefix cache

`lernex bench` keeps the prefix cache on by default, which a speed cell could benefit from. This check runs the same
prompts on Praxis BF16 (BF16X decode), once with the cache on and once with `--no-prefix-cache`.

| streams | prefill tok/s, cache on | prefill tok/s, cache off | TTFT p50, cache on | TTFT p50, cache off | window decode tok/s, cache on | window decode tok/s, cache off |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,978 | 12,811 | 39 ms | 40 ms | 303.5 | 303.5 |
| 8 | 16,396 | 15,610 | 209 ms | 221 ms | 1,226.6 | 1,223.0 |
| 32 | 15,108 | 16,643 | 592 ms | 569 ms | 2,217.5 | 2,214.8 |
| 64 | 13,662 | 16,326 | 1,200 ms | 1,000 ms | 3,151.5 | 3,149.9 |

The cache-on prefill is higher at 1 and 8 streams (by 1% and 5%) and lower at 32 and 64 (by 9% and 16%), so there is no
consistent inflation. Decode is within 0.3% at every stream count. Each cell is a single run, so differences of this
size are not resolved; the final sweep repeats these cells.

## vLLM: Praxis BF16 and FP8

Median of three runs, with the min-max range beside it.

**Decode tok/s, per-token formula (published).**

| Format | 1 stream | 8 streams | 32 streams | 64 streams |
|---|---:|---:|---:|---:|
| Praxis BF16, vLLM | 200.7 (200.7-200.8) | 992.6 (992.4-993.1) | 1,992.6 (1,992.1-1,993.1) | 2,668.7 (2,668.4-2,670.1) |
| Praxis FP8, vLLM | 234.6 (234.6-235.0) | 1,213.3 (1,213.1-1,213.4) | 2,422.7 (2,422.7-2,423.0) | 3,145.4 (3,144.0-3,145.4) |

**Decode tok/s, window definition.**

| Format | 1 stream | 8 streams | 32 streams | 64 streams |
|---|---:|---:|---:|---:|
| Praxis BF16, vLLM | 200.7 (200.7-200.8) | 998.4 (998.3-998.8) | 2,047.1 (2,046.4-2,047.3) | 2,898.1 (2,897.7-2,900.3) |
| Praxis FP8, vLLM | 234.6 (234.6-235.0) | 1,221.3 (1,220.8-1,221.4) | 2,496.6 (2,495.4-2,496.8) | 3,436.6 (3,435.3-3,437.4) |

**Per-stream rate (per-token formula) and TTFT p50.**

| Format | per stream, 1 / 8 / 32 / 64 | TTFT p50, 1 / 8 / 32 / 64 |
|---|---|---|
| Praxis BF16, vLLM | 200.7 / 124.1 / 62.3 / 41.7 | 49 ms / 277 ms / 1,009 ms / 1,806 ms |
| Praxis FP8, vLLM | 234.6 / 151.7 / 75.7 / 49.1 | 42 ms / 253 ms / 910 ms / 1,623 ms |

**Prefill tok/s (median of five one-token requests).**

| Format | T=512 | T=2048 | T=8192 |
|---|---:|---:|---:|
| Praxis BF16, vLLM | 12,119 | 14,398 | 15,945 |
| Praxis FP8, vLLM | 14,592 | 15,748 | 17,510 |

- At one stream both definitions give 200.7 tok/s for BF16. That matches the 201 that `docs/PLAN.md` gives for vLLM BF16
  at one stream. PLAN.md has 231 for FP8; this run measured 234.6.
- At 64 streams the per-token figure is 8% below the window. The BF16 log has one moment with 49 running and 15 waiting.
  KV space is not the limit there (the cache holds 540,452 tokens). The FP8 log never queued (max running 64, max
  waiting 0).

## vLLM: Logos FP8

At 0.90 GPU memory, Logos FP8's KV cache holds 74,342 tokens. vLLM reports 8.07 concurrent requests at its
9,216-token maximum, and the log peaks at 17 running with 47 waiting (KV usage 99.2%). At 32 and 64 streams, neither
definition measures decode at that concurrency, so those cells are shown only as the record.

| | 1 stream | 8 streams | 32 streams | 64 streams |
|---|---:|---:|---:|---:|
| Per-token formula | 117.9 | 600.1 | 1,559.6 (queued) | 2,907.3 (queued) |
| Window | 117.9 | 602.5 | 896.0 (queued) | 806.8 (queued) |
| TTFT p50 | 90 ms | 535 ms | 1,092 ms | 8,098 ms |

Prefill at 0.90: 6,141 / 7,587 / 8,047 tok/s at T = 512 / 2048 / 8192.

**0.96 rerun (pending).** The same run with `--gpu-memory-utilization 0.96` is queued behind other jobs on the GPU
lock. Its peak running count decides whether 32 and 64 streams are measurable at all; the numbers go here once it
finishes.

## Caveats

- **The Lernex rows are pending.** Every Lernex format, the `lernex serve` view, prefill, spec-check and the profile are
  measured on final main, not here.
- **BF16 against BF16X.** The Lernex prefix-cache numbers are BF16X, which decodes faster than plain BF16 at the same
  logits. The format-matched comparison needs `--no-bf16x`.
- **The completions stream changed.** Commit `156341d` sends one chunk per token on `/v1/completions`. The serve view's
  token count is the check that the client reads it correctly. It is not measured yet.
- **Start-up compiles.** vLLM compiles at engine start (flashinfer `fused_moe_120`, Triton kernels, "JIT kernel
  warmup"), before any measured cell. The first FP8 repeats at 1 and 8 streams had TTFT 421 ms and 759 ms, against 42 ms
  and 253 ms in the other repeats. The medians absorb this; decode is unaffected (235.0 against 234.6 tok/s).
- **Shutdown.** The Praxis FP8 server log ends with an `EngineDeadError` during shutdown. It came after the client had
  written its results, and the server was stopped on purpose.
- **NVFP4.** No vLLM cell. vLLM 0.31.0 does not load the Metis GPTQ NVFP4 folder.
- **INT8.** The engine's own 8-bit (round-to-nearest from the BF16 file at load). vLLM has no equivalent, so it is shown
  as the engine's own format and not as a win over FP8.
- **Shared GPU.** Each run holds `.gpu.lock`, and its wrapper waits until nvidia-smi lists no compute process before it
  starts. A process that does not take the lock can still start during a run. nvidia-smi is not logged per run, so a
  contaminated repeat would show only as a spread between repeats.
- **Client cost.** Not measured separately. The client runs on the same host as the server.
- **The other session's numbers.** Their vLLM results come from `vllm bench serve` with 32 prompts at concurrency 32 and
  4 at concurrency 1. Their mean TPOT at 32 streams is 17.1 ms, which gives 1,871 tok/s on the per-token formula. Ours is
  1,993 (6% higher). At one stream their 4.9 ms gives 204 tok/s, against our 200.7. Their output throughput (1,448 tok/s)
  includes the ramp, so it is not a decode rate.

## Pending (final main)

- Lernex: the BF16, `--no-bf16x`, INT8, FP8 and NVFP4 formats, and Logos FP8. Each runs `lernex bench` three times at 1,
  8, 32 and 64 streams, and `lernex prefill` at T = 512, 2048 and 8192, plus the `lernex serve` view through the client
  and `lernex spec-check` for each timed configuration.
- A repeat of the prefix-cache check, so the cache-on and cache-off cells get a spread.
- The profile of where the engine loses, for each cell where vLLM is ahead.

## See also

- `docs/SCOREBOARD.md`: the roofline and scoreboard (roofline agent).
- `docs/RIVALS.md`: the rival engines (rivals agent). Not on main yet.

## Reproduce

    scripts/remote.sh bench x 'bash bench/run_bench.sh serve-vllm /home/aiadmin/kronos-scratch/logos/engines/praxis/hf praxis-bf16 65536'
    scripts/remote.sh bench x 'bash bench/run_bench.sh serve-vllm /home/aiadmin/kronos-scratch/logos/engines/praxis/fp8 praxis-fp8 65536'
    scripts/remote.sh bench x 'bash bench/run_bench.sh serve-vllm /home/aiadmin/kronos-scratch/logos/hfport/fp8 logos-fp8 65536'
    scripts/remote.sh bench x 'bash bench/run_bench.sh cachecheck /home/aiadmin/kronos-scratch/logos/engines/praxis/hf praxis-bf16'
    scripts/remote.sh bench x 'bash bench/run_bench.sh cacheoff /home/aiadmin/kronos-scratch/logos/engines/praxis/hf praxis-bf16'

Raw outputs are in `bench/results/2026-10-28/`. `bench/make_tables.py` turns a results directory into the tables and
`summary.csv`.
