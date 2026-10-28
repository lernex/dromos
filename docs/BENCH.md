# Batched serving and prefill: Lernex against vLLM

Status: partial. The vLLM cells, the prefix-cache check and the Logos FP8 runs at a 1024-token context and 0.96 GPU
memory are measured (the Logos section). The Lernex format sweep (every format, the serve view, prefill, spec-check and
the profile) waits for final main.

Measured 2026-10-28 on `ssh lernex`. Every GPU run went through the lock (`scripts/remote.sh bench x`).

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, sm_120. Spec 896 GB/s; 851 to 853 GB/s measured across the cache-check runs (the roofline line of the bench output) |
| Driver, CUDA | driver 595.91.07; CUDA 13.4 toolkit and runtime (driver API 13.2) |
| Host | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, 30 GB RAM. The server also runs CI runners, so each vLLM run is under a memory cap |
| Lernex | engine commit `71aa511`. Local main is 8 commits above origin/main `dd0fed5`: four engine and CLI changes (`88b182b`, `156341d`, `b1f9ff6`, `71aa511`) and four bench-driver commits. None is pushed. CMake's default build is `LX_ARCHS=120a-real` |
| vLLM | 0.31.0 from `~/venvs/vllm`, started by `~/venvs/vllm-metis/bin/python`. That venv's `.pth` adds the vLLM packages and loads the vllm-metis plugin |
| Client | `bench/endpoint_bench.py` (aiohttp), on the same host; one streamed `/v1/completions` per stream. `--metrics` polls the server's `/metrics` every 0.25 s for the running and waiting counts |

vLLM is started with the same flags for every format:

    vllm serve MODEL --served-model-name TAG --trust-remote-code --generation-config vllm \
      --no-enable-prefix-caching --max-model-len 9216 --max-num-seqs 64 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90

It runs in a transient systemd user scope with `MemoryMax=20G` and `MAX_JOBS=2`. Extra flags are appended after these,
so the Logos runs at other settings pass `--max-model-len`, `--gpu-memory-utilization` and, where noted,
`--max-num-batched-tokens` again. vLLM warns about the repeated keys, and its logged settings are the appended values.

| Format | Folder | Notes |
|---|---|---|
| Praxis BF16 | `~/kronos-scratch/logos/engines/praxis/hf` (model.safetensors, 6,162,552,620 bytes) | the folder Lernex loads too |
| Praxis FP8 | `~/kronos-scratch/logos/engines/praxis/fp8` | compressed-tensors. Its conversion receipt names source sha256 `9654d9c8...`, and the BF16 folder's config names `861af5db...`. The FP8 weights may not be the BF16 weights rounded. Speed is not affected; a quality comparison would be |
| Logos FP8 | `~/kronos-scratch/logos/hfport/fp8` | compressed-tensors, 11 GB |

Workload, the same for both engines:

- **Decode.** Each stream gets its own prompt from the generator in `cli/bench.cpp` (xorshift64 seeded per stream,
  ids in [1000, vocab - 1000)). 512 prompt tokens, 256 greedy tokens, `ignore_eos`. Each stream count runs a warm-up
  (min(prompt, 64) prompt tokens, 8 generated), then three measured repeats. Stream counts 1, 8, 32 and 64.
- **Prefill.** One request with exactly T prompt tokens and `max_tokens` 1, for T = 512, 2048 and 8192 (only 512 when
  the context is 1024). The prompt is the token sequences from the prompts file, concatenated and repeated to T. Median
  of five.
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

Logos FP8's KV cache is what limits vLLM here. At the default settings it admits 17 requests, so the 32- and 64-stream
cells measure a queue, not a decode rate. Each run's cell is the count the server actually held.

| run | context | GPU memory | 32 streams | 64 streams |
|---|---:|---:|---|---|
| parent (their harness, `vllm bench serve`) | 4096 | 0.92 | 29 running, 3 waiting, KV 99.5%; peak output 1,102 tok/s, overall 667 tok/s; mean TPOT 28.2 ms | not run |
| ours, default | 9216 | 0.90 | 17 running, 47 waiting (the peaks), KV 99.2%; the cells measure the queue | the same queue |
| ours, 0.96 at 9216 | 9216 | 0.96 | 26 running, 6 waiting, KV 100%; the third repeat ran out of GPU memory | not reached |
| ours, the requested flags (1024, 0.96) | 1024 | 0.96 | **26 of 32 resident, 1,023 tok/s**; the third repeat ran out of GPU memory | not reached |
| ours, 1024 at 0.96, `--max-num-batched-tokens 2048` | 1024 | 0.96 | **32 of 32 resident, 1,055.5 tok/s** | **35 of 64 resident, 1,048.7 tok/s** |

**The parent's run.** Their server log peaks at 29 running and 3 waiting, so 29 of the 32 streams were resident. Their
`vllm bench serve` output is kept as `parent-logos-fp8-ctx4096-u092.vllm-c32.json`, with its log as
`parent-logos-fp8-ctx4096-u092.vllm.log`. Its per-token formula at the nominal 32 streams is 1,135 tok/s (32,000 /
28.19 ms); at the 29 that were resident it is 1,029 tok/s (29,000 / 28.19 ms). Their harness is `vllm bench serve`, not
ours, so the comparison is the definitions, not the client.

**Ours, default (0.90, 9216).** The record, from the first Logos run (its harness did not sample the gauges):

| | 1 stream | 8 streams | 32 streams | 64 streams |
|---|---:|---:|---:|---:|
| Per-token formula | 117.9 | 600.1 | 1,559.6 (queued) | 2,907.3 (queued) |
| Window | 117.9 | 602.5 | 896.0 (queued) | 806.8 (queued) |
| TTFT p50 | 90 ms | 535 ms | 1,092 ms | 8,098 ms |

Prefill at 0.90: 6,141 / 7,587 / 8,047 tok/s at T = 512 / 2048 / 8192.

**Ours, 0.96 at 9216 (crashed).** The first two 32-stream repeats completed: window 452.5 and 452.3 tok/s, per-token
formula 1,268.2 and 1,268.0. The third repeat failed with `torch.OutOfMemoryError`: the engine asked for 160 MiB with
125 MiB free on the 15.47 GiB card. The client then reported "the stream carried no usage", because the engine core had
died. The cause is in the server log (`logos-fp8-u96-9216-oom.vllm.log.gz`), not in the client.

**Ours, the requested flags (1024 context, 0.96).** 26 of 32 streams resident, 1,023 tok/s, over the two repeats that
finished. N = 26 is the peak running count (the server's stat lines in the same repeats also read 26). X = N x 1000 /
mean TPOT.

| repeat | window tok/s | per-token formula, 32 streams | mean TPOT | 26 resident x 1000 / TPOT | overall output tok/s | peak output tok/s, 1 s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 453.8 | 1,259.2 | 25.41 ms | 1,023.1 | 644.1 | 962 |
| 1 | 453.9 | 1,260.2 | 25.39 ms | 1,023.9 | 664.6 | 962 |

The KV budget at 0.96 with the default 8192-token prefill batch is 26,770 tokens (2.77 GiB), which holds 26 requests of
1,024 tokens, so 32 streams cannot all be resident. The third repeat, at 32 streams, ran out of GPU memory:
`torch.OutOfMemoryError`, asking for 248 MiB with 233 MiB free. The 64-stream cell was not reached. The raw text is
`logos-fp8-ctx1024-u96-0.96-oom.vllm.txt`.

**Ours, 1024 context at 0.96 with `--max-num-batched-tokens 2048`.** This is the best-shot run: the same context and GPU
memory as the requested flags, with a smaller prefill batch. The smaller prefill batch cuts the activation peak that ran
out of memory, which leaves room for the KV cache to hold all 32 requests. It is a change from the requested flags, and
the requested-flags run above is kept beside it for that reason.

| streams | repeat | resident (log peak Running) | waiting (log peak Waiting) | window tok/s | per-token formula at resident N | mean TPOT | overall output tok/s | peak output tok/s, 1 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 32 | 0 | 1,143.1 | 1,056.3 | 30.29 ms | 855.1 | 1,168 |
| 32 | 1 | 32 | 0 | 1,142.6 | 1,055.5 | 30.32 ms | 892.0 | 1,152 |
| 32 | 2 | 32 | 0 | 1,142.0 | 1,054.9 | 30.33 ms | 891.4 | 1,152 |
| 64 | 0 | 35 | 29 | 1,057.4 | 1,048.7 | 33.38 ms | 873.1 | 1,190 |
| 64 | 1 | 35 | 29 | 1,057.5 | 1,048.7 | 33.38 ms | 873.2 | 1,155 |
| 64 | 2 | 35 | 29 | 1,057.1 | 1,047.9 | 33.40 ms | 872.7 | 1,165 |

The published cells, medians over the three repeats:

- **32 streams: 32 of 32 streams resident, 1,055.5 tok/s** (X = 32 x 1000 / 30.32 ms). Window 1,142.6 tok/s, overall output
  891.4 tok/s, peak one-second output 1,152 tok/s.
- **64 streams: 35 of 64 streams resident, 1,048.7 tok/s** (X = 35 x 1000 / 33.38 ms). Window 1,057.4 tok/s, overall output
  873.1 tok/s, peak one-second output 1,165 tok/s. The 64-stream per-token formula at the nominal 64, 1,917.5 tok/s, counts
  the 29 queued requests as decoding, so it is not published.

Against the parent's cell: 29 of 32 resident at 1,029 tok/s, a peak one-second output of 1,102 tok/s and an overall output
of 667 tok/s. Ours at 32 resident is 1,055.5 tok/s, a peak of 1,152 tok/s and an overall of 891 tok/s, on a 1024 context
against their 4096. The parent's mean TPOT is faster per stream (28.2 ms against 30.3 ms); the difference is in how many
streams are resident.

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
- **Logos Triton compiles during inference.** The 9216 Logos run logged two Triton JIT compiles while serving
  (`fused_moe_kernel` at 08:54:37, `_causal_conv1d_fwd_kernel` at 08:54:44), during its 8-stream cells. The 1024 run
  with the requested flags logged the same two at 09:07:25, as its 32-stream cell began; its repeat 0 has TTFT p50 2.03 s,
  against 1.63 s in repeat 1, while the decode windows match. The variant logged the same two at 09:16:32, one second
  before its first timed repeat, so they fall in that cell's warm-up, not in the timed repeats.
- **Duplicate flags.** vLLM warns "Found duplicate keys" for the appended `--max-model-len` and `--gpu-memory-utilization`.
  Its logged settings are the appended values (`max_seq_len=1024`, GPU memory 0.9600), so the last value wins. The
  variant also appends `--max-num-batched-tokens`, which is the same case.
- **The sampler's waiting count is not trusted.** The `/metrics` gauges agree with the server's stat lines on the running
  count (32 and 35). They do not agree on the waiting count: the sampler reports 27 waiting at 32 streams where the log
  reports 0, and 59 at 64 where the log reports 29. The cells here use the log's Waiting. The cause is not found.
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
    scripts/remote.sh bench x 'LX_VLLM_STREAMS=32,64 LX_VLLM_PREFILL=512 bash bench/run_bench.sh serve-vllm /home/aiadmin/kronos-scratch/logos/hfport/fp8 logos-fp8-ctx1024-u96 65536 --max-model-len 1024 --gpu-memory-utilization 0.96'
    scripts/remote.sh bench x 'LX_VLLM_STREAMS=32,64 LX_VLLM_PREFILL=512 bash bench/run_bench.sh serve-vllm /home/aiadmin/kronos-scratch/logos/hfport/fp8 logos-fp8-ctx1024-mbt2048-u96 65536 --max-model-len 1024 --gpu-memory-utilization 0.96 --max-num-batched-tokens 2048'
    scripts/remote.sh bench x 'bash bench/run_bench.sh cachecheck /home/aiadmin/kronos-scratch/logos/engines/praxis/hf praxis-bf16'
    scripts/remote.sh bench x 'bash bench/run_bench.sh cacheoff /home/aiadmin/kronos-scratch/logos/engines/praxis/hf praxis-bf16'

Raw outputs are in `bench/results/2026-10-28/`. `bench/make_tables.py` turns a results directory into the tables and
`summary.csv`. `bench/vllm_cells.py` prints each repeat's server-log and client counts for a vLLM run.
