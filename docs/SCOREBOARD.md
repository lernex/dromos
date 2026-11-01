# Decode scoreboard

How fast Dromos decodes, measured against the memory bandwidth of the card it runs on.

Card: NVIDIA GeForce RTX 5070 Ti (16 GB, sm_120), on the dromos GPU server.
Measured: 2026-10-27, on commit 4b25613 (`Add lx_bandwidth_probe and the decode roofline to doctor and bench`, landed on main as f1f60e3), under the GPU lock, on an otherwise idle card.
The per-component, held-as, modelled NVFP4 and Logos sections were measured 2026-10-28 with the doctor changes that come with this file (`dromos doctor --modelled`, the "held as" column, the routed-experts column in the batched table).

## What the ceiling is

- **Spec bandwidth: 896 GB/s**, from the memory clock (14001 MHz) x 2 transfers per clock x 256-bit bus / 8.
- **Measured bandwidth: 853 GB/s**, from `lx_bandwidth_probe`: a 1 GiB read, best of 5 runs. Four load variants all land between 850 and 853 GB/s.
- **Bytes per token**: every weight, scale and state byte one decode step reads. Counted from the loaded model (`engine/model/traffic.cpp`) and checked against a count from the config alone. The two agree on every row below.
- **Ceiling**: bandwidth divided by bytes per step. A batch of n streams reads the union of the routed experts they pick, so bytes per step grow with n.
- **Compute ceiling is not binding**: 87.9 TFLOPS BF16 tensor (dense, FP32 accumulate) over 2 x 1.20 GMAC per token gives about 36,700 tok/s for Praxis.

Bytes are counted at context 640 (512 prompt tokens plus half of the 256 generated: the mean over a run).

## Roofline, one stream

| Model | Format | Resident weights | Bytes per token | Ceiling at spec | Ceiling at measured | Measured | % of spec | % of measured |
|---|---|---|---|---|---|---|---|---|
| Praxis | BF16 | 10.36 GB | 1828 MB | 490 tok/s | 467 tok/s | 303.3 tok/s | 61.9% | 65.0% |
| Praxis | INT8 | 3.45 GB | 1337 MB | 670 tok/s | 638 tok/s | 345.9 tok/s | 51.6% | 54.3% |
| Praxis | FP8 | 3.70 GB | 1512 MB | 593 tok/s | 564 tok/s | 336.0 tok/s | 56.7% | 59.6% |
| Praxis | NVFP4 | 2.44 GB | 1296 MB | 692 tok/s | 658 tok/s | 358.3 tok/s | 51.8% | 54.5% |
| Logos | FP8 | 11.66 GB | 3456 MB | 259 tok/s | 247 tok/s | 159.6 tok/s | 61.5% | 64.7% |

Resident weights include the BF16X lossless copies that decode reads for the wide tensors.

Where each row's weights come from: Praxis BF16 is the BF16 file as stored. Praxis INT8 is round-to-nearest from the BF16 file, at load. Praxis FP8 and NVFP4 load the checkpoints stored under `~/lx-work/logos/engines/praxis/` (`fp8`, `nvfp4`); NVFP4 is refused from a plain BF16 checkpoint. Logos FP8 is `Metis-1.6-Logos-FP8` as stored.

## Bench prompts and the routing census

**The basis changed at commit 07db7ad.** From that commit `dromos bench` cuts its prompts from `tools/rivals/corpus.txt`, tokenized with the model's own tokenizer: 10,298 tokens, FNV-1a `df6a8031` (the same ids the rival client sends, checked against the HF tokenizer). Stream i of n reads the 512-token window that starts at `(N - 512) * i / (n - 1)`, so the windows spread across the corpus. The rival client's unique-prompt mode uses the same rule by default now (`--random-windows` restores its old random starts), so a rival run at n streams and a bench run at n streams read the same tokens. `--random-prompts` restores the bench's synthetic ids, which every row before 07db7ad was measured on.

The routing census runs on the same prompts as the timed run, eagerly (graphs off, so it is not timed). It runs after the timed run, so its prefix-cache hits do not turn the timed prefill into a cache read. The warm-up runs on random ids for the same reason.

Logos FP8 at 32 streams (512 prompt tokens, 256 generated, context 640):

| Prompts (commit) | Experts per layer-pass | Decode tok/s | Roofline at measured bandwidth | Share of roofline |
|---|---|---|---|---|
| Real text, corpus windows (07db7ad) | 67.8 | 1132.4 | 1711 | 66.2% |
| Random ids (f6e0eb3, `--random-prompts`) | 66.0 | 1200.6 | 1745 | 68.8% |
| Uniform, from the config (no run) | 94.1 | not timed | 1512 (table below) | not timed |

The real-text decode rate is 5.7% below the random-id rate, and the two censuses read almost the same union (67.8 and 66.0), so the union does not explain the gap. Run-to-run noise is about 3.5%; this is not yet explained beyond that.

An earlier real-text census (e8b7997) used a 300-token prose passage in 64-token windows and read 49.1 experts per layer-pass, with a roofline of 1946. That passage was too short and too short-windowed to be the bench's basis, so its figures are superseded by the corpus rows above.

## Measured decode: `dromos bench`

One run per format and stream count on commit 4b25613:

`dromos bench MODEL --streams 1,8,32 --prompt-tokens 512 --gen-tokens 256`

Greedy sampling, one card. Each cell is decode tok/s, with the share of the measured-bandwidth roofline at that batch size in brackets.

| Model | Format | 1 stream | 8 streams | 32 streams |
|---|---|---|---|---|
| Praxis | BF16 | 303.3 (65.0%) | 1223.0 (69.7%) | 2244.2 (75.2%) |
| Praxis | INT8 | 345.9 (54.3%) | 1419.4 (60.8%) | 2580.0 (67.6%) |
| Praxis | FP8 | 336.0 (59.6%) | 1446.7 (65.7%) | 2666.8 (71.6%) |
| Praxis | NVFP4 | 358.3 (54.5%) | 1561.5 (56.5%) | 2793.1 (59.4%) |
| Logos | FP8 | 159.6 (64.7%) | 682.4 (72.7%) | 580.4 (38.4%), queue-bound: 32 asked, 26 fit; see below. At 26 streams: 1001.9 (70.6%) |

Roofline at measured bandwidth for each batch size, tok/s (context 640):

| Model | Format | 1 stream | 8 streams | 32 streams |
|---|---|---|---|---|
| Praxis | BF16 | 467 | 1754 | 2985 |
| Praxis | INT8 | 638 | 2335 | 3818 |
| Praxis | FP8 | 564 | 2203 | 3727 |
| Praxis | NVFP4 | 658 | 2765 | 4699 |
| Logos | FP8 | 247 | 939 | 1512 |

### Logos FP8 at 32 streams

This row is not a decode rate. Logos fits 26 sequences of 784 tokens (14.68 GB planned of 14.78 GB available: its BF16X copies take the room the other 6 would need). Asking for 32 streams queues the last 6 requests.

`bench` times decode from the moment the last stream emits its first token. The queued streams therefore count against time to first token (p90 7.5 s) and the prefill rate (2090 tok/s). The decode window covers the tail, where the batch is no longer full. The 580 tok/s figure is a property of that tail, not of the engine at 32 streams.

Run again at 26 streams, the slot count, so nothing queues (`--streams 26`): **1001.9 tok/s** decode, 38.5 tok/s per stream, TTFT p50 1.4 s and p90 2.4 s (the 26 prompts prefill together), prefill 4983 tok/s, against a 1419 tok/s roofline at that batch size: 70.6% of the measured roofline, 67.2% of spec. The 580 tok/s at 32 was the queue, not the engine.

## Where the bytes go

One stream, context 512, MB per token, from `dromos doctor` on each format. BF16, FP8 and NVFP4 are the stored Praxis folders. INT8 is round-to-nearest from the BF16 file. The modelled FP8 and NVFP4 counts (`dromos doctor --modelled`) match the stored folders on all 38 components.

| Component | BF16 | INT8 | FP8 | NVFP4 |
|---|---:|---:|---:|---:|
| Mamba in_proj | 639.67 | 422.88 | 422.88 | 422.88 |
| Mamba out_proj | 253.62 | 253.62 | 253.62 | 253.62 |
| Mamba conv weights | 1.23 | 1.23 | 1.23 | 1.23 |
| Mamba gated norm and scan constants | 0.18 | 0.18 | 0.18 | 0.18 |
| attention qkv | 38.14 | 25.21 | 25.21 | 25.21 |
| attention out | 25.43 | 16.81 | 16.81 | 16.81 |
| attention pass LoRA down | 0.10 | 0.10 | 0.10 | 0.10 |
| attention pass LoRA up | 0.20 | 0.20 | 0.20 | 0.20 |
| Mamba state | 86.84 | 86.84 | 86.84 | 86.84 |
| attention cache read | 4.19 | 4.19 | 4.19 | 4.19 |
| attention cache write | 0.01 | 0.01 | 0.01 | 0.01 |
| hyper-connection norms | 0.20 | 0.20 | 0.20 | 0.20 |
| hyper-connection mixes | 0.00 | 0.00 | 0.00 | 0.00 |
| latent_down | 76.28 | 50.43 | 76.28 | 50.43 |
| latent_up | 76.28 | 50.53 | 76.28 | 76.28 |
| shared expert gate_up | 38.14 | 25.26 | 38.14 | 25.26 |
| shared expert down | 19.27 | 12.68 | 19.27 | 12.68 |
| router (f32) | 15.73 | 15.73 | 15.73 | 15.73 |
| router and selection bias | 0.02 | 0.02 | 0.02 | 0.02 |
| routed experts gate_up | 152.57 | 101.06 | 101.06 | 56.62 |
| routed experts down | 77.07 | 50.72 | 50.72 | 28.31 |
| expert embeddings | 0.05 | 0.05 | 0.05 | 0.05 |
| depth memory routing features | 64.38 | 42.63 | 64.38 | 42.63 |
| depth memory read query | 16.69 | 11.03 | 16.69 | 11.03 |
| depth memory read output | 17.20 | 11.18 | 17.20 | 11.18 |
| depth memory read gate | 0.34 | 0.34 | 0.34 | 0.34 |
| depth memory bank read | 0.13 | 0.13 | 0.13 | 0.13 |
| depth memory bank write | 0.01 | 0.01 | 0.01 | 0.01 |
| depth memory pass and anchor rows | 0.01 | 0.01 | 0.01 | 0.01 |
| depth memory entry state | 4.77 | 3.15 | 4.77 | 3.15 |
| depth memory entry metadata | 0.80 | 0.80 | 0.80 | 0.80 |
| depth memory entry key | 0.61 | 0.61 | 0.61 | 0.40 |
| depth memory entry value | 0.61 | 0.61 | 0.61 | 0.40 |
| depth head (f32) | 13.11 | 13.11 | 13.11 | 13.11 |
| embedding row | 0.00 | 0.00 | 0.00 | 0.00 |
| residual stream embeddings | 0.02 | 0.02 | 0.02 | 0.02 |
| final norm | 0.00 | 0.00 | 0.00 | 0.00 |
| LM head | 203.42 | 134.48 | 203.42 | 134.48 |
| **Total** | **1827.3** | **1336.1** | **1511.1** | **1294.6** |

Group totals, MB per token:

| Group | BF16 | INT8 | FP8 | NVFP4 |
|---|---:|---:|---:|---:|
| mixer | 958.6 | 720.2 | 720.2 | 720.2 |
| state and cache | 91.0 | 91.0 | 91.0 | 91.0 |
| moe | 455.4 | 306.5 | 377.6 | 265.4 |
| depth memory | 118.7 | 83.6 | 118.7 | 83.2 |
| head | 203.4 | 134.5 | 203.4 | 134.5 |
| connections | 0.2 | 0.2 | 0.2 | 0.2 |

Held as, where the four formats differ. BF16X is the lossless copy decode reads for a BF16 tensor, about 1.5 bytes per weight; FP4 is NVFP4 (four-bit values with one 8-bit block scale per 16 values); every other projection is read at one byte per weight plus its row scale.

| Component | BF16 | INT8 | FP8 | NVFP4 |
|---|---|---|---|---|
| Mamba in_proj | BF16X | INT8 | FP8 | FP8 |
| attention qkv | BF16X | INT8 | FP8 | FP8 |
| attention out | BF16X | INT8 | FP8 | FP8 |
| latent_down | BF16X | INT8 | BF16X | FP8 |
| latent_up | BF16X | INT8 | BF16X | BF16X |
| shared expert gate_up | BF16X | INT8 | BF16X | FP8 |
| shared expert down | BF16X | INT8 | BF16X | FP8 |
| routed experts gate_up | BF16X | INT8 | FP8 | FP4 |
| routed experts down | BF16X | INT8 | FP8 | FP4 |
| depth memory routing features | BF16X | INT8 | BF16X | FP8 |
| depth memory read query | BF16X | INT8 | BF16X | FP8 |
| depth memory read output | BF16X | INT8 | BF16X | FP8 |
| depth memory entry state | BF16X | INT8 | BF16X | FP8 |
| depth memory entry key | BF16X | BF16X | BF16X | FP8 |
| depth memory entry value | BF16X | BF16X | BF16X | FP8 |
| LM head | BF16X | INT8 | BF16X | FP8 |

Mamba out_proj is BF16X in all four formats: 253.6 MB per token, the largest tensor each quantized format keeps wide.

## Why FP8 reads more than INT8

Both formats read one byte per weight for every projection they quantize, and the same bytes for Mamba in_proj, attention qkv and out, and the routed experts. The difference is elsewhere. At context 512 (doctor) FP8 reads 1511.1 MB per token and INT8 reads 1336.1 MB, a gap of 175.0 MB. The bench's context-640 figures are 1512 and 1337.

The gap is nine tensors that the FP8 recipe keeps in BF16 and the INT8 recipe quantizes. A kept-wide tensor is read as its BF16X copy, which costs about 1.52 bytes per weight rather than 1: the LM head is 203.42 MB for 134.2 M weights in FP8 and 134.48 MB in INT8.

| Tensor (instances per token) | INT8, one byte | FP8, BF16X | FP8 minus INT8 |
|---|---:|---:|---:|
| LM head (1) | 134.48 | 203.42 | +68.94 |
| latent_down (24) | 50.43 | 76.28 | +25.85 |
| latent_up (24) | 50.53 | 76.28 | +25.75 |
| depth memory routing features (26) | 42.63 | 64.38 | +21.75 |
| shared expert gate_up (24) | 25.26 | 38.14 | +12.88 |
| depth memory read output (21) | 11.18 | 17.20 | +6.02 |
| shared expert down (24) | 12.68 | 19.27 | +6.59 |
| depth memory read query (21) | 11.03 | 16.69 | +5.66 |
| depth memory entry state (6) | 3.15 | 4.77 | +1.62 |
| **Total of the nine** | **341.37** | **516.43** | **+175.06** |

The FP8 recipe keeps these nine in BF16 because they are the LM head, the MoE latent and shared projections, and the depth memory's read and write projections. The rest of the kept-wide bytes are the same in both formats: Mamba out_proj (253.62 MB), the depth memory key and value (0.61 MB each), the metadata write (0.80 MB, plain BF16) and the pass LoRA (0.30 MB). Those total 256 MB in both.

So the two formats keep wide different amounts: FP8 reads 772 MB of its 1511 MB as BF16X or BF16 (51%), INT8 reads 256 MB of its 1336 MB (19%).

Two notes from the same data. The BF16X copy is what keeps FP8 at 1.52 bytes per weight on those nine rather than the 2 that raw BF16 would cost. And the FP8 folder stores the head twice, as separate `embedding.weight` and `lm_head.weight` BF16 tensors, where the NVFP4 folder ties them. The traffic count reads one head; whether the loader keeps both resident is a memory question this measurement does not answer.

## Which bytes NVFP4 actually 4-bits

Praxis NVFP4, one stream, context 512: 1294.6 MB per token, split by what the bytes are.

| Bytes | Format | MB per token | Share |
|---|---|---:|---:|
| Routed expert values | 4-bit e2m1, two per byte | 75.5 | 5.8% |
| Routed expert block scales and per-expert globals | 8-bit e4m3 per 16 values, plus one f32 global per expert matrix | 9.4 | 0.7% |
| Dense projections, LM head, depth memory | FP8 with a f32 row scale | 756.5 | 58.4% |
| Kept wide: Mamba out_proj, latent_up, pass LoRA down | BF16X | 330.0 | 25.5% |
| State, cache, routers, depth head and other plain tensors | f32, BF16 or as stored | 122.1 | 9.4% |
| Pass LoRA up and depth metadata write | BF16 | 1.0 | 0.1% |

Only the routed experts are 4-bit. Their values are 75.5 MB, 5.8% of the step, and their block scales add 9.4 MB. The split comes from the byte formula (values are n x k / 2, scales n x k / 16) and matches the doctor totals: the gate_up stack is 50.3 MB of values and 6.3 MB of scales, the down stack 25.2 and 3.2.

Against FP8 the step is 216.5 MB smaller at context 512 (1511.1 to 1294.6). Routed experts account for 66.9 MB of that (151.8 MB in FP8, 84.9 MB in NVFP4, 31%). The other 149.7 MB (69%) is the dense tensors NVFP4 quantizes to FP8 and FP8 keeps wide: the LM head (68.9), latent_down (25.9), the shared expert (19.5), the depth memory routing features, query and output (33.4), the state write (1.6) and the key and value (0.4). So at one stream most of NVFP4's gain over FP8 on this model is not the experts.

## Logos NVFP4, modelled

No Logos NVFP4 checkpoint exists. This is the recipe the Praxis NVFP4 folder uses (`projection_format` in `engine/core/checkpoint.cpp`: experts NVFP4; dense projections, LM head and depth memory key and value FP8; latent_up and Mamba out_proj BF16; router f32) applied to Logos's config, counted with `dromos doctor --modelled --weights nvfp4`. Nothing is read from the folder.

The same command on Praxis reproduces the stored Praxis NVFP4 folder on all 38 components (identical MB), and the FP8 counterpart reproduces the stored Praxis FP8 folder the same way. The recipe is checked against real folders, not only against itself. What this does not establish is the quality of a GPTQ Logos NVFP4 checkpoint, which has to be made and measured.

Logos: 20 layers (3 attention), hidden 2560, 192 routed experts, 4 picks per token, 2 passes.

| | Logos FP8 (stored) | Logos NVFP4 (modelled) | NVFP4 against FP8 |
|---|---:|---:|---:|
| MB per step, 1 stream (context 512) | 3453.5 | 3027.2 | -426.3 (-12.3%) |
| Ceiling at spec / measured, 1 stream | 259 / 247 tok/s | 296 / 282 tok/s | +14% |
| Expected experts per layer-pass, 8 / 32 / 64 streams | 29.8 / 94.1 / 142.1 | same | |
| MB per step, 8 streams | 7245.0 | 5744.5 | -1500.5 (-20.7%) |
| MB per step, 32 streams | 17972.3 | 13788.5 | -4183.8 (-23.3%) |
| MB per step, 64 streams | 28690.4 | 22505.9 | -6184.5 (-21.6%) |
| Measured-bandwidth ceiling, 8 / 32 / 64 streams | 941 / 1517 / 1901 tok/s | 1188 / 1980 / 2426 tok/s | +26% / +31% / +28% |

Where the gain comes from, MB per step. The routed experts take the union of the streams' choices, so their share grows with the batch. The dense tensors are read once per step whatever the batch, so the non-expert part of the gain is fixed.

| Streams | Gain per step | From routed experts | Share from experts | From everything else |
|---|---:|---:|---:|---:|
| 1 | 426.3 | 166.8 | 39% | 259.5 |
| 8 | 1500.5 | 1240.9 | 83% | 259.6 |
| 32 | 4183.8 | 3924.3 | 94% | 259.5 |
| 64 | 6184.5 | 5925.0 | 96% | 259.4 |

The 259.5 MB that does not scale is the same tensors as in the Praxis comparison: the LM head (85.7), latent_down (53.6), the shared expert (48.4), the depth memory routing features, query and output (68.5), the state write and key and value (3.3). At one stream it is 61% of the gain, and by 32 streams it is 6%. On this model the case for NVFP4 rests on the experts at batch sizes above one, and on quantizing the dense and depth-memory tensors at one stream.

## The gap at one stream

Time per token is 1 / measured tok/s. Bandwidth time is bytes per token over 853 GB/s. The rest is not bandwidth.

| Row | Time per token | Bandwidth time | Not bandwidth |
|---|---|---|---|
| Praxis BF16 | 3.30 ms | 2.14 ms | 1.15 ms |
| Praxis INT8 | 2.89 ms | 1.57 ms | 1.32 ms |
| Praxis FP8 | 2.98 ms | 1.77 ms | 1.20 ms |
| Praxis NVFP4 | 2.79 ms | 1.52 ms | 1.27 ms |
| Logos FP8 | 6.27 ms | 4.05 ms | 2.22 ms |

For Praxis the non-bandwidth share is about 1.2 ms per token in every format, whatever the bytes. That is why the formats that read fewer bytes show a lower share of their roofline: their extra headroom is the same fixed cost. This is arithmetic from the table, not a profile. The per-step overhead (launches, synchronisation, sampling, the host loop) is the next thing to measure.

## Beyond the bytes-per-token roofline

Everything above assumes one token per weight sweep: a decode step reads the weights once and emits one token per stream. Two schemes break that assumption. Speculative decoding has a draft propose tokens that the target checks in one sweep. Pass-pipelined decoding overlaps the model's passes across tokens, so one sweep carries more than one token's work. For those, the bytes-per-token ceiling is no longer the limit to measure against, so this table is a second view with its own axis.

Definitions, for whoever fills it:

- **Tokens per weight sweep**: tokens a step emits divided by the weight bytes that step reads. It is 1 for the decode path reported above.
- **One-token ceiling**: bandwidth divided by bytes per token, as in the roofline table. A scheme's bandwidth ceiling is its tokens per sweep times this.
- **Share of the one-token ceiling**: measured tok/s over the one-token ceiling. It can exceed 100% when tokens per sweep exceed 1.
- **Acceptance or pass utilisation**: the draft acceptance rate for speculative decoding, or the fraction of pass slots that carry a token for pipelined decoding.

| Scheme | Model and format | Draft length or pass depth | Tokens per weight sweep | Weight MB per sweep | Streams | Measured tok/s | Share of one-token ceiling | Acceptance or pass utilisation | Commit | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Prompt lookup, lone stream | Praxis BF16 | up to 6 drafts, adaptive | 1.56 | 1827 | 1 | 441 | 94% | 0.56 accepted | a9a0e94 | 1.33x plain, chat mix, greedy |
| Prompt lookup, lone stream | Praxis INT8 | up to 6 drafts, adaptive | 1.71 | 1340 | 1 | 542 | 85% | 0.71 accepted | a9a0e94 | 1.42x plain, chat mix, greedy |
| Prompt lookup, lone stream | Logos FP8 | up to 6 drafts, adaptive | 1.56 | 3455 | 1 | 238 | 96% | 0.56 accepted | a9a0e94 | 1.37x plain, chat mix, greedy |
| Prompt lookup, lone stream, think 0.8 | Praxis BF16 | up to 6 drafts, adaptive | 1.39 | 1827 | 1 | 386 | 83% | 0.39 accepted | a9a0e94 | 1.21x plain, think_chat |
| Prompt lookup, lone stream, think 0.8 | Praxis INT8 | up to 6 drafts, adaptive | 1.25 | 1340 | 1 | 424 | 67% | 0.25 accepted | a9a0e94 | 1.16x plain, think_chat |
| Prompt lookup, lone stream, think 0.8 | Logos FP8 | up to 6 drafts, adaptive | 1.35 | 3455 | 1 | 209 | 85% | 0.35 accepted | a9a0e94 | 1.23x plain, think_chat |
| Prompt lookup, lone stream, think 0.8, repetition 1.05 | Praxis BF16 | up to 6 drafts, adaptive | 1.19 | 1827 | 1 | 350 | 75% | 0.19 accepted | a9a0e94 | 1.11x plain, think_chat |
| Prompt lookup, lone stream, think 0.8, repetition 1.05 | Praxis INT8 | up to 6 drafts, adaptive | 1.28 | 1340 | 1 | 416 | 65% | 0.28 accepted | a9a0e94 | 1.15x plain, think_chat |
| Prompt lookup, lone stream, think 0.8, repetition 1.05 | Logos FP8 | up to 6 drafts, adaptive | 1.13 | 3455 | 1 | 181 | 73% | 0.13 accepted | a9a0e94 | 1.07x plain, think_chat |

Speculative rows are prompt-lookup drafts on a lone stream, from `dromos spec` on the chat mix (the five spec prompts, think and direct_tools, 512 tokens, two repeats). Tokens per weight sweep is tokens per decode step, verify and plain steps together. Weight MB per sweep is the bandwidth time in the roofline table times 853 GB/s. Speedups are chat-mix aggregates over plain decode of the same format, and greedy text is identical to plain on every prompt. The think rows are think_chat alone, one prompt, so they are noisier than the mix.

- **Praxis BF16:** 1.33x greedy, 1.32x at temperature 0.3, 1.21x for think at 0.8.
- **Praxis INT8:** 1.42x greedy, 1.32x at 0.3, 1.16x for think at 0.8.
- **Logos FP8:** 1.37x greedy, 1.36x at 0.3, 1.23x for think at 0.8.

Penalised requests speculate too. With a repetition penalty of 1.05 the think gain drops (Praxis BF16 1.21x to 1.11x, Logos FP8 1.23x to 1.07x): the penalty pushes down the context tokens prompt lookup copies, so fewer drafts are accepted (0.39 to 0.19 accepted per step on Praxis BF16). The verify step judges each row with the penalties the plain sampler would apply. Pass-pipelined rows are not filled yet.

Think mode on one stream, from `dromos spec` on the think prompt at temperature 0.8: plain decoding against prompt lookup. Measured on the rebased stack before it landed, the same speculation code as a9a0e94.

| Model and format | Repetition penalty | Plain tok/s | Prompt lookup tok/s | Ratio |
|---|---|---|---|---|
| Praxis BF16 | 1.0 | 319.8 | 385.5 | 1.21x |
| Praxis BF16 | 1.05 | 316.3 | 350.5 | 1.11x |
| Praxis INT8 | 1.0 | 366.1 | 423.7 | 1.16x |
| Praxis INT8 | 1.05 | 362.1 | 415.9 | 1.15x |
| Logos FP8 | 1.0 | 170.4 | 209.1 | 1.23x |
| Logos FP8 | 1.05 | 169.3 | 180.7 | 1.07x |

Sampling modes on one stream: tokens per decode step with prompt lookup, over the eight prompts of `all8.json`. Plain decoding is 1.00. Every prompt in every mode took fewer steps than output tokens, so speculation engaged in all of them. These numbers show engagement, not speed.

| Sampling mode | Praxis BF16 | Praxis INT8 | Logos FP8 |
|---|---|---|---|
| Greedy | 1.79 | 1.95 | 1.56 |
| Temperature 0.7, top_p 0.9 | 1.77 | 1.86 | 1.46 |
| Think card: temperature 0.8, top_p 0.95, top_k 20, repetition 1.05 | 1.27 | 1.35 | 1.28 |
| Direct card: temperature 0.3 | 1.76 | 1.88 | 1.65 |
| min_p 0.05 at temperature 0.8 | 1.70 | 1.78 | 1.41 |
| top_k 40 at temperature 0.8 | 1.70 | 1.67 | 1.62 |

## Against other engines

Same bench shape as above (512 prompt tokens, 256 generated, greedy), same card, under the GPU lock. Each row names the engine's version and the commit or release it ran.

| Engine | Version | Model and format | 1 stream (tok/s) | 8 streams (tok/s) | 32 streams (tok/s) | Share of roofline at 1 stream | Status |
|---|---|---|---|---|---|---|---|
| vLLM | pending | Praxis BF16 | pending | pending | pending | pending | not measured |
| SGLang 0.5.22 + sglang-metis 0.1.0 | 0.5.22 | Praxis BF16 | 141.8 (decode) | 674 (e2e) | 980 (e2e) | 30% of 467 (Praxis BF16 measured ceiling) | measured, rivals; `docs/RIVALS.md` |
| SGLang 0.5.22 + sglang-metis 0.1.0 | 0.5.22 | Praxis FP8 checkpoint | 141.9 (decode) | 740 (e2e) | 1596 (e2e) | 25% of 564 (Praxis FP8 measured ceiling) | measured, rivals; `docs/RIVALS.md` |
| llama.cpp metis16 | 07cacf7fa | Praxis BF16 GGUF | 185.2 (tg256) | not run | not run | 40% of 467 (Praxis BF16 measured ceiling) | measured, rivals; `docs/RIVALS.md` |
| llama.cpp metis16 | 07cacf7fa | Praxis Q4_K_M GGUF (5.5 bits) | 253.3 (tg256) | not run | not run | not comparable (a different format; 38% of the 658 tok/s NVFP4 ceiling, for reference) | measured, rivals; `docs/RIVALS.md` |
| MLX | pending | Praxis, MLX | pending | pending | pending | pending | not measured |

The SGLang rows at 8 and 32 streams are end-to-end output tok/s from `tools/rivals/load.py` (the
prefill ramp is included), not the decode column above, so compare them with the Dromos e2e
number from the same client. The 1-stream SGLang rows are the per-stream decode rate. Dromos's
column is filled from the final-main run of that same client (see `docs/RIVALS.md`).

## How to read the numbers

- Decode tok/s counts the tokens emitted after the last stream's first token, over the time to the last finish (`cli/bench.cpp`, `measure()`). Any batch larger than the engine's slot count is distorted by this, as the Logos 32-stream row shows.
- Prefill tok/s is prompt tokens over the time to the last first token, so it includes queueing as well.
- "% of spec" divides by 896 GB/s. "% of measured" divides by 853 GB/s.
- Doctor counts a format it cannot load in yet with `--modelled`: the loader's recipe for that format applied to the folder's config, nothing read. Use it only after checking the recipe against a stored folder, as above.

## Open items

- **Bench and slot count.** `dromos bench` should refuse, or at least flag, a stream count above the plan's slot count. Otherwise the row looks like a decode rate and is not one.
- **Planner estimate.** `plan.cpp` (`decode_bytes_per_token`) estimates Praxis BF16 at 1730 MB per token and 290 tok/s. The counted figure is 1827 MB at context 512. The gap is the depth-memory routing features, the depth query and output, and the depth head. `doctor`'s "decode est." column uses the planner figure. Not changed here.
- **Logos NVFP4 quality.** The modelled Logos NVFP4 roofline is a byte count. It needs a GPTQ checkpoint and a quality measurement before any speed is quoted for it.
- **Compute peaks.** The 87.9 TFLOPS BF16 figure for sm_120 is derived from the 1406 sparse FP4 TOPS on NVIDIA's product page (halved to dense, then divided by 8 for BF16). Rows for other cards in `cli/roofline.cpp` come from datasheet values and have not been re-checked online.
