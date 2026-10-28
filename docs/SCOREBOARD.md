# Decode scoreboard

How fast the Lernex engine decodes, measured against the memory bandwidth of the card it runs on.

Card: NVIDIA GeForce RTX 5070 Ti (16 GB, sm_120), on the lernex GPU server.
Measured: 2026-10-27, on commit 4b25613 (`Add lx_bandwidth_probe and the decode roofline to doctor and bench`), under the GPU lock, on an otherwise idle card.

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

Where each row's weights come from: Praxis BF16 is the BF16 file as stored. Praxis INT8 is round-to-nearest from the BF16 file, at load. Praxis FP8 and NVFP4 load the checkpoints stored under `kronos-scratch/logos/engines/praxis/` (`fp8`, `nvfp4`); NVFP4 is refused from a plain BF16 checkpoint. Logos FP8 is `Metis-1.6-Logos-FP8` as stored.

## Measured decode: `lernex bench`

One run per format and stream count on commit 4b25613:

`lernex bench MODEL --streams 1,8,32 --prompt-tokens 512 --gen-tokens 256`

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

Praxis BF16, one stream, context 512: 1827 MB per token.

| Component | MB per token | Share |
|---|---|---|
| Mamba in_proj (20 layers) | 639.7 | 35.0% |
| Mamba out_proj (20 layers) | 253.6 | 13.9% |
| LM head | 203.4 | 11.1% |
| Mamba state | 86.8 | 4.8% |
| Routed experts, gate_up / down | 152.6 / 77.1 | 8.3% / 4.2% |
| Latent down / up | 76.3 / 76.3 | 4.2% each |
| Depth memory routing features | 64.4 | 3.5% |
| Attention qkv / out | 38.1 / 25.4 | 2.1% / 1.4% |
| Shared expert gate_up / down | 38.1 / 19.3 | 2.1% / 1.1% |
| Router (f32) | 15.7 | 0.9% |
| Depth memory query / output | 16.7 / 17.2 | 0.9% each |
| Depth head (f32) | 13.1 | 0.7% |

Groups: mixer 958.6, MoE 455.4, LM head 203.4, depth memory 118.7, state and cache 91.0 (MB per token).

Praxis INT8, one stream, context 512: 1336 MB per token.

| Component | MB per token | Share |
|---|---|---|
| Mamba in_proj (20 layers) | 422.9 | 31.7% |
| Mamba out_proj (20 layers, kept BF16) | 253.6 | 19.0% |
| LM head | 134.5 | 10.1% |
| Mamba state | 86.8 | 6.5% |
| Routed experts, gate_up / down | 101.1 / 50.7 | 7.6% / 3.8% |
| Latent down / up | 50.4 / 50.5 | 3.8% each |
| Depth memory routing features | 42.6 | 3.2% |
| Attention qkv / out | 25.2 / 16.8 | 1.9% / 1.3% |
| Shared expert gate_up / down | 25.3 / 12.7 | 1.9% / 0.9% |
| Router (f32) | 15.7 | 1.2% |
| Depth memory query / output | 11.0 / 11.2 | 0.8% each |
| Depth head (f32) | 13.1 | 1.0% |

Groups: mixer 720.2, MoE 306.5, LM head 134.5, depth memory 83.6, state and cache 91.0 (MB per token).

Two things the INT8 breakdown shows. Mamba out_proj stays BF16 in the INT8 format, and at 253.6 MB it is now the second-largest read (19%). Quantizing it would remove about 127 MB per token, which bounds the one-stream ceiling near 700 tok/s at measured bandwidth. That is arithmetic only: its quality cost is not measured. Mamba state (86.8 MB) is read every token in every format.

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

## Against other engines

Same bench shape as above (512 prompt tokens, 256 generated, greedy), same card, under the GPU lock. Each row names the engine's version and the commit or release it ran.

| Engine | Version | Model and format | 1 stream (tok/s) | 8 streams (tok/s) | 32 streams (tok/s) | Share of roofline at 1 stream | Status |
|---|---|---|---|---|---|---|---|
| vLLM | pending | Praxis BF16 | pending | pending | pending | pending | not measured |
| SGLang | pending | Praxis BF16 | pending | pending | pending | pending | not measured |
| llama.cpp | pending | Praxis, GGUF | pending | pending | pending | pending | not measured |
| MLX | pending | Praxis, MLX | pending | pending | pending | pending | not measured |

## How to read the numbers

- Decode tok/s counts the tokens emitted after the last stream's first token, over the time to the last finish (`cli/bench.cpp`, `measure()`). Any batch larger than the engine's slot count is distorted by this, as the Logos 32-stream row shows.
- Prefill tok/s is prompt tokens over the time to the last first token, so it includes queueing as well.
- "% of spec" divides by 896 GB/s. "% of measured" divides by 853 GB/s.

## Open items

- **Bench and slot count.** `lernex bench` should refuse, or at least flag, a stream count above the plan's slot count. Otherwise the row looks like a decode rate and is not one.
- **Planner estimate.** `plan.cpp` (`decode_bytes_per_token`) estimates Praxis BF16 at 1730 MB per token and 290 tok/s. The counted figure is 1827 MB at context 512. The gap is the depth-memory routing features, the depth query and output, and the depth head. `doctor`'s "decode est." column uses the planner figure. Not changed here.
- **Compute peaks.** The 87.9 TFLOPS BF16 figure for sm_120 is derived from the 1406 sparse FP4 TOPS on NVIDIA's product page (halved to dense, then divided by 8 for BF16). Rows for other cards in `cli/roofline.cpp` come from datasheet values and have not been re-checked online.
