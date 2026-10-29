# GEMV picks sweep and moe decode lever results

Only one change landed: 1ad2177, a kPicks row for the shared-down GEMV. The GEMV sweep found nothing else that clears the 5% standalone bar. The moe_down experiments were dropped. This closes the expert-decode lever for this release.

Measurements are 1-stream decode on the RTX 5070 Ti (sm_120a). Standalone numbers are graph-chain timings from `bench/sweep_gemv.py` or ncu decode-window timings. Pair numbers are alternating `dromos bench` runs at 1 stream with 512 prompt tokens and 256 generated tokens.

## What landed

- **1ad2177**, in `engine/cuda/gemv.cu`: the kPicks row for (LX_W_INT8, MT=1, N 1024, K 512) moves from s1r4d4 to s1r1d4. Split S stays 1, so the summation order and the bits do not change. The shared-down GEMV goes from 32 CTAs to 128.
- Exactness: eval logits were byte-identical to the nvkl origin/main control on Praxis BF16, INT8, FP8, NVFP4 and Logos FP8, and spec-check passed on all five formats. Those runs were on the three-change build (below), which contains this pick. The pick-only build was not evaluated separately in this record.

## Pair results and their controls

Each row compares a build with its own base. The two controls differ, so compare tok/s only within a row.

| Build | Control | Model | Base to candidate, tok/s | Result |
|---|---|---|---|---|
| pick only (1ad2177) | a1bc78d | Praxis INT8, 5 pairs | 385.9 to 388.1 | 5 of 5 wins, about 14 us per token |
| pick only (1ad2177) | a1bc78d | Logos FP8, 3 pairs | 174.2 to 174.8 | 1 win, 2 ties |
| three changes | nvkl origin/main | Praxis INT8, 5 pairs | 379.2 to 381.2 | 5 of 5 wins, about 14 us per token |
| three changes | nvkl origin/main | Praxis BF16, 5 pairs | 329.64 to 329.58 | noise, 0 of 5 |
| three changes | nvkl origin/main | Logos FP8, 5 pairs | 172.82 to 172.72 | noise, 2 of 5 |

The three-change build was the shared-down pick plus the moe_down 4-warp shrink plus the moe_down ring depth 8. Its INT8 gain matches the pick-only gain against its own control. Base-to-base spread is about 0.3 tok/s, roughly 2 us per token, so at this noise level the two moe_down changes show no visible pair gain. Pairs cannot split the three-change gain between its parts.

## Standalone chains and pairs disagree on INT8 1024x512

- Standalone chain: s1r4d4 at 1.085 us, s1r1d4 (landed) at 1.201 us. The old pick is 10.7% faster alone.
- In-engine pairs: s1r1d4 is about 14 us per token faster, 5 of 5.

Pick changes must be decided by in-engine pairs, not standalone chains. Why the two disagree is not measured. The likely reason is that a back-to-back chain does not reproduce the neighbouring launches inside the graph, but that is a hypothesis. s1r1d4 stays.

## Logos FP8 has no M=1 entries of its own

Logos FP8 has no kPicks rows at M=1. Its decode GEMVs run the nearest existing pick (`nearest_pick` in `gemv.cu`). The M=1 sweep covered the 12 dense products. Standalone, best same-split variant against the current pick:

| Shape (N x K) | Current pick | Best same-S | Standalone gain |
|---|---|---|---|
| 256x2560 | s8r1d2 | s8r1d4 | 1.08% |
| 256x7936 | s8r1d4 | s8r2d2 | -11.78% |
| 1024x768 | s2r1d4 | s2r1d2 | -4.16% |
| 1024x2560 | s2r1d4 | s2r2d3 | 7.30% |
| 1536x1024 | s2r2d2 | s2r2d3 | 0.11% |
| 2560x256 | s1r1d2 | s1r1d4 | 0.00% |
| 2560x1024 | s2r2d2 | s2r4d2 | 0.32% |
| 2560x2560 | s1r2d3 | s1r2d4 | 1.64% |
| 2560x5120 | s2r2d2 | s2r1d8 | 1.74% |
| 3840x2560 | s1r2d3 | s1r1d8 | 0.97% |
| 12880x2560 | s2r4d2 | s2r2d3 | 0.56% |
| 65536x2560 | s2r2d3 | s2r4d3 | -0.15% |

Only 1024x2560 (latent_down) clears 5%: s2r2d3 is 3.22 us against 3.46 us for s2r1d4. The estimate is 6 to 8 us per token, below the 10 us threshold. It was not landed, per the coordinator's decision for 0.1.0. If it is landed later, one kPicks row (FP8, MT=1, N 1024, K 2560, S=2 to s2r2d3) keeps the split and so keeps the bits. M=2, 4 and 8 need their own sweep first.

## How the GEMV sweep was run

- `dromos tune` does not sweep GEMVs here. It reports two kernels, memory_fuse and mhc_read (26 problems), and no gemv_w problems. The sweep used `bench/sweep_gemv.py` instead, which times each variant in a back-to-back graph chain, run on the experts slot through `scripts/remote.sh`.
- Selection: same split S as the current pick. The checked-loop variant (G=1) is built only for the M=8 bucket, so it is not a candidate at M=1. A change needs at least 5% standalone before pairs decide.
- Server CSVs: `~/lernex-engine-dev/experts-scripts/sweep/int8-m1.csv`, `bf16-m1.csv`, `logos-fp8-m1.csv`.
- A local patch to `cli/tune.cpp` (a `--weights` flag) was reverted and is not landed.

## moe decode lever results (Praxis INT8, 1 stream, ncu decode window)

- **moe_gate_up**: 8.19 us for 4.21 MB, about 514 GB/s. Grid of 256 blocks, 58 registers, 4 blocks per SM (register limit), 62% warps active. 256 blocks is 0.91 waves on 70 SMs. Not changed. Untested next step: cap registers at 48 or try a smaller tile before any structural change.
- **moe_down 4-warp blocks** (256 blocks instead of 128): 6.72 to 6.59 us standalone, about 2%. Warps active stayed near 30%, since the per-SM warp maximum is unchanged. No visible pair change. Dropped.
- **moe_down ring depth 8** (INT8 and FP8): 6.98 us standalone, slower than both the 6.72 us baseline and the 6.59 us shrink. Registers rose to 80. The run's report gives 56 as the starting count and the baseline table lists 60. No gain on the Logos FP8 pairs. Dropped.
- **Shared down** (landed as 1ad2177): 3.71 to 3.52 us standalone, warps active 17% to 33%.

## Not done

- Logos FP8 sweep at M=2, 4 and 8.
- The shared_join stall check from the brief. No timeline was produced.
- Activation fold into the gate_up epilogue. Skipped per the coordinator. silu_mul runs on stream 15, and side-stream launches cost the step little.
- The 48-register cap on moe_gate_up, and the latent_down row, both wait for a later release.

## Follow-up: 1-stream and BF16X at 8 streams

- **1 stream, moe_gate_up.** `__launch_bounds__(kThreads, 5)` gives 48 registers and five blocks per SM. Two of the five instantiations spill (32 and 40 bytes of stack; tip is 58 to 59 registers with no spill). The spec-check is 30 of 30 bit-identical. Five alternating pairs at 1 stream on Praxis INT8 give 404.5 to 404.7 tok/s against 405.2 to 405.4 for tip, five of five losses. Dropped.
- **1 stream, INT8 M = 1 picks.** For all 12 Praxis INT8 products the chosen variant is already the best same-split variant in `int8-m1.csv`, so no pick change clears 5%. Praxis INT8 measures about 405 tok/s at 1 stream on tip with 512 prompt and 256 generated tokens, not 480.
- **BF16X at M = 8.** All 23 real Praxis and Logos products, both x types, 12 variants each, timed in a rotated 256 MB pool and checked bit for bit against the production output. Two candidates clear 5% and are bit-identical:
  - Logos N 2560, K 1024, split 1: r1d2g0 to r4d2g1 (7.98 to 6.11 us).
  - Logos N 2560, K 5120, split 2: r4d2g1 to r2d3g1 (32.43 to 28.70 us).

  Both are kPicks rows in `engine/cuda/bf16x.cu`. Logos FP8 at 8 streams, five alternating pairs on origin/main: 677.0 to 683.3 tok/s aggregate, five of five wins (about +1%). The suite passes and the spec-check is 30 of 30 on five formats.
- **Table behaviour.** A kPicks row also changes the nearest-entry pick of products that never run, and can reach M = 4 for some shapes. The two rows above change no pick for the 23 real products at M = 1, 2 or 4. Check the real-product list the same way for any future row.
