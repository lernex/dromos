# The push from one fast stream to a product

Started 2026-10-24, after milestone 2 (295 tok/s single-stream Praxis, int8 decode).

## What it has to become

`lernex run praxis` and `lernex serve logos`: one program that picks the weight format and the
math for whatever NVIDIA GPU it lands on, batches many users, and is the fastest way to run
Metis in every format, compared format for format with vLLM, SGLang and llama.cpp.

The speed claim is about the engine, never the quantization. Every benchmark names its format
and is compared against other engines running the same format.

## Where decode time goes (from milestone 2)

Per token, single stream: weight bytes / 896 GB/s (BF16 2.38 ms, 8-bit 1.37 ms) plus about
2.0 ms of everything else, the same in every format (vLLM: about 2.6 ms). That 2 ms is about 900
kernel launches from Metis's small parts, run twice. The math itself is ~1% of a decode step.
Prefill and large batches are the opposite: math-bound, which is where FP8 and FP4 tensor cores pay.

## Workstreams

Each runs in its own clone of this repo with its own `scripts/remote.sh` slot. Anything that
loads a whole model or measures speed goes through `remote.sh SLOT x '...'` (one GPU lock, so
runs never share the card). `engine/model/metis.cpp` is integrated by the lead only: workstreams
deliver kernels and modules behind the interfaces below, with tests, and may hand back a
proposed metis.cpp diff for their part.

| | Owns | Interface |
|---|---|---|
| Formats | decode GEMV and expert kernels for FP8 / NVFP4 / INT8, load-time quantizers, the checkpoint loader (BF16, compressed-tensors FP8 and NVFP4), an NVFP4 checkpoint of post-trained Praxis | `kernels.h` lx_gemv_w, lx_moe_experts_decode, lx_quantize_weight; `core/checkpoint.h` |
| Native math | tensor-core GEMM and grouped expert GEMM in every format and math mode, activation quantization, the per-arch kernel families, autotunable variants | `kernels.h` lx_gemm_w, lx_moe_experts_grouped |
| Decode overhead | fused decode kernels that take the ~900 launches per token down, the decode Mamba step, and the restructured decode path | new `kernels.h` entries it specifies |
| Hardware | multi-architecture build, device query and arch override, fallbacks for every sm_90+-only kernel, the autotuner and its cache, the plan (format, math, slots, context from VRAM) | `runtime/plan.h`, `runtime/tuner.h` |
| Text | tokenizer, chat template, reasoning and tool-call parsing, streaming detokenization | `text/tokenizer.h` |
| Serving | scheduler (continuous batching, chunked prefill), GPU sampling, the OpenAI-compatible server, `lernex run / serve / pull / bench`, model download | `runtime/engine.h`, `server/` |

## Correctness

Every format is judged by top-20 KL against the FP32 native logits (`tools/compare.py`,
`work/praxisrefs`; Logos refs in `hfport/dev`). BF16 must sit inside the native BF16 noise
floor (Praxis 7.3e-3 / 1.4e-2 / 2.2e-3). Quantized formats are reported next to the same file
run in vLLM (Praxis FP8 checkpoint: 0.0077 / 0.040 / 0.0066), and must not be worse.

## Targets (RTX 5070 Ti)

| | Format | To beat |
|---|---|---|
| Praxis, 1 stream | BF16 / FP8 | vLLM 201 / 231 tok/s |
| Praxis, 32 streams | BF16 / FP8 | vLLM 1,836 / 2,386 tok/s |
| Logos, 1 / 32 streams | FP8 | vLLM 117 / 770 tok/s |
| Praxis, 1 stream | 4-bit | llama.cpp Q4_K_M 250 tok/s |
| Prefill | every format | vLLM and llama.cpp, same file |
