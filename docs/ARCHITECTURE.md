# Dromos: architecture

An inference engine written for the Metis models (Metis-1.6 Logos 10.2B, Praxis 3.0B) and
nothing else. General engines run Metis through layers built for other models; this one is
shaped around what Metis actually is, which is where its speed has to come from.

## The model, as the engine sees it

Per token: embed, broadcast to S = 4 residual streams, then the stack of L blocks runs twice
(pass 0 and 1) with the same weights. Each block:

1. depth-memory read: attend over the token's own memory bank, gate the result into the streams,
   and compute routing features `tanh(route_projection([state, retrieved, difference, history]))`
2. hyper-connection read (4x4 Sinkhorn mix + weighted read + RMSNorm) -> mixer -> write
   (mixer = Mamba-2, or GQA attention with no positional encoding and a per-pass LoRA on QKV)
3. hyper-connection read -> latent MoE (route on [latent, routing features], top-4 of E plus a
   shared expert) -> write
4. after an attention block, and at the end of each pass, a new memory entry is written

Then mean over streams, RMSNorm, tied LM head. The reference implementation is
`engines-mac/mlx-metis/metis16.py` (exact port of the native model, verified to its BF16 noise
floor); `docs/KERNELS` is `engine/cuda/kernels.h`.

Only the mixers look across positions, so a sequence's state is: per (pass, Mamba layer) a conv
state and an SSM state (fixed size, f32), per (pass, attention layer) a KV cache (small: few
attention layers, GQA). Everything else (memory bank, routing history) is per token and dies
with the step.

## Numerics

The correctness bar is the FP32 native runtime: KL to its logits, top-20, on the three
reference prompts, must stay inside the BF16 noise floor of the native model itself
(Praxis 7.3e-3 / 1.4e-2 / 2.2e-3; Logos 2.5e-3 / 1.3e-2 / 3.7e-3).

- Residual streams, norms, routing, memory, SSM and conv state: f32.
- GEMM inputs bf16 (tensor cores), f32 accumulation. Routers and the depth head are f32 weights
  and run as exact f32 GEMMs.
- The 4x4 mixes depend only on (layer, pass): computed once at load on the host.

## Layout

- `engine/cuda/` kernels behind a C ABI (`kernels.h`). Each is tested on its own against a
  PyTorch reference (`tests/kernels`, through ctypes).
- `engine/core/` config, safetensors loading, device memory.
- `engine/model/` the Metis forward pass, built from the kernels.
- `engine/runtime/` state slots, scheduler, batching (after the first milestone).
- `cli/` the `dromos` binary.
- `tools/compare.py` end-to-end check against the reference logits.

## Milestones

1. Praxis BF16 on CUDA, one sequence, prefill + decode, inside the noise floor.
2. One-sequence speed: quantized weights, fused kernels, CUDA graphs. Target: beat 250 tok/s
   (llama.cpp Q4_K_M on the 5070 Ti).
3. Continuous batching and the server. Target: beat vLLM's 2,386 tok/s (Praxis FP8, 32 streams).
4. Logos (117 / 770 tok/s to beat), Metal, then the CLI and installers.
