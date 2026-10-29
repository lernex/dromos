# Draft heads for Metis-1.6: EAGLE-4, EAGLE-3 or a block drafter

Written 2026-10-24 by the speculative decoding workstream. Numbers marked *measured* come from
runs in this clone on the RTX 5070 Ti; numbers marked *computed* are arithmetic on tensor shapes
and measured rates; numbers from papers are the papers' own and say so.

**Status, 2026-10-28: not for 1.6.** The owner decided Metis-1.6 ships without a trained drafter: its
speculation is prompt lookup only, and the 1.6 work stays on kernels. Metis-1.7 has drafters built in,
trained by the owner. The engine path below (the feature taps, `BlockDrafter`, chain verification)
stays as the place they plug in; the recommendation that follows is kept as the record of the study.

## Recommendation

Train a **block drafter** for Praxis now: DFlash's one-pass backbone (the target's hidden states
injected as keys and values in every draft layer), DSpark's confidence head and loss, chain
verification, one block filling exactly one engine verify step. The engine side, the trainer, the
data pipeline and the one-command launch are built and checked (below); training has not been run.

Why this one and not EAGLE-4:

1. **EAGLE-4's gain is its tree, and Metis can't use a tree.** EAGLE-4 drafts token by token like
   EAGLE-3 and wins mainly through a hardware-aware dynamic tree verified in one pass (up to 8
   branches per node). On Metis a tree costs twice: the Mamba-2 state is a chain recurrence (the
   rollback history replays one path, and a tree needs a state per branch), and the verify step
   is capped at 7 tokens on Praxis and 6 on Logos (beyond that attention's split changes the
   order of its sums and verify rows stop being bit-identical to plain steps), with each extra
   token costing about 0.11 of a plain step because MoE reads more experts. A binary tree of 7
   nodes covers paths of 3 tokens, where a chain of 7 covers 6.
2. **Without the tree, EAGLE-4 is EAGLE-3 plus two ideas the block drafter takes anyway**: a stop
   rule from predicted acceptance (here: DSpark's confidence head feeding `DraftCost`, the
   measured verify-cost curve) and a low-rank preselect for the LM head (an option below).
3. **Per-token drafting is the expensive shape at batch 1.** Six drafts are six sequential
   passes of a draft layer and a head (*computed* 724 MB per round at int8 for an EAGLE-3 head on
   Praxis, 428 MB with EAGLE-4's low-rank head) against one pass of a block (273 MB), and every
   pass is a chain of small kernels whose launch latency dominates at this size.
4. **EAGLE-4 ships no training code** (no training section in the paper, no script in
   eagle-hq/EAGLE-4) and its license is source-available with a revenue cap; EAGLE-3, DFlash and
   DSpark train from MIT/Apache code (SafeAILab/EAGLE, deepseek-ai/DeepSpec).
5. **Metis-1.7 has DFlash/DSpark-style drafters built in.** The engine path built here (block
   proposal, KV injection from feature taps, confidence-scheduled length, exact chain
   verification) is the one 1.7's drafters plug into.

## What Metis allows (measured)

| | Praxis BF16 (int8 decode), 5070 Ti |
|---|---|
| Plain decode | 290 to 293 tok/s greedy, 251 to 284 sampled (512 tokens) |
| Largest batch-invariant verify step | 7 tokens (Logos: 6) |
| Verify step of T tokens, in plain steps (1k context) | 1.04, 1.16, 1.29, 1.36, 1.49, 1.57, 1.70 for T = 1 .. 7 |
| Greedy exactness | every speculative run emits the plain run's tokens (512 tokens, 5 prompts, every drafter) |

So a round that verifies 6 drafts must average more than 1.70 tokens (plus the drafter's own
time) just to break even, and each extra draft adds 0.07 to 0.13 of a step. That is why the
drafters here choose their length per step instead of always drafting 6.

## The three methods, from their papers and code

| | EAGLE-4 | EAGLE-3 | DFlash / DSpark |
|---|---|---|---|
| Drafting | 1 layer, token by token, dynamic tree, stop/continue rule | 1 layer, token by token, tree | one pass per block (DFlash 5 layers; DSpark adds a Markov head and confidence-scheduled length) |
| Target features | low/mid/high (paper unspecific) | layers 2, L/2, L-3, fused by a linear | 5 layers, projected, injected as K/V in every draft layer |
| Head | full vocabulary, low-rank preselect (r 128 to 512) | reduced 32k vocabulary | the target's frozen LM head |
| Reported | Qwen3.8-27B, batch 1, T=0: 3.59x (DFlash 2: 2.80x), T=1: 2.59x (own harness, 4x B300) | LLaMA-3.1-8B T=0: tau 6.23, 4.44x; T=1 tau 4.92 (tree) | Qwen3-8B T=0: tau 6.49, 4.86x (EAGLE-3 tree 60: tau 3.40); DSpark Qwen3-4B T=1 GSM8K tau 6.11 vs DFlash 5.40 vs EAGLE-3 5.14 |
| Training recipe | not published | ShareGPT + UltraChat regenerated, AdamW lr 5e-5, 40 epochs, 7-step training-time test | ~800k (DFlash) / 1.3M (DSpark) regenerated prompts, 512 anchors per sequence, AdamW lr 6e-4, 6 to 10 epochs |
| Code, license | serving only, source-available | Apache-2.0 | MIT (DeepSpec) |

None of the four reports on a hybrid Mamba target except EAGLE-4's Nemotron run, which keeps its
chain length fixed there and describes no Mamba-specific change.

## Sized for Metis (computed)

Bytes are the weights one verify round reads at int8 (per-row scales), k = 6 drafts on Praxis
and 5 on Logos. The block drafter reads the target's own int8 head (shared, not copied).

| | Praxis params | Praxis bytes / round | Logos params | Logos bytes / round | Draft passes per round |
|---|---|---|---|---|---|
| EAGLE-3 (1 layer, 32k head) | 131M | 724 MB | 184M | 841 MB | k |
| EAGLE-4 (1 layer, low-rank head r=256, no tree) | 81M + shared head | 428 MB | 117M + shared head | 512 MB | k |
| Block drafter, lernex.block/1 (3 layers) | 132.1M | 273 MB | 206.5M | 384 MB | 1 |
| Block drafter with a low-rank head (option) | 149M | 159 MB | 224M | 237 MB | 1 |

Training cost is about the same for all three on the same data: the target's forward pass over
the corpus (the hidden states) is identical, and the draft side is *computed* at about 5 TFLOP per
1,000-token sequence both for a 7-step EAGLE-3 unroll and for 512 block anchors.

## lernex.block/1

`tools/draft/block_drafter.py` (training, the reference) and `engine/runtime/block_drafter.cpp`
(decode) implement the same function from the same files (`config.json`, `model.safetensors`):

- **Inputs**: six taps of the target, the stream means after (pass, layer) = (0,1), (0,6), (0,11),
  (1,4), (1,9) and the final state (1,-1) on Praxis (Logos: (0,1), (0,10), (0,19), (1,8), (1,17),
  (1,-1)), spread over the 24 (40) virtual layers as DFlash spreads its five, plus the state the
  LM head reads. `fused = rmsnorm(fc(concat))`.
- **Context**: per draft layer, keys `rope(k_norm(k(fused)))` and values `v(fused)` of the last 256
  committed positions (DSpark deploys a 128-token window), kept by the engine in a ring that
  `observe_features` fills, so a proposal never touches the target's history.
- **Block**: B = verify cap (7 on Praxis, 6 on Logos). Row 0 is the pending token's embedding (the
  target's tied table), rows 1 .. B-1 a learned mask vector. Three pre-norm layers at the target's
  width (32 heads, 8 KV heads, head 64, QK-norm, RoPE theta 10000 on absolute positions with angles
  in float64, SwiGLU 4096), every block row attending to the context window and to all block rows.
- **Outputs**: row j's logits through the target's frozen head; `confidence = sigmoid(conf(h))`,
  two numbers per row: the chance it holds under greedy and under sampling, given the rows before
  it held. `DraftCost.choose` turns them into the draft length.
- **Proposals**: greedy rows propose their argmax; sampled rows draw from their own filtered
  distribution (the target's temperature / top-p / top-k / min-p) and hand those rows to
  `lx_spec_verify` as q. Rows don't see each other's draws, which speculative sampling allows.

Every product reads bf16 activations, as training under bf16 autocast did. At 7 rows a GEMV that
reads f32 rows re-reads them for every weight word and ran at 290 to 390 GB/s of weights on this card
(*measured*, `lx_gemv`, int8 weights, N x K from 2048 x 2048 to 69k x 2048), against 420 to 630 GB/s
with bf16 rows.

Checks (*measured*, Praxis, a freshly initialized drafter, 5 prompts):

- The batched training forward equals a position-by-position reference to 1e-9 in float64
  (`tests/draft`, 5 tests).
- The engine's kernels match PyTorch (`tests/kernels/test_spec.py`: per-head norm with RoPE, block
  attention, 10 cases).
- The engine's proposal against the PyTorch model on the same feature rows (`dromos block-check` +
  `tools/draft/check_block.py`): with bf16 activations, logits 2.3e-3 to 3.2e-3 from a float64
  reference (relative to each row's largest logit), where PyTorch's own float32 run of the same
  model is 2.8e-3 to 3.2e-3 from it; confidences 1.2e-3 to 6.1e-3 (float32: 8.2e-4 to 8.7e-3);
  every row's argmax agrees. Before the switch to bf16 activations the engine was 4.9e-7 to 7.3e-7
  from the float32 model, so the remaining distance is the rounding of bf16 activations, not the
  math.
- Greedy speculative decoding with the block drafter emits exactly the plain tokens (up to 256 tokens,
  5 prompts).
- A proposal (one block of 7 rows, 3 layers, the head) takes 0.55 to 0.57 ms greedy and 0.67 to
  0.69 ms sampled, against 3.4 ms for a plain decode step (512 tokens, 5 prompts).
- A fresh drafter accepts nothing, the worst case. Verifying all 6 of its drafts every step
  (`block-fixed`) runs at x0.55 of plain decode greedy and x0.53 to x0.54 sampled: the verify
  step's 1.70 plus the proposal. The adaptive drafter (`block`) runs at x0.92 to x0.93 greedy and
  x0.94 sampled, proposing in 22 to 42 of up to 511 steps: it trusts the confidence head only as far
  as its predictions keep holding in this sequence, and sits out steps (1, 3, 7, then 15 at a time,
  with a proposal between) while its proposals' running gain is below their cost. What remains is
  the cost of the features themselves (the per-step context update and the target's spec step).

## Training recipe

- **Data**: `mlabonne/open-perfectblend` (DSpark's prompts: chat, math, code, instructions),
  deduplicated, 200,000 conversations plus 96 held out. Praxis regenerates every final answer with
  its own sampling (temperature 0.6, top-p 0.95, top-k 20), 20% of them with thinking on, answers up
  to 2,048 tokens; sequences are cut at 2,048 tokens for training.
- **Hidden states**: `dromos features` runs each sequence through the engine with the six taps and
  streams bf16 rows (24.6 KB per token on Praxis) through a bounded queue in /dev/shm to the
  trainer, which deletes what it reads. A 200k-sequence epoch at 1,000 tokens per sequence is
  about 5 TB of features, against under 50 GB free on the server's disk (98% full on 2026-10-24), so they are recomputed every
  epoch instead of stored. The final
  tap rebuilds the target's own distribution at every position (*measured* KL to the engine's
  logits 1e-4 to 3e-4 mean, from the bf16 rounding), so no logits are stored either.
- **Loss** (DSpark): per block row j, weight exp(-(j-1)/4), `0.1 CE(token) + 0.9 TV(p_target,
  q_draft)`, plus BCE of the confidence head against [argmax match, sum min(p, q)] (detached from
  the backbone). 512 random anchors per sequence whose first draft is an answer token.
- **Optimizer**: AdamW (0.9, 0.95), no weight decay, lr 3e-4 (DSpark's 6e-4 scaled to a 128-sequence
  batch), 4% warmup, cosine to 10%, clip 1.0, bf16 autocast, 3 epochs.

## Cost on the 5070 Ti

Praxis, 200,000 conversations, 3 epochs, from rates *measured* on the server's RTX 5070 Ti:

| Stage | Measured | For the full run (*computed*) |
|---|---|---|
| Answers (vLLM, `regenerate.py`) | 1,943 answer tok/s over a 384-prompt run, 677 tokens per answer, 16% of answers at the 2,048 cap; peak 13.2 GB GPU, 7.1 GB host | 135M tokens: **19.4 h** |
| Hidden states (`dromos features`), alone | 4,250 to 4,820 tok/s | |
| Trainer (`train_block.py`), alone | 2,500 to 2,900 blocks/s at 512 anchors, peak 6.9 GiB | |
| Both at once, as the pipeline runs them | 2,460 to 2,700 feature tok/s steady on 832-token sequences with 512 anchors (0.86 of the time the two take one after the other); peak 15.7 GB GPU | 0.31 to 0.34 s per sequence: **17 to 19 h per epoch** |
| Total | | **71 to 76 GPU-hours, about three days** |

Per sequence the pipeline costs L / R_features + A / R_trainer, times 0.86 for running side by side,
with L = 1,106 tokens (sequences cut at 2,048) and A = 355 anchors (at most 512, only on answer
tokens), the means of the regenerated smoke set. That set came from the local prompt mix (math,
code, reading comprehension, essays), not open-perfectblend, so the real lengths will differ; the
answer rate is the low end, since the smoke run's last minute was its longest answers with few
sequences in flight, which the pipeline's 4,096-prompt chunks amortize. The eval set (96
sequences, every 500 of the 4,688 steps) adds minutes.

Where the time goes: two thirds of each epoch is the target's forward pass over the corpus, one
sequence per prefill. Batching several sequences into one prefill in `dromos features` is the
first thing to try if the run needs to be shorter; the second is 2 epochs instead of 3 (18 h less).

Logos: not measured. The hidden states need the engine to load the FP8 checkpoint; the drafter's
side is *computed* at about 1.6 times Praxis's per block (206.5M parameters against 132.1M).

## Commands

```
# on the GPU server, in the engine checkout, after a build
tools/draft/pipeline.sh praxis
#   WORK=~/kronos-scratch/draft/praxis by default; N_TRAIN, EPOCHS, BATCH, LR, ... in the script's header
#   rerun the same command to resume after any stop; progress in $WORK/train.log, features.log

# then, the drafter it wrote:
build/dromos spec /home/aiadmin/kronos-scratch/logos/engines/praxis/hf out/spec_prompts.json \
    --drafters plain,ngram,block,block-fixed --draft-head ~/kronos-scratch/draft/praxis/train/draft
build/dromos spec ... --temp 0.6 --top-p 0.95 --top-k 20      # sampled, same drafter
```

The pipeline takes the GPU lock for each GPU stage, so it waits for other work and blocks it
while it runs.

Logos: the same command (`pipeline.sh logos`) once the engine loads the FP8 checkpoint (the
Formats workstream's loader); answers already run in vLLM, but the hidden states need the engine.
