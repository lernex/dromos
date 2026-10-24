"""lx_attention_varlen against a float32 PyTorch reference.

The reference rounds each step's keys and values to bf16 the way the cache stores them, then
attends over the cache in float32 with a causal limit per token. The cache is compared bit for
bit and the output within bf16 rounding. Caches start as NaN everywhere except the rows before
each sequence's start position, so any read past a token's visible range shows up as NaN.
"""

import ctypes
import math
from itertools import accumulate

import pytest
import torch

from conftest import DEVICE, close

NAN = float("nan")
torch.backends.cuda.matmul.allow_tf32 = False  # the reference is full float32


def caches(n_slots, capacity, kvh, dh, seqs):
    """K and V caches: NaN, except earlier steps' random rows before each sequence's start."""
    k = torch.full((n_slots, capacity, kvh, dh), NAN, device=DEVICE, dtype=torch.bfloat16)
    v = torch.full_like(k, NAN)
    for slot, start, _ in seqs:
        if start:
            k[slot, :start] = torch.randn(start, kvh, dh, device=DEVICE).to(torch.bfloat16)
            v[slot, :start] = torch.randn(start, kvh, dh, device=DEVICE).to(torch.bfloat16)
    return k, v


def reference(qkv, k_ref, v_ref, slots, starts, cu, H, KVH, Dh):
    """Causal GQA in float32. Writes this step's bf16-rounded k and v into k_ref and v_ref."""
    G = H // KVH
    scale = 1.0 / math.sqrt(Dh)
    qw, kw = H * Dh, KVH * Dh
    out = torch.empty(qkv.shape[0], H * Dh, device=DEVICE, dtype=torch.float32)
    for s, (slot, start) in enumerate(zip(slots, starts)):
        a, b = cu[s], cu[s + 1]
        n = b - a
        if n == 0:
            continue
        k_ref[slot, start:start + n] = qkv[a:b, qw:qw + kw].reshape(n, KVH, Dh).to(torch.bfloat16)
        v_ref[slot, start:start + n] = qkv[a:b, qw + kw:qw + 2 * kw].reshape(n, KVH, Dh).to(torch.bfloat16)
        for j in range(n):
            t = a + j
            # Explicit causal mask over the whole cache: token j sees rows 0 .. start + j.
            visible = torch.arange(k_ref.shape[1], device=DEVICE) < start + j + 1
            k = torch.where(visible[:, None, None], k_ref[slot].float(), 0.0)
            v = torch.where(visible[:, None, None], v_ref[slot].float(), 0.0)
            q = qkv[t, :qw].reshape(KVH, G, Dh) * scale
            scores = torch.einsum("kgd,lkd->kgl", q, k).masked_fill(~visible, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[t] = torch.einsum("kgl,lkd->kgd", p, v).reshape(-1)
    return out


def run(lx, qkv, col, k_cache, v_cache, slots, starts, cu, H, KVH, Dh):
    """One kernel step. qkv is a f32 buffer, possibly wider than the step; q | k | v start at col."""
    out = torch.empty(qkv.shape[0], H * Dh, device=DEVICE, dtype=torch.bfloat16)
    lx("attention_varlen", ctypes.c_void_p(qkv.data_ptr() + 4 * col), qkv.stride(0), k_cache, v_cache,
       slots, starts, cu, slots.numel(), k_cache.shape[1], out, H, KVH, Dh, 1.0 / math.sqrt(Dh))
    return out


def same_bits(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "cache rows differ from the reference"


def check_step(lx, *, H, KVH, Dh, capacity, seqs, n_slots, seed, col=0, pad=0):
    """Runs one varlen step; seqs is a list of (slot, start_pos, n_new) and is checked against the
    reference: the output within bf16 rounding, the cache bit for bit."""
    width = (H + 2 * KVH) * Dh
    lens = [n for _, _, n in seqs]
    cu = [0, *accumulate(lens)]
    slot_list, start_list = [s for s, _, _ in seqs], [p for _, p, _ in seqs]
    torch.manual_seed(seed)
    qkv = torch.randn(sum(lens), col + width + pad, device=DEVICE)
    qkv[:, :col] = NAN
    qkv[:, col + width:] = NAN
    k, v = caches(n_slots, capacity, KVH, Dh, seqs)
    k_ref, v_ref = k.clone(), v.clone()
    want = reference(qkv[:, col:col + width], k_ref, v_ref, slot_list, start_list, cu, H, KVH, Dh)
    t_slots = torch.tensor(slot_list, dtype=torch.int32, device=DEVICE)
    t_starts = torch.tensor(start_list, dtype=torch.int32, device=DEVICE)
    t_cu = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    out = run(lx, qkv, col, k, v, t_slots, t_starts, t_cu, H, KVH, Dh)
    close(out, want, atol=1e-2, rtol=1e-2)
    same_bits(k, k_ref)
    same_bits(v, v_ref)
    return out


@pytest.mark.parametrize("H,KVH", [(32, 8), (40, 10)], ids=["praxis", "logos"])
def test_metis_shapes(lx, H, KVH):
    check_step(lx, H=H, KVH=KVH, Dh=64, capacity=64, n_slots=1, seqs=[(0, 0, 37)], seed=H)


@pytest.mark.parametrize(
    "H,KVH,Dh",
    [(6, 2, 16), (3, 1, 8), (5, 1, 16), (10, 2, 24), (2, 2, 8), (9, 3, 72), (4, 1, 128)],
    ids=["h6kv2d16", "h3kv1d8", "g5-two-passes", "d24", "mha-g1", "d72-two-chunks", "d128"],
)
def test_odd_small_shapes(lx, H, KVH, Dh):
    check_step(lx, H=H, KVH=KVH, Dh=Dh, capacity=16, n_slots=2, seqs=[(1, 0, 9), (0, 4, 3)], seed=H + Dh)


def test_varlen_batch_mixed_starts(lx):
    # Slots out of order, a continuation, a decode, a sequence with no new tokens, and a fresh one.
    seqs = [(5, 0, 9), (2, 37, 1), (7, 100, 20), (0, 50, 0), (3, 0, 4)]
    check_step(lx, H=8, KVH=2, Dh=64, capacity=160, n_slots=8, seqs=seqs, seed=7)


def test_decode_batch(lx):
    seqs = [(s, 100 + 37 * s, 1) for s in range(16)]
    check_step(lx, H=32, KVH=8, Dh=64, capacity=700, n_slots=16, seqs=list(reversed(seqs)), seed=16)


def test_prefill_then_decode_matches_single_shot(lx):
    H, KVH, Dh, T, cap = 8, 2, 64, 30, 64
    width = (H + 2 * KVH) * Dh
    torch.manual_seed(3)
    qkv = torch.randn(T, width, device=DEVICE)
    slots = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    one_shot_cu = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    starts = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    k, v = caches(4, cap, KVH, Dh, [(2, 0, T)])
    k_ref, v_ref = k.clone(), v.clone()
    want = reference(qkv, k_ref, v_ref, [2], [0], [0, T], H, KVH, Dh)
    single = run(lx, qkv, 0, k, v, slots, starts, one_shot_cu, H, KVH, Dh)
    close(single, want, atol=1e-2, rtol=1e-2)

    # The same tokens as a 12-token prefill and then one token per step.
    kc, vc = caches(4, cap, KVH, Dh, [(2, 0, T)])
    parts = []
    for a, b in [(0, 12)] + [(j, j + 1) for j in range(12, T)]:
        cu = torch.tensor([0, b - a], dtype=torch.int32, device=DEVICE)
        start = torch.tensor([a], dtype=torch.int32, device=DEVICE)
        parts.append(run(lx, qkv[a:b], 0, kc, vc, slots, start, cu, H, KVH, Dh))
    close(torch.cat(parts), single, atol=1e-2, rtol=1e-2)
    same_bits(kc, k)
    same_bits(vc, v)


@pytest.mark.parametrize("start,n", [(4000, 1), (1000, 33), (32767, 1)], ids=["4k-decode", "1k-prefill33", "32k-decode"])
def test_long_context(lx, start, n):
    # A cache of `start` rows and n new tokens: up to 32768 visible positions. One spare row past
    # the visible end stays NaN, so a read one position too far shows up in the output.
    check_step(lx, H=32, KVH=8, Dh=64, capacity=start + n + 1, n_slots=1, seqs=[(0, start, n)], seed=start)


def test_qkv_column_slice_of_wider_buffer(lx):
    # q | k | v sit at column 7 of a buffer 40 columns wider than the step. Everything around them is NaN.
    check_step(lx, H=8, KVH=2, Dh=64, capacity=32, n_slots=2, seqs=[(1, 3, 5), (0, 0, 6)], seed=11,
               col=7, pad=21)
