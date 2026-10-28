"""Speculative decoding kernels (engine/cuda/spec.cu) against the plain kernels and PyTorch.

The Mamba-2 tests drive random verify sequences (fold some pending positions, keep some, drop the
rest, append new tokens) on several slots at once, and require every output, and the base state
after every fold, to be bit-identical to the plain kernels run over the accepted path. The sampling
tests check the filtered distribution against a PyTorch reference of sampler.cu's semantics, and the
acceptance kernels statistically: a speculative sampling loop over Markov-chain target and drafter
must emit token sequences distributed as plain sampling from the target, by a chi-square test with
fixed seeds (so the test is deterministic).
"""

import ctypes
import math
import random

import pytest
import torch

from conftest import close

NAN = float("nan")
MAX_PENDING = 8


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def cumulative(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return i32(cu)


# ---- Verify sequences ------------------------------------------------------------------------


class Track:
    """One sequence: the inputs of its folded positions (in the base state) and pending ones."""

    def __init__(self, slot, has):
        self.slot = slot
        self.has = has  # the base state was valid before the first step
        self.folded = []  # token indices into the input pool
        self.pending = []


def random_plan(rng, tracks, max_new):
    """Per track: (fold, keep, new count) honouring the kernels' bounds."""
    plan = []
    for tr in tracks:
        m = len(tr.pending)
        fold = rng.randint(0, m)
        keep = rng.randint(0, m - fold)
        new = rng.randint(0, min(max_new, MAX_PENDING - keep))
        plan.append((fold, keep, new))
    return plan


def apply_plan(tracks, plan, next_token):
    """Advances the tracks; returns each track's new token indices."""
    news = []
    for tr, (fold, keep, new) in zip(tracks, plan):
        tr.folded += tr.pending[:fold]
        kept = tr.pending[fold : fold + keep]
        ids = list(range(next_token, next_token + new))
        next_token += new
        tr.pending = kept + ids
        news.append(ids)
    return news, next_token


# ---- Causal conv ------------------------------------------------------------------------------


def plain_conv(lx, pool, ids, w, bias, init_state, has, C, W):
    """lx_conv1d_varlen over the tokens `ids` from `init_state`: (outputs, state after)."""
    state = init_state.clone()[None]
    if not ids:
        return torch.empty(0, C, device="cuda"), state[0]
    x = pool[ids].contiguous()
    out = torch.empty(len(ids), C, device="cuda")
    lx("conv1d_varlen", x, C, w, bias, state, i32([0]), i32([1 if has else 0]), cumulative([len(ids)]), 1, out, C, W)
    return out, state[0]


@pytest.mark.parametrize("C,W,steps,seed", [(6144, 4, 12, 0), (37, 4, 10, 1), (100, 2, 10, 2), (64, 8, 10, 3)])
def test_spec_conv1d_matches_plain(lx, C, W, steps, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    pool = torch.randn(steps * 3 * MAX_PENDING + 8, C, device="cuda")
    w = (torch.randn(C, W, device="cuda") * 0.5).to(torch.bfloat16)
    bias = (torch.randn(C, device="cuda") * 0.1).to(torch.bfloat16)
    n_slots = 5
    tracks = [Track(4, True), Track(1, False), Track(2, True)]
    init = torch.randn(n_slots, W - 1, C, device="cuda")
    state = init.clone()
    hist = torch.full((n_slots, MAX_PENDING, C), NAN, device="cuda")
    nxt = 0
    for step in range(steps):
        plan = random_plan(rng, tracks, MAX_PENDING)
        has = [1 if (tr.has or tr.folded) else 0 for tr in tracks]
        news, nxt = apply_plan(tracks, plan, nxt)
        lengths = [len(n) for n in news]
        T = sum(lengths)
        ld = C + 5
        x = torch.full((max(T, 1), ld), NAN, device="cuda")
        flat = [i for n in news for i in n]
        if flat:
            x[:T, :C] = pool[flat]
        out = torch.full((max(T, 1), C), NAN, device="cuda")
        lx("spec_conv1d", x, ld, w, bias, state, hist, i32([tr.slot for tr in tracks]), i32([tr.slot for tr in tracks]), i32(has),
           i32([p[0] for p in plan]), i32([p[1] for p in plan]), cumulative(lengths), len(tracks), out, C, W)
        start = 0
        for tr, ids, n in zip(tracks, news, lengths):
            path = tr.folded + tr.pending
            ref_out, _ = plain_conv(lx, pool, path, w, bias, init[tr.slot], tr.has, C, W)
            assert torch.equal(out[start : start + n], ref_out[len(path) - n :]), f"step {step}: outputs differ"
            if tr.has or tr.folded:
                _, ref_state = plain_conv(lx, pool, tr.folded, w, bias, init[tr.slot], tr.has, C, W)
                assert torch.equal(state[tr.slot], ref_state), f"step {step}: base state differs"
            start += n


# ---- Selective scan ---------------------------------------------------------------------------


def plain_scan(lx, pool, ids, init_state, has, H, P, G, N, consts):
    """lx_ssm_scan_varlen over the tokens `ids`: (y, state after)."""
    state = init_state.clone()[None]
    if not ids:
        return torch.empty(0, H, P, device="cuda"), state[0]
    x, dt, B, Cm = (pool[k][ids].contiguous() for k in ("x", "dt", "B", "C"))
    y = torch.empty(len(ids), H, P, device="cuda")
    lx("ssm_scan_varlen", x, H * P, dt, H, B, Cm, G * N, consts["A_log"], consts["D"], consts["dt_bias"], state,
       i32([0]), i32([1 if has else 0]), cumulative([len(ids)]), 1, y, H, P, G, N)
    return y, state[0]


SPEC_SCAN_CASES = [
    # H, P, G, N, steps, seed
    (64, 64, 8, 128, 10, 0),  # Praxis
    (80, 64, 10, 128, 6, 1),  # Logos
    (6, 5, 3, 20, 12, 2),  # odd sizes, N not a multiple of 4
    (4, 32, 2, 256, 8, 3),  # widest state
]


@pytest.mark.parametrize("H,P,G,N,steps,seed", SPEC_SCAN_CASES)
def test_spec_ssm_scan_matches_plain(lx, H, P, G, N, steps, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    size = steps * 3 * MAX_PENDING + 8
    pool = {
        "x": torch.randn(size, H * P, device="cuda") * 0.5,
        "dt": torch.randn(size, H, device="cuda"),
        "B": torch.randn(size, G * N, device="cuda") * 0.5,
        "C": torch.randn(size, G * N, device="cuda") * 0.5,
    }
    consts = {
        "A_log": torch.log(torch.rand(H, device="cuda") * 3.5 + 0.5),
        "D": torch.randn(H, device="cuda"),
        "dt_bias": torch.randn(H, device="cuda") * 0.5,
    }
    n_slots = 4
    tracks = [Track(3, True), Track(0, False), Track(1, True)]
    init = torch.randn(n_slots, H, P, N, device="cuda") * 0.5
    state = init.clone()
    hist_floats = lx.lib.lx_spec_ssm_hist_floats
    hist_floats.restype = ctypes.c_int64
    hist = torch.full((n_slots, hist_floats(H, P, N)), NAN, device="cuda")
    nxt = 0
    for step in range(steps):
        plan = random_plan(rng, tracks, MAX_PENDING)
        has = [1 if (tr.has or tr.folded) else 0 for tr in tracks]
        news, nxt = apply_plan(tracks, plan, nxt)
        lengths = [len(n) for n in news]
        T = max(sum(lengths), 1)
        flat = [i for n in news for i in n]
        ld_x, ld_dt, ld_bc = H * P + 7, H + 3, G * N + 9
        bufs = {}
        for key, width, ld in (("x", H * P, ld_x), ("dt", H, ld_dt), ("B", G * N, ld_bc), ("C", G * N, ld_bc)):
            b = torch.full((T, ld), NAN, device="cuda")
            if flat:
                b[: len(flat), :width] = pool[key][flat]
            bufs[key] = b
        y = torch.full((T, H, P), NAN, device="cuda")
        lx("spec_ssm_scan", bufs["x"], ld_x, bufs["dt"], ld_dt, bufs["B"], bufs["C"], ld_bc, consts["A_log"],
           consts["D"], consts["dt_bias"], state, hist, i32([tr.slot for tr in tracks]), i32([tr.slot for tr in tracks]), i32(has),
           i32([p[0] for p in plan]), i32([p[1] for p in plan]), cumulative(lengths), len(tracks), y, H, P, G, N)
        start = 0
        for tr, n in zip(tracks, lengths):
            path = tr.folded + tr.pending
            ref_y, _ = plain_scan(lx, pool, path, init[tr.slot], tr.has, H, P, G, N, consts)
            assert torch.equal(y[start : start + n], ref_y[len(path) - n :]), f"step {step}: y differs"
            if tr.has or tr.folded:
                _, ref_state = plain_scan(lx, pool, tr.folded, init[tr.slot], tr.has, H, P, G, N, consts)
                assert torch.equal(state[tr.slot], ref_state), f"step {step}: base state differs"
            start += n


def test_spec_ssm_verify_then_commit(lx):
    """The plain story: verify k = 7 drafts on top of a prefilled sequence, accept j of them, and the
    folded state equals feeding those j + 1 tokens normally, for every j."""
    H, P, G, N = 64, 64, 8, 128
    torch.manual_seed(7)
    L, K = 40, 8
    pool = {
        "x": torch.randn(L + K, H * P, device="cuda") * 0.5,
        "dt": torch.randn(L + K, H, device="cuda"),
        "B": torch.randn(L + K, G * N, device="cuda") * 0.5,
        "C": torch.randn(L + K, G * N, device="cuda") * 0.5,
    }
    consts = {
        "A_log": torch.log(torch.rand(H, device="cuda") * 3.5 + 0.5),
        "D": torch.randn(H, device="cuda"),
        "dt_bias": torch.randn(H, device="cuda") * 0.5,
    }
    zero = torch.zeros(H, P, N, device="cuda")
    _, prefilled = plain_scan(lx, pool, list(range(L)), zero, False, H, P, G, N, consts)
    hist_floats = lx.lib.lx_spec_ssm_hist_floats
    hist_floats.restype = ctypes.c_int64
    for j in range(K + 1):
        state = prefilled.clone()[None]
        hist = torch.full((1, hist_floats(H, P, N)), NAN, device="cuda")
        ids = list(range(L, L + K))
        x, dt, B, Cm = (pool[k][ids].contiguous() for k in ("x", "dt", "B", "C"))
        y = torch.empty(K, H, P, device="cuda")
        lx("spec_ssm_scan", x, H * P, dt, H, B, Cm, G * N, consts["A_log"], consts["D"], consts["dt_bias"], state,
           hist, i32([0]), i32([0]), i32([1]), i32([0]), i32([0]), cumulative([K]), 1, y, H, P, G, N)
        assert torch.equal(state[0], prefilled), "a verify step must not touch the base state"
        empty = torch.empty(1, H * P + G * N * 2 + H, device="cuda")
        lx("spec_ssm_scan", empty, H * P, empty, H, empty, empty, G * N, consts["A_log"], consts["D"],
           consts["dt_bias"], state, hist, i32([0]), i32([0]), i32([1]), i32([j]), i32([0]), cumulative([0]), 1, empty, H, P,
           G, N)
        _, want = plain_scan(lx, pool, list(range(L + j)), zero, False, H, P, G, N, consts)
        assert torch.equal(state[0], want), f"commit {j}: state differs"


# ---- Batch-invariant f32 GEMV -----------------------------------------------------------------


@pytest.mark.parametrize("N,K", [(128, 1280), (192, 1280), (256, 6400), (256, 7936), (1, 256), (37, 333)])
def test_spec_gemv_f32(lx, N, K):
    torch.manual_seed(0)
    M = 8
    ld_x, ld_y = K + 3, N + 5
    x = torch.randn(M, ld_x, device="cuda")
    w = torch.randn(N, K, device="cuda") * 0.05
    bias = torch.randn(N, device="cuda")
    y = torch.full((M, ld_y), NAN, device="cuda")
    lx("spec_gemv_f32", x, ld_x, w, bias, y, ld_y, M, N, K)
    want = x[:, :K].double() @ w.double().T + bias.double()
    close(y[:, :N], want, atol=2e-5 * want.abs().max().item())
    # every row alone, and every prefix of rows, gives the same bits
    for m in range(1, M + 1):
        part = torch.full((m, ld_y), NAN, device="cuda")
        lx("spec_gemv_f32", x[:m].contiguous(), ld_x, w, bias, part, ld_y, m, N, K)
        assert torch.equal(part[:, :N], y[:m, :N]), f"rows differ with M = {m}"
    for r in range(M):
        one = torch.full((1, ld_y), NAN, device="cuda")
        lx("spec_gemv_f32", x[r : r + 1].contiguous(), ld_x, w, None, one, ld_y, 1, N, K)
        ref = torch.full((1, ld_y), NAN, device="cuda")
        lx("spec_gemv_f32", x[r : r + 1].contiguous(), ld_x, w, bias, ref, ld_y, 1, N, K)
        assert torch.equal(ref[:, :N], y[r : r + 1, :N])


# ---- The filtered distribution ------------------------------------------------------------------


class Filter(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_p", ctypes.c_float), ("top_k", ctypes.c_int32),
                ("min_p", ctypes.c_float)]


class Rng(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_uint64), ("position", ctypes.c_uint64)]


def struct_tensor(items, cls):
    """An array of ctypes structures as a CUDA byte tensor."""
    raw = (cls * len(items))(*items)
    host = torch.frombuffer(bytearray(bytes(raw)), dtype=torch.uint8)
    return host.cuda()


def rng_tensor(seeds, positions):
    """lx_spec_rng [n] from int64 tensors (or a scalar position): {uint64 seed, uint64 position}."""
    if not torch.is_tensor(positions):
        positions = torch.full_like(seeds, positions)
    return torch.stack([seeds.long(), positions.long()], 1).contiguous()


def reference_probs(logits, temperature, top_p, top_k, min_p):
    """sampler.cu's distribution of one row, in PyTorch: scaled logits in f32 as the kernel forms
    them, the kept set by value thresholds, the masses in float64."""
    V = logits.shape[0]
    if not temperature > 0:
        out = torch.zeros(V, dtype=torch.float64, device=logits.device)
        out[int(torch.argmax(logits))] = 1.0  # argmax returns the first maximum
        return out
    inv_t = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(temperature, dtype=torch.float32)
    w = logits.float() * inv_t.to(logits.device)
    top = w.max()
    keep = torch.ones(V, dtype=torch.bool, device=w.device)
    if 0 < top_k < V:
        kth = torch.sort(w, descending=True).values[top_k - 1]
        keep &= w >= kth
    if 0 < top_p < 1:
        cand = w[keep]
        e = torch.exp((cand - top).double())
        target = top_p * e.sum()
        # the largest value whose tokens at or above it hold the target mass: in descending order, the
        # value where the running mass first reaches the target (ties only add mass at that value)
        order = torch.argsort(cand, descending=True)
        running = torch.cumsum(e[order], 0)
        first = int(torch.nonzero(running >= target)[0])
        keep &= w >= cand[order[first]]
    if min_p > 0:
        thr = top + torch.log(torch.tensor(min(min_p, 1.0), dtype=torch.float32, device=w.device))
        keep &= w >= thr
    e = torch.where(keep, torch.exp((w - top).double()), torch.zeros((), dtype=torch.float64, device=w.device))
    return e / e.sum()


FILTERS = [
    (1.0, 1.0, 0, 0.0),
    (0.7, 0.9, 0, 0.0),
    (0.8, 1.0, 40, 0.0),
    (1.3, 0.95, 50, 0.05),
    (0.6, 1.0, 0, 0.1),
    (1.0, 0.5, 3000, 0.0),  # more than 1024 candidates: the vocabulary-wide top-p search
    (0.0, 1.0, 0, 0.0),  # greedy
]


@pytest.mark.parametrize("V", [65536, 1000, 17])
def test_spec_probs(lx, V):
    torch.manual_seed(V)
    rows = len(FILTERS)
    logits = torch.randn(rows, V, device="cuda") * 3.0
    logits[-1, V // 2] = logits[-1].max() + 1.0
    logits[-1, V // 3] = logits[-1, V // 2]  # a tie for the argmax: the lower index wins
    filters = struct_tensor([Filter(*f) for f in FILTERS], Filter)
    probs = torch.full((rows, V), NAN, device="cuda")
    lx("spec_probs", logits, V, filters, rows, probs)
    for r, f in enumerate(FILTERS):
        want = reference_probs(logits[r], *f)
        got = probs[r]
        assert torch.equal(got > 0, want > 0), f"filter {f}: kept sets differ"
        close(got, want, atol=1e-6, rtol=1e-5)


# ---- Greedy verification ------------------------------------------------------------------------


def test_spec_verify_greedy(lx):
    torch.manual_seed(0)
    V = 5000
    # three sequences: 4 drafts (2 hold), 0 drafts, 3 drafts (all hold)
    lengths = [5, 1, 4]
    T = sum(lengths)
    logits = torch.randn(T, V, device="cuda")
    amax = logits.argmax(-1).tolist()
    tokens = [0] * T
    # seq 0: rows 0..4; drafts are tokens 1..4: hold, hold, miss, (any)
    tokens[1], tokens[2], tokens[3], tokens[4] = amax[0], amax[1], (amax[2] + 1) % V, 7
    # seq 2: rows 6..9, all drafts hold
    tokens[7], tokens[8], tokens[9] = amax[6], amax[7], amax[8]
    # a tie in row 6 between the argmax and a lower index: the lower index is the greedy token
    lo = min(amax[6], 3)
    if lo != amax[6]:
        logits[6, lo] = logits[6, amax[6]]
        tokens[7] = lo
    out_amax = torch.empty(T, dtype=torch.int32, device="cuda")
    acc = torch.empty(3, dtype=torch.int32, device="cuda")
    nxt = torch.empty(3, dtype=torch.int32, device="cuda")
    lx("spec_verify_greedy", logits, V, i32(tokens), cumulative(lengths), 3, T, out_amax, acc, nxt)
    a = out_amax.tolist()
    assert acc.tolist() == [2, 0, 3]
    assert nxt.tolist() == [a[2], a[5], a[9]]
    assert a[6] == tokens[7]


# ---- Speculative sampling: exactness --------------------------------------------------------------


def chi_square_p(counts, probs):
    """Pearson's test of counts against probabilities, pooling cells whose expected count is under 5.
    Returns the p-value (upper tail). A sample in a cell of probability zero fails outright."""
    counts = counts.double().flatten()
    probs = probs.double().flatten()
    if bool((counts[probs == 0] > 0).any()):
        return 0.0
    expected = probs * counts.sum()
    order = torch.argsort(expected)
    counts, expected = counts[order], expected[order]
    stat, cells = 0.0, 0
    acc_c, acc_e = 0.0, 0.0
    for c, e in zip(counts.tolist(), expected.tolist()):
        if e == 0:
            continue
        acc_c += c
        acc_e += e
        if acc_e >= 5.0:
            stat += (acc_c - acc_e) ** 2 / acc_e
            cells += 1
            acc_c, acc_e = 0.0, 0.0
    if acc_e > 0:
        stat += (acc_c - acc_e) ** 2 / acc_e
        cells += 1
    dof = cells - 1
    if dof < 1:
        return 1.0
    return float(torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64),
                                         torch.tensor(stat / 2.0, dtype=torch.float64)))


def markov_table(lx, logits, filt):
    """Filtered next-token distributions of a Markov chain: row v is the distribution after token v."""
    V = logits.shape[0]
    probs = torch.empty(V, V, device="cuda")
    lx("spec_probs", logits.contiguous(), V, struct_tensor([Filter(*filt)] * V, Filter), V, probs)
    return probs


def run_spec_chain(lx, P, Q, k, n_seq, length, seed, point_mass):
    """Speculative sampling of n_seq sequences from the Markov chain P with drafts from Q (sampled from
    its rows, or Q's argmax when point_mass). Returns the first `length` emitted tokens of each."""
    V = P.shape[0]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    last = torch.zeros(n_seq, dtype=torch.long, device="cuda")  # every sequence starts after token 0
    out = torch.full((n_seq, length), -1, dtype=torch.long, device="cuda")
    filled = torch.zeros(n_seq, dtype=torch.long, device="cuda")
    position = 0
    while int(filled.min()) < length:
        # drafts d_1..d_k from Q, q rows and p rows per sequence: rows j = 0..k
        drafts = torch.empty(n_seq, k, dtype=torch.long, device="cuda")
        prev = last
        for j in range(k):
            if point_mass:
                d = Q[prev].argmax(-1)
            else:
                d = torch.multinomial(Q[prev], 1, generator=gen).squeeze(1)
            drafts[:, j] = d
            prev = d
        fed = torch.cat([last[:, None], drafts], 1)  # [n_seq, k + 1]: the pending token, then drafts
        p_rows = P[fed].reshape(n_seq * (k + 1), V).contiguous()
        q_rows = Q[fed].reshape(n_seq * (k + 1), V).contiguous()
        tokens = fed.reshape(-1).to(torch.int32).contiguous()
        cu = torch.arange(0, n_seq * (k + 1) + 1, k + 1, dtype=torch.int32, device="cuda")
        rngs = rng_tensor(seed * 1000003 + torch.arange(n_seq, device="cuda"), position)
        acc = torch.empty(n_seq, dtype=torch.int32, device="cuda")
        nxt = torch.empty(n_seq, dtype=torch.int32, device="cuda")
        lx("spec_verify", p_rows, None if point_mass else q_rows, V, tokens, cu, rngs, n_seq, acc, nxt)
        acc, nxt = acc.long(), nxt.long()
        # emitted this round: drafts[:, :acc], then nxt
        emitted = torch.cat([drafts, torch.zeros(n_seq, 1, dtype=torch.long, device="cuda")], 1)
        emitted.scatter_(1, acc[:, None], nxt[:, None])
        count = acc + 1
        for j in range(k + 1):
            take = (j < count) & (filled < length)
            idx = torch.nonzero(take).squeeze(1)
            out[idx, filled[idx]] = emitted[idx, j]
            filled = filled + take.long()
        last = nxt
        position += k + 1  # positions only need to be distinct per sequence across rounds
    return out


def joint_probs(P, length):
    """Exact distribution of the first `length` tokens after token 0 under the chain P."""
    V = P.shape[0]
    joint = P[0].double()
    for _ in range(length - 1):
        joint = joint[..., None] * P.double().reshape((1,) * (joint.dim() - 1) + (V, V))
    return joint


SPEC_SAMPLING_CASES = [
    # target filter, drafter filter, k, point-mass drafts, drafter is the target itself
    ((1.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), 3, False, False),
    ((0.8, 0.9, 6, 0.05), (1.2, 1.0, 0, 0.0), 4, False, False),
    ((0.7, 1.0, 0, 0.1), (0.7, 1.0, 0, 0.1), 2, False, True),  # every draft holds
    ((1.0, 0.85, 0, 0.0), (1.0, 1.0, 0, 0.0), 3, True, False),  # n-gram style drafts (a point mass)
    ((0.9, 1.0, 5, 0.0), (1.0, 1.0, 0, 0.0), 1, False, False),
]


@pytest.mark.parametrize("target,drafter,k,point_mass,same", SPEC_SAMPLING_CASES)
def test_spec_verify_is_exact(lx, target, drafter, k, point_mass, same):
    torch.manual_seed(11)
    V, L, n_seq = 8, 3, 1 << 18
    P = markov_table(lx, torch.randn(V, V, device="cuda") * 1.5, target)
    q_logits = torch.randn(V, V, device="cuda") * 1.5
    Q = P.clone() if same else markov_table(lx, q_logits, drafter)
    seqs = run_spec_chain(lx, P, Q, k, n_seq, L, seed=5, point_mass=point_mass)
    code = (seqs[:, 0] * V + seqs[:, 1]) * V + seqs[:, 2]
    counts = torch.bincount(code, minlength=V**L)
    want = joint_probs(P.cpu(), L).cuda()
    p = chi_square_p(counts, want)
    assert p > 1e-4, f"speculative samples do not follow the target (chi-square p = {p:.3g})"
    # and plain sampling through the same kernel (no drafts) passes the same test
    plain = run_spec_chain(lx, P, Q, 0, n_seq, L, seed=6, point_mass=True)
    code = (plain[:, 0] * V + plain[:, 1]) * V + plain[:, 2]
    p_plain = chi_square_p(torch.bincount(code, minlength=V**L), want)
    assert p_plain > 1e-4, f"plain samples do not follow the target (chi-square p = {p_plain:.3g})"


def test_spec_verify_greedy_target_is_deterministic(lx):
    """A greedy target (temperature 0) through lx_spec_verify emits its argmax chain exactly, whatever
    the drafter proposes."""
    torch.manual_seed(3)
    V = 8
    P = markov_table(lx, torch.randn(V, V, device="cuda"), (0.0, 1.0, 0, 0.0))
    Q = markov_table(lx, torch.randn(V, V, device="cuda"), (1.0, 1.0, 0, 0.0))
    seqs = run_spec_chain(lx, P, Q, 3, 4096, 6, seed=1, point_mass=False)
    chain, t = [], 0
    for _ in range(6):
        t = int(P[t].argmax())
        chain.append(t)
    assert (seqs == torch.tensor(chain, device="cuda")).all()


def test_spec_sample(lx):
    torch.manual_seed(2)
    V, n = 50, 1 << 17
    probs = torch.softmax(torch.randn(1, V, device="cuda") * 2, -1).contiguous()
    rows = torch.zeros(n, dtype=torch.int32, device="cuda")
    rngs = rng_tensor(torch.full((n,), 99, device="cuda"), torch.arange(n, device="cuda"))
    tokens = torch.empty(n, dtype=torch.int32, device="cuda")
    lx("spec_sample", probs, V, rows, rngs, ctypes.c_uint32(0x44524654), n, tokens)
    p = chi_square_p(torch.bincount(tokens.long(), minlength=V), probs[0])
    assert p > 1e-4, f"draws do not follow probs (chi-square p = {p:.3g})"


# ---- Block drafter kernels ---------------------------------------------------------------------


def head_norm_rope_reference(x, w, eps, pos0, theta):
    """x [rows, heads, Dh] in float64: per-head RMS norm (when w is given), then RoPE (rotate halves)."""
    rows, _, Dh = x.shape
    if w is not None:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.double()
    inv = theta ** (-torch.arange(0, Dh, 2, dtype=torch.float64, device=x.device) / Dh)
    angle = (pos0 + torch.arange(rows, dtype=torch.float64, device=x.device))[:, None] * inv
    c, s = torch.cos(angle)[:, None, :], torch.sin(angle)[:, None, :]
    x1, x2 = x[..., : Dh // 2], x[..., Dh // 2 :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


@pytest.mark.parametrize("rows,heads,Dh,norm,pos0", [
    (7, 32, 64, True, 0), (7, 8, 64, True, 131071), (3, 10, 128, False, 12345), (8, 4, 256, True, 9), (1, 1, 2, True, 3),
])
def test_spec_head_norm_rope(lx, rows, heads, Dh, norm, pos0):
    gen = torch.Generator(device="cuda").manual_seed(rows * 1000 + heads * 10 + Dh)
    pad = 5
    full = torch.randn(rows, heads * Dh + pad, generator=gen, device="cuda") * 3
    before = full.clone()
    w = (torch.randn(Dh, generator=gen, device="cuda") * 0.5 + 1).to(torch.bfloat16) if norm else None
    lx("spec_head_norm_rope", full, heads * Dh + pad, rows, heads, Dh, w, 1e-6, ctypes.c_int64(pos0),
       ctypes.c_double(10000.0))
    ref = head_norm_rope_reference(before[:, : heads * Dh].double().view(rows, heads, Dh), w, 1e-6, pos0, 10000.0)
    close(full[:, : heads * Dh].view(rows, heads, Dh), ref, atol=2e-5, rtol=2e-5)
    assert torch.equal(full[:, heads * Dh :], before[:, heads * Dh :])  # the stride's padding is untouched


def block_attention_reference(q, k_ctx, v_ctx, k_blk, v_blk, H, KVH, Dh, scale):
    """float64: every block row attends over the context rows and all block rows."""
    B = q.shape[0]
    keys = torch.cat([k_ctx.view(-1, KVH, Dh), k_blk.view(B, KVH, Dh)]).double()
    values = torch.cat([v_ctx.view(-1, KVH, Dh), v_blk.view(B, KVH, Dh)]).double()
    qh = q.view(B, H, Dh).double()
    kv = torch.arange(H, device=q.device) // (H // KVH)
    scores = torch.einsum("bhd,nhd->bhn", qh, keys[:, kv]) * scale
    return torch.einsum("bhn,nhd->bhd", scores.softmax(-1), values[:, kv]).reshape(B, H * Dh)


@pytest.mark.parametrize("B,H,KVH,Dh,n_ctx", [
    (7, 32, 8, 64, 256), (6, 40, 10, 64, 255), (7, 32, 8, 64, 0), (1, 4, 4, 64, 1), (8, 16, 2, 128, 3000),
])
def test_spec_block_attention(lx, B, H, KVH, Dh, n_ctx):
    gen = torch.Generator(device="cuda").manual_seed(B * 100 + n_ctx)
    ld_q, ld_blk, ld_out = H * Dh + 8, 3 * KVH * Dh, H * Dh + 4  # strided rows, as the engine passes them
    q = torch.randn(B, ld_q, generator=gen, device="cuda")
    k_ctx = torch.randn(max(n_ctx, 1), KVH, Dh, generator=gen, device="cuda")[:n_ctx].contiguous()
    v_ctx = torch.randn(max(n_ctx, 1), KVH, Dh, generator=gen, device="cuda")[:n_ctx].contiguous()
    kv_blk = torch.randn(B, ld_blk, generator=gen, device="cuda")
    k_blk, v_blk = kv_blk[:, : KVH * Dh], kv_blk[:, KVH * Dh : 2 * KVH * Dh]
    out = torch.full((B, ld_out), NAN, device="cuda")
    scale = 1.0 / math.sqrt(Dh)
    v_ptr = ctypes.c_void_p(kv_blk.data_ptr() + 4 * KVH * Dh)
    lx("spec_block_attention", q, ld_q, k_ctx if n_ctx else None, v_ctx if n_ctx else None, n_ctx, kv_blk, v_ptr,
       ld_blk, out, ld_out, B, H, KVH, Dh, scale)
    ref = block_attention_reference(q[:, : H * Dh], k_ctx, v_ctx, k_blk.contiguous(), v_blk.contiguous(), H, KVH,
                                    Dh, scale)
    close(out[:, : H * Dh], ref, atol=2e-5, rtol=2e-5)
    assert torch.isnan(out[:, H * Dh :]).all()
