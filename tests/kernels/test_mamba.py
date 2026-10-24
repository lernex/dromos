"""Mamba-2 mixer kernels against sequential float64 PyTorch references.

Inputs are strided: each activation sits in a wider buffer whose unused columns are NaN, so a
kernel that reads past its width or misses a row shows up as NaN in the output.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import close

NAN = float("nan")


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def cumulative(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return i32(cu)


def nan_padded(rows, width, data):
    buf = torch.full((rows, width), NAN, device="cuda")
    buf[:, : data.shape[1]] = data
    return buf


# ---- Causal conv -----------------------------------------------------------------------------


def conv_reference(x, w, bias, state, slots, has, lengths, W):
    x, w, bias = x.double(), w.double(), bias.double()
    state = state.double().clone()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device="cuda")
    start = 0
    for s, (slot, n) in enumerate(zip(slots, lengths)):
        C = x.shape[1]
        hist = state[slot] if has[s] else torch.zeros(W - 1, C, dtype=torch.float64, device="cuda")
        full = torch.cat([hist, x[start : start + n]], 0)
        windows = full.unfold(0, W, 1)  # [n, C, W]: windows[t, c, k] = full[t + k, c]
        out[start : start + n] = F.silu(bias + (windows * w).sum(-1))
        state[slot] = full[n:]
        start += n
    return out, state


CONV_CASES = [
    # C, W, lengths, slot ids (unsorted), has_state
    (6144, 4, [1, 3, 17, 300], [4, 1, 5, 0], [1, 0, 1, 1]),
    (37, 4, [1, 2, 1, 5], [2, 0, 3, 1], [0, 1, 1, 0]),
    (100, 2, [2, 9, 1], [3, 0, 1], [1, 1, 0]),
    (64, 3, [2048], [0], [0]),
]


@pytest.mark.parametrize("C,W,lengths,slots,has", CONV_CASES)
def test_conv1d_varlen(lx, C, W, lengths, slots, has):
    torch.manual_seed(0)
    T = sum(lengths)
    x = nan_padded(T, C + 11, torch.randn(T, C, device="cuda"))
    w = (torch.randn(C, W, device="cuda") * 0.5).to(torch.bfloat16)
    bias = (torch.randn(C, device="cuda") * 0.1).to(torch.bfloat16)
    state = torch.randn(max(slots) + 1, W - 1, C, device="cuda")
    out = torch.full((T, C), NAN, device="cuda")
    state_k = state.clone()

    want, want_state = conv_reference(x[:, :C], w, bias, state, slots, has, lengths, W)
    lx("conv1d_varlen", x, x.shape[1], w, bias, state_k, i32(slots), i32(has), cumulative(lengths),
       len(lengths), out, C, W)
    close(out, want.float(), atol=1e-5, rtol=1e-4)
    close(state_k, want_state.float(), atol=1e-6)


def test_conv1d_decode_matches_prefill(lx):
    """Prefill, then one token per call, must match the whole sequence in one call."""
    torch.manual_seed(1)
    C, W, L, K = 6144, 4, 37, 9
    x = torch.randn(L + K, C, device="cuda")
    w = (torch.randn(C, W, device="cuda") * 0.5).to(torch.bfloat16)
    bias = (torch.randn(C, device="cuda") * 0.1).to(torch.bfloat16)

    whole_state = torch.zeros(1, W - 1, C, device="cuda")
    whole = torch.empty(L + K, C, device="cuda")
    lx("conv1d_varlen", x, C, w, bias, whole_state, i32([0]), i32([0]), cumulative([L + K]), 1, whole, C, W)

    step_state = torch.zeros(2, W - 1, C, device="cuda")
    parts = []
    prefill = torch.empty(L, C, device="cuda")
    lx("conv1d_varlen", x[:L], C, w, bias, step_state, i32([1]), i32([0]), cumulative([L]), 1, prefill, C, W)
    parts.append(prefill)
    for i in range(K):
        step = torch.empty(1, C, device="cuda")
        lx("conv1d_varlen", x[L + i : L + i + 1], C, w, bias, step_state, i32([1]), i32([1]),
           cumulative([1]), 1, step, C, W)
        parts.append(step)
    close(torch.cat(parts), whole, atol=1e-6)
    close(step_state[1], whole_state[0], atol=1e-6)


# ---- Selective scan --------------------------------------------------------------------------


def scan_reference(x, dt_raw, B, C, A_log, D, dt_bias, state, slots, has, lengths, H, P, G, N):
    x, dt_raw, B, C = x.double(), dt_raw.double(), B.double(), C.double()
    A_log, D, dt_bias = A_log.double(), D.double(), dt_bias.double()
    state = state.double().clone()
    y = torch.zeros(x.shape[0], H, P, dtype=torch.float64, device="cuda")
    negA = -torch.exp(A_log)
    rep = H // G
    start = 0
    for s, (slot, n) in enumerate(zip(slots, lengths)):
        S = state[slot].clone() if has[s] else torch.zeros(H, P, N, dtype=torch.float64, device="cuda")
        for t in range(start, start + n):
            dt = F.softplus(dt_raw[t] + dt_bias)  # [H]
            decay = torch.exp(dt * negA)
            Bh = B[t, : G * N].reshape(G, N).repeat_interleave(rep, 0)  # [H, N]
            Ch = C[t, : G * N].reshape(G, N).repeat_interleave(rep, 0)
            xt = x[t, : H * P].reshape(H, P)
            S = decay[:, None, None] * S + (dt[:, None] * xt)[:, :, None] * Bh[:, None, :]
            y[t] = (S * Ch[:, None, :]).sum(-1) + D[:, None] * xt
        state[slot] = S
        start += n
    return y, state


def scan_inputs(H, P, G, N, lengths, n_slots, seed, pad_bc=9):
    """pad_bc is the gap between B and C rows: 0 makes them contiguous, 9 makes them strided."""
    torch.manual_seed(seed)
    T = sum(lengths)
    ld_x, ld_dt, ld_bc = H * P + 7, H + 3, G * N + pad_bc
    x = nan_padded(T, ld_x, torch.randn(T, H * P, device="cuda") * 0.5)
    dt_raw = nan_padded(T, ld_dt, torch.randn(T, H, device="cuda"))
    B = nan_padded(T, ld_bc, torch.randn(T, G * N, device="cuda") * 0.5)
    C = nan_padded(T, ld_bc, torch.randn(T, G * N, device="cuda") * 0.5)
    A_log = torch.log(torch.rand(H, device="cuda") * 3.5 + 0.5)
    D = torch.randn(H, device="cuda")
    dt_bias = torch.randn(H, device="cuda") * 0.5
    state = torch.randn(n_slots, H, P, N, device="cuda") * 0.5  # junk for has_state == 0
    return dict(x=x, ld_x=ld_x, dt_raw=dt_raw, ld_dt=ld_dt, B=B, C=C, ld_bc=ld_bc,
                A_log=A_log, D=D, dt_bias=dt_bias, state=state)


def run_scan(lx, inp, slots, has, lengths, H, P, G, N, state):
    T = sum(lengths)
    y = torch.full((T, H, P), NAN, device="cuda")
    lx("ssm_scan_varlen", inp["x"], inp["ld_x"], inp["dt_raw"], inp["ld_dt"], inp["B"], inp["C"],
       inp["ld_bc"], inp["A_log"], inp["D"], inp["dt_bias"], state, i32(slots), i32(has),
       cumulative(lengths), len(lengths), y, H, P, G, N)
    return y


SCAN_CASES = [
    # H, P, G, N, lengths, slot ids (unsorted), has_state, pad_bc
    (64, 64, 8, 128, [1, 3, 17, 300], [4, 1, 5, 0], [1, 0, 1, 1], 9),  # Praxis shape, strided B/C
    (80, 64, 8, 128, [5, 64], [1, 0], [0, 1], 0),  # Logos head count, contiguous B/C
    (6, 5, 3, 20, [1, 4, 9], [2, 0, 1], [1, 0, 0], 0),  # odd sizes, N not a multiple of 4
    (4, 32, 2, 256, [33], [0], [0], 0),  # widest state
    (2, 16, 1, 16, [2, 3], [1, 0], [1, 0], 3),
    (64, 64, 8, 128, [2048], [0], [0], 0),  # a few thousand tokens, one sequence
]


@pytest.mark.parametrize("H,P,G,N,lengths,slots,has,pad_bc", SCAN_CASES)
def test_ssm_scan_varlen(lx, H, P, G, N, lengths, slots, has, pad_bc):
    n_slots = max(slots) + 1
    inp = scan_inputs(H, P, G, N, lengths, n_slots, seed=2, pad_bc=pad_bc)
    state_k = inp["state"].clone()
    want_y, want_state = scan_reference(
        inp["x"][:, : H * P], inp["dt_raw"][:, :H], inp["B"][:, : G * N], inp["C"][:, : G * N],
        inp["A_log"], inp["D"], inp["dt_bias"], inp["state"], slots, has, lengths, H, P, G, N,
    )
    y = run_scan(lx, inp, slots, has, lengths, H, P, G, N, state_k)
    close(y, want_y.float(), atol=1e-4 * want_y.abs().max().item())
    close(state_k, want_state.float(), atol=1e-4 * want_state.abs().max().item())


def test_ssm_scan_decode_matches_prefill(lx):
    """Prefill, then one token per call, must match the whole sequence in one call."""
    H, P, G, N, L, K = 64, 64, 8, 128, 40, 8
    inp = scan_inputs(H, P, G, N, [L + K], 2, seed=3)
    whole_state = torch.zeros(2, H, P, N, device="cuda")
    whole = run_scan(lx, inp, [1], [0], [L + K], H, P, G, N, whole_state)

    step_state = torch.zeros(2, H, P, N, device="cuda")
    parts = []
    parts.append(run_scan(lx, _slice(inp, 0, L), [1], [0], [L], H, P, G, N, step_state))
    for i in range(K):
        parts.append(run_scan(lx, _slice(inp, L + i, L + i + 1), [1], [1], [1], H, P, G, N, step_state))
    close(torch.cat(parts), whole, atol=1e-5 * whole.abs().max().item())
    close(step_state[1], whole_state[1], atol=1e-5 * whole_state.abs().max().item())


def _slice(inp, t0, t1):
    """The same buffers, restricted to tokens [t0, t1) (row slices of contiguous tensors)."""
    out = dict(inp)
    for key in ("x", "dt_raw", "B", "C"):
        out[key] = inp[key][t0:t1].contiguous()
    return out


# ---- Gated RMSNorm --------------------------------------------------------------------------


def gated_norm_reference(y, z, w, groups, eps):
    v = y.double() * F.silu(z.double())
    T, I = v.shape
    g = v.reshape(T, groups, I // groups)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    return g.reshape(T, I) * w.double()


@pytest.mark.parametrize("T,I,groups", [(1, 4096, 8), (7, 4096, 8), (33, 4096, 8), (5, 60, 5), (3, 96, 3), (2, 64, 1)])
def test_gated_rmsnorm(lx, T, I, groups):
    torch.manual_seed(0)
    eps = 1e-5
    y = torch.randn(T, I, device="cuda")
    z = nan_padded(T, I + 5, torch.randn(T, I, device="cuda"))
    w = (torch.rand(I, device="cuda") + 0.5).to(torch.bfloat16)
    out = torch.full((T, I), NAN, device="cuda", dtype=torch.bfloat16)
    lx("gated_rmsnorm", y, z, z.shape[1], w, out, T, I, groups, eps)
    want = gated_norm_reference(y, z[:, :I], w, groups, eps)
    close(out, want, atol=1e-6, rtol=1e-2)
