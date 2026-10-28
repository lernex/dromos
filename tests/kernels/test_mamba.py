"""Mamba-2 mixer kernels against sequential float64 PyTorch references.

Inputs are strided: each activation sits in a wider buffer whose unused columns are NaN, so a
kernel that reads past its width or misses a row shows up as NaN in the output.
"""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import DEVICE, close

NAN = float("nan")


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def cumulative(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return i32(cu)


def nan_padded(rows, width, data):
    buf = torch.full((rows, width), NAN, device=DEVICE)
    buf[:, : data.shape[1]] = data
    return buf


# ---- Causal conv -----------------------------------------------------------------------------


def conv_reference(x, w, bias, state, slots, has, lengths, W):
    x, w, bias = x.double(), w.double(), bias.double()
    state = state.double().clone()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=DEVICE)
    start = 0
    for s, (slot, n) in enumerate(zip(slots, lengths)):
        C = x.shape[1]
        hist = state[slot] if has[s] else torch.zeros(W - 1, C, dtype=torch.float64, device=DEVICE)
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
    x = nan_padded(T, C + 11, torch.randn(T, C, device=DEVICE))
    w = (torch.randn(C, W, device=DEVICE) * 0.5).to(torch.bfloat16)
    bias = (torch.randn(C, device=DEVICE) * 0.1).to(torch.bfloat16)
    state = torch.randn(max(slots) + 1, W - 1, C, device=DEVICE)
    out = torch.full((T, C), NAN, device=DEVICE)
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
    x = torch.randn(L + K, C, device=DEVICE)
    w = (torch.randn(C, W, device=DEVICE) * 0.5).to(torch.bfloat16)
    bias = (torch.randn(C, device=DEVICE) * 0.1).to(torch.bfloat16)

    whole_state = torch.zeros(1, W - 1, C, device=DEVICE)
    whole = torch.empty(L + K, C, device=DEVICE)
    lx("conv1d_varlen", x, C, w, bias, whole_state, i32([0]), i32([0]), cumulative([L + K]), 1, whole, C, W)

    step_state = torch.zeros(2, W - 1, C, device=DEVICE)
    parts = []
    prefill = torch.empty(L, C, device=DEVICE)
    lx("conv1d_varlen", x[:L], C, w, bias, step_state, i32([1]), i32([0]), cumulative([L]), 1, prefill, C, W)
    parts.append(prefill)
    for i in range(K):
        step = torch.empty(1, C, device=DEVICE)
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
    y = torch.zeros(x.shape[0], H, P, dtype=torch.float64, device=DEVICE)
    negA = -torch.exp(A_log)
    rep = H // G
    start = 0
    for s, (slot, n) in enumerate(zip(slots, lengths)):
        S = state[slot].clone() if has[s] else torch.zeros(H, P, N, dtype=torch.float64, device=DEVICE)
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
    x = nan_padded(T, ld_x, torch.randn(T, H * P, device=DEVICE) * 0.5)
    dt_raw = nan_padded(T, ld_dt, torch.randn(T, H, device=DEVICE))
    B = nan_padded(T, ld_bc, torch.randn(T, G * N, device=DEVICE) * 0.5)
    C = nan_padded(T, ld_bc, torch.randn(T, G * N, device=DEVICE) * 0.5)
    A_log = torch.log(torch.rand(H, device=DEVICE) * 3.5 + 0.5)
    D = torch.randn(H, device=DEVICE)
    dt_bias = torch.randn(H, device=DEVICE) * 0.5
    state = torch.randn(n_slots, H, P, N, device=DEVICE) * 0.5  # junk for has_state == 0
    return dict(x=x, ld_x=ld_x, dt_raw=dt_raw, ld_dt=ld_dt, B=B, C=C, ld_bc=ld_bc,
                A_log=A_log, D=D, dt_bias=dt_bias, state=state)


def run_scan(lx, inp, slots, has, lengths, H, P, G, N, state):
    T = sum(lengths)
    y = torch.full((T, H, P), NAN, device=DEVICE)
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
    whole_state = torch.zeros(2, H, P, N, device=DEVICE)
    whole = run_scan(lx, inp, [1], [0], [L + K], H, P, G, N, whole_state)

    step_state = torch.zeros(2, H, P, N, device=DEVICE)
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
    y = torch.randn(T, I, device=DEVICE)
    z = nan_padded(T, I + 5, torch.randn(T, I, device=DEVICE))
    w = (torch.rand(I, device=DEVICE) + 0.5).to(torch.bfloat16)
    out = torch.full((T, I), NAN, device=DEVICE, dtype=torch.bfloat16)
    lx("gated_rmsnorm", y, z, z.shape[1], w, out, T, I, groups, eps)
    want = gated_norm_reference(y, z[:, :I], w, groups, eps)
    close(out, want, atol=1e-6, rtol=1e-2)


# ---- Fused decode step (lx_mamba_decode_step) -----------------------------------------------
# The fused step is the conv, the scan and the gated norm run in turn, one token per sequence, bit for
# bit. The norm's sums are over K = I / G channels on 256 threads in both, so the claim is exact where
# K >= 256 (every Metis model); smaller K is checked to a tolerance, with the states still exact.


def same_bits(actual, expected, what):
    """Bit-for-bit equal, NaNs included; names the first element that differs."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, what
    view = torch.int32 if actual.dtype == torch.float32 else torch.int16
    diff = (actual.contiguous().view(view) != expected.contiguous().view(view)).nonzero()
    assert diff.numel() == 0, (
        f"{what}: {diff.shape[0]} elements differ, first at {diff[0].tolist()}: "
        f"got {actual[tuple(diff[0].tolist())].item()}, want {expected[tuple(diff[0].tolist())].item()}"
    )


def decode_weights(H, P, G, N, W):
    I, GN = H * P, G * N
    conv_dim = I + 2 * GN
    return dict(
        H=H, P=P, G=G, N=N, W=W, I=I, GN=GN, conv_dim=conv_dim, proj=2 * I + 2 * GN + H,
        K=I // G,
        conv_w=(torch.randn(conv_dim, W, device=DEVICE) * 0.5).to(torch.bfloat16),
        conv_b=(torch.randn(conv_dim, device=DEVICE) * 0.1).to(torch.bfloat16),
        A_log=torch.log(torch.rand(H, device=DEVICE) * 3.5 + 0.5),
        D=torch.randn(H, device=DEVICE),
        dt_bias=torch.randn(H, device=DEVICE) * 0.5,
        norm_w=(torch.rand(I, device=DEVICE) + 0.5).to(torch.bfloat16),
    )


def decode_rows(w, n, pad=3):
    """in_proj's output for n sequences, one row each, with NaN past its last column (never read)."""
    return nan_padded(n, w["proj"] + pad, torch.randn(n, w["proj"], device=DEVICE))


def at(t, elems):
    """A pointer to element `elems` of contiguous tensor t, for the kernels' offset inputs."""
    return ctypes.c_void_p(t.data_ptr() + elems * t.element_size())


def split_step(lx, w, zb, slots, has, conv_state, ssm_state, eps=1e-5):
    """The three kernels in turn: lx_conv1d_varlen, lx_ssm_scan_varlen, lx_gated_rmsnorm (as the model ran them)."""
    n, ld = zb.shape[0], zb.shape[1]
    I, GN, conv_dim, H, P, G, N = w["I"], w["GN"], w["conv_dim"], w["H"], w["P"], w["G"], w["N"]
    xbc = torch.empty(n, conv_dim, device=DEVICE)
    y = torch.empty(n, I, device=DEVICE)
    out = torch.empty(n, I, device=DEVICE, dtype=torch.bfloat16)
    cu = cumulative([1] * n)
    lx("conv1d_varlen", at(zb, I), ld, w["conv_w"], w["conv_b"], conv_state, slots, has, cu, n, xbc, conv_dim, w["W"])
    lx("ssm_scan_varlen", xbc, conv_dim, at(zb, I + conv_dim), ld, at(xbc, I), at(xbc, I + GN), conv_dim,
       w["A_log"], w["D"], w["dt_bias"], ssm_state, slots, has, cu, n, y, H, P, G, N)
    lx("gated_rmsnorm", y, zb, ld, w["norm_w"], out, n, I, G, eps)
    return y, out


def fused_step(lx, w, zb, slots, has, conv_state, ssm_state, sync, eps=1e-5):
    """lx_mamba_decode_step; sync is the group counters, which the kernel must leave zero."""
    if not hasattr(lx.lib, "lx_mamba_decode_step"):
        pytest.skip("this backend does not export lx_mamba_decode_step (the model runs the three kernels)")
    n, ld = zb.shape[0], zb.shape[1]
    I, H, P, G, N, W = w["I"], w["H"], w["P"], w["G"], w["N"], w["W"]
    y = torch.full((n, I), NAN, device=DEVICE)  # NaN where nothing is written
    out = torch.full((n, I), NAN, device=DEVICE, dtype=torch.bfloat16)
    lx("mamba_decode_step", zb, ld, w["conv_w"], w["conv_b"], conv_state, ssm_state, w["A_log"], w["D"],
       w["dt_bias"], slots, has, w["norm_w"], eps, y, out, sync, n, H, P, G, N, W)
    assert int(sync.abs().sum()) == 0, "the group counters must be zero after a step"
    return y, out


DECODE_CASES = [
    # H, P, G, N, W, n_seq: Praxis, Praxis at one stream, Logos's head count, widest state, W = 2 and 1, N not a
    # multiple of 16 (partial chunks), and odd sizes whose K = I / G < 256 (checked to a tolerance)
    (64, 64, 8, 128, 4, 1),
    (64, 64, 8, 128, 4, 2),
    (64, 64, 8, 128, 4, 8),
    (64, 64, 8, 128, 4, 33),
    (64, 64, 8, 128, 4, 64),
    (80, 64, 10, 128, 4, 5),  # Logos: 80 heads in 10 groups
    (8, 64, 2, 256, 4, 3),
    (64, 64, 8, 128, 2, 4),
    (64, 64, 8, 128, 1, 3),
    (16, 64, 4, 40, 3, 2),
    (6, 5, 3, 22, 4, 3),
]


@pytest.mark.parametrize("H,P,G,N,W,n", DECODE_CASES)
def test_mamba_decode_step_matches_three_kernels(lx, H, P, G, N, W, n):
    torch.manual_seed(7)
    w = decode_weights(H, P, G, N, W)
    exact = w["K"] >= 256
    slots = torch.randperm(2 * n)[:n].to(torch.int32).to(DEVICE)
    # W = 1 keeps no conv rows; one row keeps the buffer non-null (the CPU backend checks its pointers)
    conv_split = torch.randn(2 * n, max(W - 1, 1), w["conv_dim"], device=DEVICE)
    ssm_split = torch.randn(2 * n, H, P, N, device=DEVICE) * 0.5
    conv_fused, ssm_fused = conv_split.clone(), ssm_split.clone()
    sync = torch.zeros(n * G, dtype=torch.int32, device=DEVICE)
    for step in range(3):  # the states carry from step to step, and the counters are reused
        zb = decode_rows(w, n)
        has = (torch.randint(0, 2, (n,)) if step == 0 else torch.ones(n, dtype=torch.long)).to(torch.int32)
        has = has.to(DEVICE)
        y_s, out_s = split_step(lx, w, zb, slots, has, conv_split, ssm_split)
        y_f, out_f = fused_step(lx, w, zb, slots, has, conv_fused, ssm_fused, sync)
        same_bits(y_f, y_s, f"step {step}: scan output")
        same_bits(conv_fused, conv_split, f"step {step}: conv state")
        same_bits(ssm_fused, ssm_split, f"step {step}: ssm state")
        if exact:
            same_bits(out_f, out_s, f"step {step}: gated norm")
        else:
            close(out_f, out_s, atol=1e-2)


@pytest.mark.parametrize("H,P,G,N,W,n", [(64, 64, 8, 128, 4, 5), (80, 64, 10, 128, 4, 3)])
def test_mamba_decode_step_reference(lx, H, P, G, N, W, n):
    """Against float64 PyTorch: the conv, the scan and the gate, one step from carried states."""
    torch.manual_seed(11)
    w = decode_weights(H, P, G, N, W)
    I, GN, conv_dim, proj = w["I"], w["GN"], w["conv_dim"], w["proj"]
    slots = torch.randperm(2 * n)[:n].to(torch.int32).to(DEVICE)
    has = torch.randint(0, 2, (n,)).to(torch.int32).to(DEVICE)
    conv0 = torch.randn(2 * n, W - 1, conv_dim, device=DEVICE)
    ssm0 = torch.randn(2 * n, H, P, N, device=DEVICE) * 0.5
    zb = decode_rows(w, n)
    lengths = [1] * n
    slot_list, has_list = slots.tolist(), has.tolist()

    conv_out, want_conv = conv_reference(zb[:, I:I + conv_dim].contiguous(), w["conv_w"], w["conv_b"], conv0,
                                         slot_list, has_list, lengths, W)
    y_want, want_ssm = scan_reference(
        conv_out[:, :I], zb[:, I + conv_dim:I + conv_dim + H].contiguous(), conv_out[:, I:I + GN],
        conv_out[:, I + GN:], w["A_log"], w["D"], w["dt_bias"], ssm0, slot_list, has_list, lengths, H, P, G, N,
    )
    out_want = gated_norm_reference(y_want.reshape(n, I), zb[:, :I], w["norm_w"], G, 1e-5)

    conv, ssm = conv0.clone(), ssm0.clone()
    sync = torch.zeros(n * G, dtype=torch.int32, device=DEVICE)
    y, out = fused_step(lx, w, zb, slots, has, conv, ssm, sync)
    close(y, y_want.reshape(n, I), atol=1e-4 * y_want.abs().max().item())
    close(out, out_want, atol=1e-6, rtol=1e-2)
    close(conv, want_conv.float(), atol=1e-6)
    close(ssm, want_ssm.float(), atol=1e-4 * want_ssm.abs().max().item())


def test_mamba_decode_step_batch_invariant(lx):
    """A sequence's bits do not depend on the other sequences in the step: alone or among 63 others."""
    torch.manual_seed(13)
    H, P, G, N, W, n = 64, 64, 8, 128, 4, 64
    w = decode_weights(H, P, G, N, W)
    slots = torch.arange(n, dtype=torch.int32, device=DEVICE)
    has = torch.randint(0, 2, (n,)).to(torch.int32).to(DEVICE)
    conv0 = torch.randn(n, W - 1, w["conv_dim"], device=DEVICE)
    ssm0 = torch.randn(n, H, P, N, device=DEVICE) * 0.5
    zb = decode_rows(w, n)

    conv_b, ssm_b = conv0.clone(), ssm0.clone()
    y_b, out_b = fused_step(lx, w, zb, slots, has, conv_b, ssm_b, torch.zeros(n * G, dtype=torch.int32, device=DEVICE))
    for k in (0, 17, 63):
        conv_a, ssm_a = conv0.clone(), ssm0.clone()
        row = zb[k:k + 1].contiguous()
        y_a, out_a = fused_step(lx, w, row, slots[k:k + 1], has[k:k + 1], conv_a, ssm_a,
                                torch.zeros(G, dtype=torch.int32, device=DEVICE))
        same_bits(y_a[0], y_b[k], f"sequence {k}: scan output alone vs in the batch")
        same_bits(out_a[0], out_b[k], f"sequence {k}: gated norm alone vs in the batch")
        same_bits(ssm_a[k], ssm_b[k], f"sequence {k}: ssm state")
        same_bits(conv_a[k], conv_b[k], f"sequence {k}: conv state")
