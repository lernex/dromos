import math

import pytest
import torch

from conftest import DEVICE, close


@pytest.mark.parametrize(
    "T,S,M,E,n",
    [
        (1, 4, 256, 8, 1),  # one entry: the softmax is 1, the output is its value
        (1, 4, 256, 8, 8),
        (300, 4, 256, 8, 3),
        (300, 4, 256, 8, 8),
        (5, 3, 100, 5, 2),  # odd widths, bank larger than the valid prefix
        (2, 1, 33, 8, 5),
        # Decode path (T <= 8, M % 4 == 0, M <= 256): vector loads, one warp per row.
        (8, 4, 256, 8, 8),
        (4, 4, 256, 8, 1),
        (8, 4, 256, 8, 5),
        (1, 8, 256, 8, 8),
        (7, 2, 64, 8, 8),
        (4, 4, 256, 3, 3),  # bank of 3 entries: the kernel must not read past it
        (2, 4, 512, 8, 8),  # wider than the decode path's vector cap: one warp per row, scalar
    ],
)
def test_memory_attend(lx, T, S, M, E, n):
    torch.manual_seed(0)
    q = torch.randn(T, S, M, device=DEVICE)
    bank_k = torch.randn(T, E, M, device=DEVICE)
    bank_v = torch.randn(T, E, M, device=DEVICE)
    # Entries past n are never valid, so they hold NaN: any read of them would show in the output.
    bank_k[:, n:] = float("nan")
    bank_v[:, n:] = float("nan")
    out = torch.empty(T, S, M, device=DEVICE, dtype=torch.bfloat16)

    lx("memory_attend", q, bank_k, bank_v, n, E, out, T, S, M)

    k = bank_k[:, :n].double()
    v = bank_v[:, :n].double()
    scores = torch.einsum("tsm,tem->tse", q.double(), k) / math.sqrt(M)
    want = torch.einsum("tse,tem->tsm", torch.softmax(scores, dim=-1), v)
    close(out, want, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(
    "T,S,D",
    [
        (1, 4, 2048),
        (300, 4, 2048),
        (3, 3, 37),
        (2, 4, 2560),
        # Decode path: a cluster of 8 blocks per token, one column per thread (rows up to 8 * 512).
        (8, 4, 2048),
        (4, 4, 2048),
        (1, 8, 2048),
        (5, 8, 300),
        (8, 3, 37),
        (2, 4, 4096),
        (1, 4, 8192),  # past the decode path: one block per token
        (1, 4, 9000),
    ],
)
def test_memory_fuse(lx, T, S, D):
    torch.manual_seed(0)
    streams = torch.randn(T, S, D, device=DEVICE)
    projected = torch.randn(T, S, D, device=DEVICE) * 0.5
    gate = torch.randn(S, D, device=DEVICE).to(torch.bfloat16)
    gate_bias = torch.randn(S, device=DEVICE).to(torch.bfloat16)
    summary = torch.empty(T, D, device=DEVICE)

    # The gate reads the streams before the projected memory is added, so the reference gates
    # the original values. The kernel updates streams in place, so take the reference first.
    s = streams.double()
    logit = torch.einsum("tsd,sd->ts", s, gate.double()) / math.sqrt(D) + gate_bias.double()
    want_streams = s + torch.sigmoid(logit)[..., None] * projected.double()
    want_summary = projected.double().mean(1)

    lx("memory_fuse", streams, projected, gate, gate_bias, summary, T, S, D)

    close(streams, want_streams, atol=1e-5, rtol=1e-5)
    close(summary, want_summary, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("T,D", [(1, 2048), (300, 2048), (4, 37)])
@pytest.mark.parametrize("limit", [6.0, 0.5])
def test_continuation_finish(lx, T, D, limit):
    torch.manual_seed(0)
    logit = torch.randn(T, device=DEVICE) * 3.0
    state = torch.randn(T, D, device=DEVICE)
    difference = torch.randn(T, D, device=DEVICE) * 0.3
    if T > 1:
        state[0] = 0.0  # an all-zero state takes the 1e-6 floor on the scale
    out = torch.empty(T, device=DEVICE)

    lx("continuation_finish", logit, state, difference, limit, out, T, D)

    scale = torch.clamp(state.double().pow(2).mean(-1).sqrt(), min=1e-6)
    innovation = difference.double().pow(2).mean(-1).sqrt()
    z = logit.double() + torch.tanh(innovation / scale) - 0.5
    want = torch.sigmoid(torch.clamp(z, -limit, limit))
    close(out, want, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("T,S,M,E,n", [(8, 4, 256, 8, 8), (1, 4, 256, 8, 3)])
def test_decode_attend_is_repeatable(lx, T, S, M, E, n):
    torch.manual_seed(0)
    q = torch.randn(T, S, M, device=DEVICE)
    bank_k = torch.randn(T, E, M, device=DEVICE)
    bank_v = torch.randn(T, E, M, device=DEVICE)
    first = None
    for _ in range(20):
        out = torch.empty(T, S, M, device=DEVICE, dtype=torch.bfloat16)
        lx("memory_attend", q, bank_k, bank_v, n, E, out, T, S, M)
        if first is None:
            first = out
        else:
            assert torch.equal(out, first), "memory_attend differs between runs"


@pytest.mark.parametrize("T,S,D", [(8, 4, 2048), (1, 8, 2560)])
def test_decode_fuse_is_repeatable(lx, T, S, D):
    torch.manual_seed(0)
    streams0 = torch.randn(T, S, D, device=DEVICE)
    projected = torch.randn(T, S, D, device=DEVICE) * 0.5
    gate = torch.randn(S, D, device=DEVICE).to(torch.bfloat16)
    gate_bias = torch.randn(S, device=DEVICE).to(torch.bfloat16)
    first = None
    for _ in range(20):
        streams = streams0.clone()  # the kernel updates streams in place
        summary = torch.empty(T, D, device=DEVICE)
        lx("memory_fuse", streams, projected, gate, gate_bias, summary, T, S, D)
        if first is None:
            first = (streams, summary)
        else:
            assert torch.equal(streams, first[0]), "memory_fuse streams differ between runs"
            assert torch.equal(summary, first[1]), "memory_fuse summary differs between runs"
