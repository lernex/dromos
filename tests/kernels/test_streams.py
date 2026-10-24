import pytest
import torch

from conftest import close


@pytest.mark.parametrize(
    "T,S,D",
    [(1, 4, 2048), (300, 4, 2048), (5, 3, 37), (2, 4, 2560)],
)
def test_embed_streams(lx, T, S, D):
    torch.manual_seed(0)
    V = 1000
    table = torch.randn(V, D, device="cuda").to(torch.bfloat16)
    stream_emb = (torch.randn(S, D, device="cuda") * 0.1).to(torch.bfloat16)
    ids = torch.randint(0, V, (T,), device="cuda", dtype=torch.int32)
    streams = torch.empty(T, S, D, device="cuda")

    lx("embed_streams", table, ids, stream_emb, streams, T, S, D)

    want = table.double()[ids.long()][:, None, :] + stream_emb.double()[None]
    close(streams, want, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("T,S,D", [(1, 4, 2048), (300, 4, 2048), (3, 3, 37), (70000, 4, 37)])
def test_stream_mean(lx, T, S, D):
    torch.manual_seed(0)
    streams = torch.randn(T, S, D, device="cuda")
    mean = torch.empty(T, D, device="cuda")

    lx("stream_mean", streams, mean, T, S, D)

    close(mean, streams.double().mean(1), atol=1e-6, rtol=1e-5)


def mhc_read_reference(streams, matrix, read, norm_w, eps):
    s = streams.double()
    mixed = torch.einsum("oi,tid->tod", matrix.double(), s)
    source = torch.einsum("i,tid->td", read.double(), s)
    h = source * torch.rsqrt(source.pow(2).mean(-1, keepdim=True) + eps) * norm_w.double()
    return mixed, h


@pytest.mark.parametrize(
    "T,S,D",
    [(1, 4, 2048), (300, 4, 2048), (3, 3, 37), (2, 1, 5), (2, 4, 2560)],
)
def test_mhc_read(lx, T, S, D):
    torch.manual_seed(0)
    eps = 1e-6
    streams = torch.randn(T, S, D, device="cuda")
    matrix = torch.randn(S, S, device="cuda")
    read = torch.randn(S, device="cuda")
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    mixed = torch.empty(T, S, D, device="cuda")
    h = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)

    lx("mhc_read", streams, matrix, read, norm_w, eps, mixed, h, T, S, D)

    want_mixed, want_h = mhc_read_reference(streams, matrix, read, norm_w, eps)
    close(mixed, want_mixed, atol=1e-6, rtol=1e-5)
    # h is bf16: one rounding step is 2^-8 relative, so rtol 1e-2 with a small absolute floor.
    close(h, want_h, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("T,S,D", [(1, 4, 2048), (300, 4, 2048), (3, 3, 37)])
@pytest.mark.parametrize("alias", [False, True])
def test_mhc_write(lx, T, S, D, alias):
    torch.manual_seed(0)
    mixed = torch.randn(T, S, D, device="cuda")
    write = torch.randn(S, device="cuda")
    update = torch.randn(T, D, device="cuda")
    want = mixed.double() + write.double()[None, :, None] * update.double()[:, None, :]

    # alias: the spec allows streams to be the mixed buffer itself, updated in place.
    streams = mixed if alias else torch.empty_like(mixed)
    lx("mhc_write", mixed, write, update, streams, T, S, D)

    close(streams, want, atol=1e-6, rtol=1e-5)
