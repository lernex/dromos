"""The model's fused row ops (kernels.h, "Model glue"): each writes exactly the bits of the copies it replaces."""

import pytest
import torch

from conftest import DEVICE, DTYPE, close


@pytest.mark.parametrize("T,D,R", [(1, 2048, 512), (9, 2048, 512), (300, 64, 17), (3, 33, 5)])
def test_concat4_bf16_is_the_four_copies(lx, T, D, R):
    """[a | b | c | d] as one bf16 row per token: the routing features' GEMM input."""
    torch.manual_seed(0)
    a, b, c = (torch.randn(T, D, device=DEVICE) for _ in range(3))
    d = torch.randn(T, R, device=DEVICE)
    out = torch.zeros(T, 3 * D + R, device=DEVICE).to(torch.bfloat16)

    lx("concat4_bf16", a, b, c, d, out, D, R, T)

    close(out, torch.cat([a, b, c, d], dim=1).to(torch.bfloat16), atol=0.0)


@pytest.mark.parametrize("T,Lat,R", [(1, 384, 96), (5, 64, 8), (2, 17, 3)])
def test_pack_router_is_the_latent_cast_and_the_feature_copy(lx, T, Lat, R):
    """The latent's bf16 copy, and the features written into the router input's tail."""
    torch.manual_seed(1)
    router_in = torch.randn(T, Lat + R, device=DEVICE)
    features = torch.randn(T, R, device=DEVICE)
    latent = torch.zeros(T, Lat, device=DEVICE).to(torch.bfloat16)
    want_latent = router_in[:, :Lat].to(torch.bfloat16)
    want_router = torch.cat([router_in[:, :Lat], features], dim=1)

    lx("pack_router", router_in, latent, features, Lat, R, T)

    close(latent, want_latent, atol=0.0)
    close(router_in, want_router, atol=0.0)


@pytest.mark.parametrize("T,R", [(1, 512), (7, 33), (4, 2)])
def test_meta_entry_is_the_summary_copy_and_the_scalars(lx, T, R):
    """Bit-identical to the copy2d and meta_scalars sequence it replaces in memory_write."""
    torch.manual_seed(2)
    summary = torch.randn(T, R, device=DEVICE)
    entropy, confidence, continuation = (torch.rand(T, device=DEVICE) for _ in range(3))
    mean_k, inv_log_e = 0.37, 1.0 / 3.2
    fused = torch.zeros(T, R + 4, device=DEVICE).to(torch.bfloat16)
    old = torch.zeros(T, R + 4, device=DEVICE).to(torch.bfloat16)

    lx("meta_entry", summary, entropy, confidence, continuation, mean_k, inv_log_e, fused, R, T)

    lx("copy2d", summary, DTYPE[torch.float32], R, old, DTYPE[torch.bfloat16], R + 4, T, R)
    lx("meta_scalars", entropy, confidence, continuation, mean_k, inv_log_e, old, R + 4, R, T)
    close(fused, old, atol=0.0)


@pytest.mark.parametrize("rows,cols", [(1, 512), (3, 37), (300, 64)])
def test_add_two_rows_is_two_add_rows(lx, rows, cols):
    """y += row_a, then y += row_b, in one launch: the bits of the two add_row calls memory_write used to make."""
    torch.manual_seed(3)
    y0 = torch.randn(rows, cols, device=DEVICE)
    a, b = (torch.randn(cols, device=DEVICE).to(torch.bfloat16) for _ in range(2))
    fused, old = y0.clone(), y0.clone()

    lx("add_two_rows_bf16", fused, a, b, rows, cols)

    lx("add_row_bf16", old, a, rows, cols)
    lx("add_row_bf16", old, b, rows, cols)
    close(fused, old, atol=0.0)


@pytest.mark.parametrize("T,D,R", [(1, 2048, 512), (9, 64, 17), (3, 33, 5)])
def test_concat4_f32_is_the_four_copies(lx, T, D, R):
    """[a | b | c | d] as one f32 row per token: the continuation head's input, the copy2d calls it used to make."""
    torch.manual_seed(4)
    a, b, c = (torch.randn(T, D, device=DEVICE) for _ in range(3))
    d = torch.randn(T, R, device=DEVICE)
    out = torch.zeros(T, 3 * D + R, device=DEVICE)

    lx("concat4_f32", a, b, c, d, out, D, R, T)

    close(out, torch.cat([a, b, c, d], dim=1), atol=0.0)


@pytest.mark.parametrize("T,S,D,R", [(1, 4, 2048, 512), (9, 4, 64, 17), (3, 2, 33, 5)])
def test_stream_concat4_is_the_stream_mean_then_concat(lx, T, S, D, R):
    """The stream mean as the first segment of the routing row, in one launch: the mean is written as lx_stream_mean
    writes it, and the row is the bf16 concat of that mean with b, c and d."""
    torch.manual_seed(5)
    streams = torch.randn(T, S, D, device=DEVICE)
    b, c = (torch.randn(T, D, device=DEVICE) for _ in range(2))
    d = torch.randn(T, R, device=DEVICE)
    mean, out = torch.zeros(T, D, device=DEVICE), torch.zeros(T, 3 * D + R, device=DEVICE).to(torch.bfloat16)
    ref_mean, ref_out = torch.zeros(T, D, device=DEVICE), torch.zeros(T, 3 * D + R, device=DEVICE).to(torch.bfloat16)

    lx("stream_concat4_bf16", streams, mean, S, b, c, d, out, D, R, T)

    lx("stream_mean", streams, ref_mean, T, S, D)
    lx("concat4_bf16", ref_mean, b, c, d, ref_out, D, R, T)
    close(mean, ref_mean, atol=0.0)
    close(out, ref_out, atol=0.0)
