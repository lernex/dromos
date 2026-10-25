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
