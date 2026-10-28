"""A row's bits do not depend on the step it runs in.

Each dense GEMM pins the algorithm it runs per shape, not per row count: cuBLASLt's (gemm.cu, pinned_algo) and lx_gemm_w's
variant and split (gemm_w.cu, run_family). A prefill chunk's rows must come out the same whatever else is in the step, or
a request's state (and so its tokens) depends on what other requests are doing when it arrives. Both the BF16 path (cuBLASLt)
and the FP8 path (lx_gemm_w, the engine's FP8 products) are checked, at tails of one and nine rows as well.
"""

import pytest
import torch

from conftest import DEVICE, DTYPE
from test_gemm_w import call_gemm_w, workspace_bytes
from wformats import MATH_BF16, MATH_FP8, W_FP8, quantize_weight

SHAPES = [(2048, 10304), (4096, 2048)]  # (K, N): Praxis's in_proj and out_proj
ROWS = [  # (rows alone, rows in the larger step): tails of one and nine rows included
    (509, 1908), (509, 1181), (512, 884), (300, 765), (45, 48), (9, 512), (1, 509),
]


@pytest.mark.parametrize("K,N", SHAPES)
@pytest.mark.parametrize("small,big", ROWS)
def test_gemm_bf16_rows_do_not_depend_on_the_step(lx, K, N, small, big):
    torch.manual_seed(small * 31 + big)
    x = torch.randn(big, K, device=DEVICE).to(torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE).to(torch.bfloat16)
    y_big = torch.empty(big, N, device=DEVICE, dtype=torch.bfloat16)
    y_small = torch.empty(small, N, device=DEVICE, dtype=torch.bfloat16)
    x_small = x[:small].contiguous()
    lx("gemm_bf16", x, K, w, None, y_big, DTYPE[torch.bfloat16], N, big, N, K, 1.0, 0.0)
    lx("gemm_bf16", x_small, K, w, None, y_small, DTYPE[torch.bfloat16], N, small, N, K, 1.0, 0.0)
    a, b = y_small.float(), y_big[:small].float()
    differing = int((a != b).any(dim=1).sum().item())
    assert differing == 0, (
        f"{differing} of {small} rows differ from the same rows in a {big}-row step "
        f"(max |diff| {(a - b).abs().max().item():.3g})"
    )


# The engine's FP8 products run through lx_gemm_w: fp8 weights with fp8 activations (the Logos FP8 path), and with bf16
# activations. The same shape-pinned rule applies to its variant and split.
FP8_PAIRS = [("fp8", W_FP8, MATH_FP8), ("w8a16_fp8", W_FP8, MATH_BF16)]


@pytest.mark.parametrize("name,wtype,math", FP8_PAIRS)
@pytest.mark.parametrize("K,N", SHAPES)
@pytest.mark.parametrize("small,big", ROWS)
def test_gemm_w_fp8_rows_do_not_depend_on_the_step(lx, name, wtype, math, K, N, small, big):
    torch.manual_seed(small * 31 + big + wtype * 7 + math * 11)
    x = torch.randn(big, K, device=DEVICE).to(torch.bfloat16)
    w = quantize_weight(torch.randn(N, K, device=DEVICE) / K**0.5, wtype)
    y_big = torch.empty(big, N, device=DEVICE, dtype=torch.bfloat16)
    y_small = torch.empty(small, N, device=DEVICE, dtype=torch.bfloat16)
    ws_big = torch.zeros(max(workspace_bytes(lx, big, K, math), 1), dtype=torch.uint8, device=DEVICE)
    ws_small = torch.zeros(max(workspace_bytes(lx, small, K, math), 1), dtype=torch.uint8, device=DEVICE)
    status = call_gemm_w(lx, x, K, w, math, None, y_big, N, big, N, K, 1.0, 0.0, ws_big)
    assert status == 0, f"lx_gemm_w returned CUDA error {status} at {big} rows"
    status = call_gemm_w(lx, x[:small].contiguous(), K, w, math, None, y_small, N, small, N, K, 1.0, 0.0, ws_small)
    assert status == 0, f"lx_gemm_w returned CUDA error {status} at {small} rows"
    a, b = y_small.float(), y_big[:small].float()
    differing = int((a != b).any(dim=1).sum().item())
    assert differing == 0, (
        f"{name}: {differing} of {small} rows differ from the same rows in a {big}-row step "
        f"(max |diff| {(a - b).abs().max().item():.3g})"
    )
