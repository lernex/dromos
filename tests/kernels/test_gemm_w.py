"""lx_gemm_w (engine/cuda/gemm_w.cu) against PyTorch, for every weight format x math pair and every
kernel family: each case runs on the card's own family and again with lx_set_arch_override(89) and
(80), which run the sm_89 and sm_80 families on this GPU; a pair a family cannot run must return
cudaErrorNotSupported.

The reference multiplies exactly what the kernel multiplies, in float64: the weight dequantized from its
stored format, and the activations bf16-rounded (BF16 math) or quantized per token with the reference
quantizer (whose output test_quant_act checks bit for bit against the kernel's). So the only difference
left is float32 accumulation order, and the tolerance is tight."""

import ctypes

import pytest
import torch

from conftest import DTYPE, close
from wformats import (
    MATH_BF16,
    MATH_FP4,
    MATH_FP8,
    MATH_INT8,
    W_BF16,
    W_FP8,
    W_INT8,
    W_NVFP4,
    act_operand,
    quantize_weight,
)

NOT_SUPPORTED = 801

# (name, weight format, math, oldest family that runs it)
PAIRS = [
    ("bf16", W_BF16, MATH_BF16, 80),
    ("w8a16_int8", W_INT8, MATH_BF16, 80),
    ("w8a16_fp8", W_FP8, MATH_BF16, 80),
    ("w4a16", W_NVFP4, MATH_BF16, 80),
    ("int8", W_INT8, MATH_INT8, 80),
    ("fp8", W_FP8, MATH_FP8, 89),
    ("w4a8", W_NVFP4, MATH_FP8, 120),
    ("fp4", W_NVFP4, MATH_FP4, 120),
]
ARCHS = [0, 89, 80]  # 0: the card's own
SHAPES = [
    (1, 64, 256),
    (7, 136, 512),  # N not a multiple of the 128-wide tile
    (64, 1024, 2048),
    (300, 3072, 1024),
    (777, 520, 2048),  # M and N tails on the largest tile
]


def workspace_bytes(lx, M, K, math):
    fn = lx.lib.lx_gemm_workspace_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return fn(M, K, math)


def set_arch(lx, arch):
    fn = lx.lib.lx_set_arch_override
    fn.restype = ctypes.c_int
    return fn(arch)


def native_arch(lx):
    set_arch(lx, 0)
    return lx.lib.lx_arch()


def call_gemm_w(lx, x, ld_x, w, math, bias, y, ld_y, M, N, K, alpha, beta, ws):
    fn = lx.lib.lx_gemm_w
    fn.restype = ctypes.c_int
    ptr = lambda t: ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(None)
    status = fn(
        ptr(x), DTYPE[x.dtype], ld_x, ctypes.byref(w.struct), math, ptr(bias),
        DTYPE[bias.dtype] if bias is not None else 0, ptr(y), DTYPE[y.dtype], ld_y, M, N, K,
        ctypes.c_float(alpha), ctypes.c_float(beta), ptr(ws), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()
    return status


def reference(x, w, math, bias, alpha, beta, y0):
    out = act_operand(x, math) @ w.ref().T
    if bias is not None:
        out = out + bias.double()
    out = alpha * out
    if beta != 0.0:
        out = out + beta * y0.double()
    return out


def run_case(lx, name, wtype, math, M, N, K, x_dtype, y_dtype, alpha, beta, with_bias, pad):
    torch.manual_seed(M * 7 + N * 13 + K + wtype * 101 + math * 1009)
    x_buf = torch.randn(M, K + pad, device="cuda").to(x_dtype)
    x = x_buf[:, :K]
    w = quantize_weight(torch.randn(N, K, device="cuda") / K**0.5, wtype)
    bias = (torch.randn(N, device="cuda") * 0.1).to(torch.bfloat16 if with_bias == "bf16" else torch.float32) if with_bias else None
    y0 = torch.randn(M, N + pad, device="cuda").to(y_dtype)
    y = y0.clone()
    ws = torch.full((max(workspace_bytes(lx, M, K, math), 1),), 0xFF, dtype=torch.uint8, device="cuda")
    status = call_gemm_w(lx, x_buf, K + pad, w, math, bias, y, N + pad, M, N, K, alpha, beta, ws)
    return status, y, y0, (x, w, bias)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_gemm_w(lx, name, wtype, math, oldest, M, N, K, arch):
    have = native_arch(lx)
    family = arch if arch else have
    try:
        set_arch(lx, arch)
        status, y, y0, (x, w, bias) = run_case(lx, name, wtype, math, M, N, K, torch.bfloat16, torch.float32, 1.0, 0.0, None, 0)
    finally:
        set_arch(lx, 0)
    if family < oldest:
        assert status == NOT_SUPPORTED, f"{name} on sm_{family} returned {status}, expected NotSupported"
        return
    assert status == 0, f"lx_gemm_w returned CUDA error {status}"
    want = reference(x, w, math, None, 1.0, 0.0, None)
    close(y, want.float(), atol=2e-4 * want.abs().max().item())


@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
@pytest.mark.parametrize(
    "x_dtype,y_dtype,alpha,beta,with_bias,pad",
    [
        (torch.float32, torch.float32, 0.5, 1.0, "bf16", 8),
        (torch.bfloat16, torch.bfloat16, 1.0, 1.0, "f32", 8),
        (torch.float32, torch.bfloat16, 2.0, 0.0, None, 0),
        (torch.bfloat16, torch.float32, 1.0, 0.5, "bf16", 3),
    ],
)
def test_gemm_w_epilogue(lx, name, wtype, math, oldest, x_dtype, y_dtype, alpha, beta, with_bias, pad):
    # f32 activations, bias in either dtype, alpha and beta, bf16 output and padded (odd) row strides.
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    M, N, K = 70, 200, 512
    status, y, y0, (x, w, bias) = run_case(lx, name, wtype, math, M, N, K, x_dtype, y_dtype, alpha, beta, with_bias, pad)
    assert status == 0, f"lx_gemm_w returned CUDA error {status}"
    want = reference(x, w, math, bias, alpha, beta, y0[:, :N])
    tol = (1e-2 if y_dtype == torch.bfloat16 else 2e-4) * want.abs().max().item()
    close(y[:, :N], want.float(), atol=tol)
    if pad:
        assert torch.equal(y[:, N:].float(), y0[:, N:].float()), "wrote past the N columns of y"


def test_gemm_w_tuned_variants(lx):
    # Every registered tile variant of every family gives the same answer when the tuner picks it. The
    # bf16 family runs with an f32 bias (a bf16 bias goes to cuBLASLt).
    names = (ctypes.c_char_p * 256)()
    lx.lib.lx_tune_kernels.restype = ctypes.c_int
    count = lx.lib.lx_tune_kernels(names, 256)
    lx.lib.lx_tune_variant_name.restype = ctypes.c_char_p
    families = [names[i].decode() for i in range(min(count, 256)) if names[i].decode().startswith("gemm_w.")]
    assert families, "gemm_w registered no variants"
    by_kind = {"bf16": PAIRS[0], "w8i8": PAIRS[1], "w8f8": PAIRS[2], "w4a16": PAIRS[3], "i8": PAIRS[4], "f8": PAIRS[5],
               "w4a8": PAIRS[6], "f4": PAIRS[7]}
    have = native_arch(lx)
    # The family each kind runs on this card without an override.
    native = {}
    for fam in families:
        _, kind, sm = fam.split(".")
        if int(sm[2:]) <= have:
            native[kind] = max(native.get(kind, 0), int(sm[2:]))
    M, N, K = 200, 384, 1024
    for fam in families:
        _, kind, sm = fam.split(".")
        name, wtype, math, _ = by_kind[kind]
        fam_arch = int(sm[2:])
        if fam_arch > have:
            continue
        bias = "f32" if kind == "bf16" else None
        for v in range(lx.lib.lx_tune_variants(fam.encode())):
            vname = lx.lib.lx_tune_variant_name(fam.encode(), v).decode()
            split = "_x" in vname
            # Split-K variants need thread block clusters (sm_90+), so an override below 90 never picks
            # them: they run on the families this card reaches natively, and the f8.sm89 and w8f8.sm80 ones
            # only on an sm_90 or sm_100 card.
            if split and (have < 90 or fam_arch != native[kind]):
                continue
            lx.lib.lx_tune_set(fam.encode(), M, N, K, v)
            try:
                set_arch(lx, 0 if split else (fam_arch if fam_arch < have else 0))
                status, y, y0, (x, w, b) = run_case(lx, name, wtype, math, M, N, K, torch.bfloat16, torch.float32, 1.0, 0.0, bias, 0)
            finally:
                set_arch(lx, 0)
                lx.lib.lx_tune_clear()
            label = f"{fam} variant {vname}"
            assert status == 0, f"{label} returned {status}"
            want = reference(x, w, math, b, 1.0, 0.0, None)
            close(y, want.float(), atol=2e-4 * want.abs().max().item())


@pytest.mark.parametrize("split", [2, 4])
@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_gemm_w_split_k(lx, name, wtype, math, oldest, split):
    # The split-K variants (clusters along K, sm_90+) at a size where some CTAs of a cluster get no K tile
    # at all, with partial tiles, bias, beta and padded strides; the sum over the cluster runs in rank
    # order, so two calls give the same bits.
    have = native_arch(lx)
    if have < max(oldest, 90):
        pytest.skip("not on this GPU")
    kind = {"bf16": "bf16", "w8a16_int8": "w8i8", "w8a16_fp8": "w8f8", "w4a16": "w4a16", "int8": "i8", "fp8": "f8",
            "w4a8": "w4a8", "fp4": "f4"}[name]
    names = (ctypes.c_char_p * 256)()
    count = lx.lib.lx_tune_kernels(names, 256)
    fams = [names[i].decode() for i in range(min(count, 256)) if names[i].decode().startswith(f"gemm_w.{kind}.sm")]
    fam = max((f for f in fams if int(f.split(".sm")[1]) <= have), key=lambda f: int(f.split(".sm")[1])).encode()
    lx.lib.lx_tune_variant_name.restype = ctypes.c_char_p
    pick = [v for v in range(lx.lib.lx_tune_variants(fam)) if lx.lib.lx_tune_variant_name(fam, v).decode().endswith(f"_x{split}")]
    assert len(pick) == 1, f"{fam} has no _x{split} variant"
    M, N, K = 5, 136, 256
    outs = []
    lx.lib.lx_tune_set(fam, M, N, K, pick[0])
    try:
        for _ in range(2):
            status, y, y0, (x, w, bias) = run_case(lx, name, wtype, math, M, N, K, torch.bfloat16, torch.float32, 1.0, 1.0,
                                                   "f32", 4)
            assert status == 0, f"lx_gemm_w returned CUDA error {status}"
            outs.append(y)
    finally:
        lx.lib.lx_tune_clear()
    want = reference(x, w, math, bias, 1.0, 1.0, y0[:, :N])
    close(outs[0][:, :N], want.float(), atol=2e-4 * want.abs().max().item())
    assert torch.equal(outs[0], outs[1]), "split-K sum is not deterministic"
    assert torch.equal(outs[0][:, N:], y0[:, N:]), "wrote past the N columns of y"
