import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import DTYPE, close

DTYPES = [torch.float32, torch.bfloat16]


def check(actual, expected, dtype):
    """f32 outputs to 1e-5 relative; bf16 outputs to bf16 rounding (one step is 2^-8 relative)."""
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    close(actual, expected, atol=1e-6, rtol=rtol)


@pytest.mark.parametrize("src_dtype", DTYPES)
@pytest.mark.parametrize("dst_dtype", DTYPES)
@pytest.mark.parametrize(
    "rows,cols,col_off,pad_src,pad_dst",
    [
        (1, 2048, 0, 0, 0),  # Metis row, dense
        (300, 2048, 0, 0, 0),  # prefill-sized
        (7, 33, 0, 19, 5),  # odd width, padded strides on both sides
        (6, 40, 17, 24, 8),  # a column window out of a wider source
        (70000, 3, 0, 1, 2),  # more rows than one grid dimension's blocks
    ],
)
def test_copy2d(lx, src_dtype, dst_dtype, rows, cols, col_off, pad_src, pad_dst):
    torch.manual_seed(0)
    ld_src, ld_dst = col_off + cols + pad_src, cols + pad_dst
    src = torch.randn(rows, ld_src, device="cuda").to(src_dtype)
    dst = torch.full((rows, ld_dst), 7.0, device="cuda").to(dst_dtype)
    src_ptr = ctypes.c_void_p(src.data_ptr() + col_off * src.element_size())

    lx("copy2d", src_ptr, DTYPE[src_dtype], ld_src, dst, DTYPE[dst_dtype], ld_dst, rows, cols)

    # Casts round to nearest even in both torch and the kernel, so the copy is exact.
    close(dst[:, :cols], src[:, col_off : col_off + cols].to(dst_dtype), atol=0.0)
    if pad_dst:
        assert torch.equal(dst[:, cols:].float(), torch.full((rows, pad_dst), 7.0, device="cuda"))


@pytest.mark.parametrize("act,ref", [(0, torch.tanh), (1, F.silu), (2, torch.sigmoid)])
@pytest.mark.parametrize("n", [1, 7, 4096, 300 * 2048 + 3])
def test_activate_f32(lx, act, ref, n):
    torch.manual_seed(0)
    y = torch.randn(n, device="cuda") * 4.0
    want = ref(y.double())
    lx("activate_f32", y, n, act)
    close(y, want, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("out_dtype", DTYPES)
@pytest.mark.parametrize(
    "rows,cols,pad",
    [(1, 512, 0), (300, 512, 0), (5, 33, 7), (3, 2048, 0), (300, 2560, 0)],
)
def test_silu_mul(lx, out_dtype, rows, cols, pad):
    torch.manual_seed(0)
    ld_x = 2 * cols + pad
    x = torch.randn(rows, ld_x, device="cuda")
    out = torch.empty(rows, cols, device="cuda", dtype=out_dtype)

    lx("silu_mul", x, ld_x, out, DTYPE[out_dtype], rows, cols)

    gate = x[:, :cols].double()
    up = x[:, cols : 2 * cols].double()
    check(out, F.silu(gate) * up, out_dtype)


@pytest.mark.parametrize("rows,cols", [(1, 2048), (300, 2048), (4, 37), (9, 5)])
def test_add_row_bf16(lx, rows, cols):
    torch.manual_seed(0)
    y = torch.randn(rows, cols, device="cuda")
    row = torch.randn(cols, device="cuda").to(torch.bfloat16)
    want = y.double() + row.double()

    lx("add_row_bf16", y, row, rows, cols)

    close(y, want, atol=1e-6, rtol=1e-5)


def test_rejects_bad_arguments(lx):
    """Bad activation codes and strides that cannot clear their window return cudaErrorInvalidValue."""
    invalid = 1
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    y = torch.zeros(8, device="cuda")
    x = torch.zeros(2, 8, device="cuda")
    out = torch.zeros(2, 4, device="cuda")
    assert lx.lib.lx_activate_f32(ctypes.c_void_p(y.data_ptr()), ctypes.c_int64(8), ctypes.c_int(7), stream) == invalid
    assert lx.lib.lx_silu_mul(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_int(7), ctypes.c_void_p(out.data_ptr()), ctypes.c_int(0),
        ctypes.c_int(2), ctypes.c_int(4), stream,
    ) == invalid
    assert lx.lib.lx_copy2d(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_int(0), ctypes.c_int(3), ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(0), ctypes.c_int(4), ctypes.c_int(2), ctypes.c_int(4), stream,
    ) == invalid
