"""The persisting L2 window (kernels.h, lx_rt_l2_persist): a graph captured while a window is set replays to the same
bits as the same products run eagerly, and a backend without the window declines it.
"""

import ctypes

import pytest
import torch

from conftest import DEVICE

LX_ERR_NOT_SUPPORTED = 801


def _bind(lib):
    lib.lx_rt_l2_persist.restype = ctypes.c_int
    lib.lx_rt_l2_persist.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_float, ctypes.POINTER(ctypes.c_int64)]
    lib.lx_rt_capture_begin.restype = ctypes.c_int
    lib.lx_rt_capture_begin.argtypes = [ctypes.c_void_p]
    lib.lx_rt_capture_end.restype = ctypes.c_int
    lib.lx_rt_capture_end.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.lx_rt_graph_launch.restype = ctypes.c_int
    lib.lx_rt_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.lx_rt_graph_destroy.restype = ctypes.c_int
    lib.lx_rt_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.lx_gemv.restype = ctypes.c_int


def _gemv(lib, x, w, y, M, N, K, stream):
    # lx_gemv(x, x_dtype, ld_x, w, w_type, w_scale, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream):
    # bf16 x and w (LX_W_BF16 = 0), f32 y (LX_F32 = 0)
    status = lib.lx_gemv(
        ctypes.c_void_p(x.data_ptr()), 1, K, ctypes.c_void_p(w.data_ptr()), 0, None, None, 0,
        ctypes.c_void_p(y.data_ptr()), 0, N, M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0),
        ctypes.c_void_p(stream),
    )
    assert status == 0


def test_window_graph_matches_eager(lx):
    if DEVICE != "cuda":
        pytest.skip("the window is a CUDA L2 feature")
    lib = lx.lib
    _bind(lib)
    torch.manual_seed(0)
    M, N, K = 4, 1024, 2048
    w = torch.randn(N, K, device=DEVICE).to(torch.bfloat16)
    x = torch.randn(M, K, device=DEVICE).to(torch.bfloat16)
    y_eager = torch.empty(M, N, device=DEVICE)
    y_graph = torch.empty(M, N, device=DEVICE)
    stream = torch.cuda.Stream()
    try:
        # Asking first (no base) grants the same size the window then gets, and changes nothing.
        asked = ctypes.c_int64()
        status = lib.lx_rt_l2_persist(None, w.numel() * 2, 1.0, ctypes.byref(asked))
        if status == LX_ERR_NOT_SUPPORTED:
            pytest.skip("this device has no persisting L2 carve-out")
        assert status == 0 and 0 < asked.value <= w.numel() * 2
        granted = ctypes.c_int64()
        status = lib.lx_rt_l2_persist(w.data_ptr(), w.numel() * 2, 1.0, ctypes.byref(granted))
        assert status == 0 and granted.value == asked.value
        torch.cuda.synchronize()
        s = stream.cuda_stream
        _gemv(lib, x, w, y_eager, M, N, K, s)
        stream.synchronize()
        graph = ctypes.c_void_p()
        assert lib.lx_rt_capture_begin(s) == 0
        _gemv(lib, x, w, y_graph, M, N, K, s)
        assert lib.lx_rt_capture_end(s, ctypes.byref(graph)) == 0
        assert lib.lx_rt_graph_launch(graph, s) == 0
        stream.synchronize()
        assert torch.equal(y_graph, y_eager)
        assert lib.lx_rt_graph_destroy(graph) == 0
    finally:
        assert lib.lx_rt_l2_persist(None, 0, 0.0, None) == 0
