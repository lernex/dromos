"""The architecture override and the tuning table: the fallback paths compute what the cluster paths do.

Run the whole suite under an override with --lx-arch=80 (or 89): every kernel then takes the paths a card
of that architecture would. These tests check the switches themselves, and that both paths of the
decode kernels that have two agree.
"""

import ctypes

import pytest
import torch

from conftest import close


def _lib(lx):
    lib = lx.lib
    lib.lx_arch.restype = ctypes.c_int
    lib.lx_set_arch_override.restype = ctypes.c_int
    lib.lx_set_arch_override.argtypes = [ctypes.c_int]
    lib.lx_tune_set.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.lx_tune_set.restype = None
    lib.lx_tune_get.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.lx_tune_get.restype = ctypes.c_int
    lib.lx_tune_clear.restype = None
    return lib


def test_override_lowers_only(lx):
    lib = _lib(lx)
    session = lib.lx_arch()  # the architecture the rest of the suite runs as: restored at the end
    real = lib.lx_set_arch_override(0)
    assert real >= 80
    assert lib.lx_set_arch_override(80) == 80
    assert lib.lx_arch() == 80
    assert lib.lx_set_arch_override(10_000) == real  # a higher override is refused
    assert lib.lx_set_arch_override(session if session != real else 0) == session


@pytest.mark.parametrize("T,S,D", [(1, 4, 2048), (4, 4, 2048), (8, 4, 2560), (3, 8, 37)])
def test_mhc_read_table_paths_agree(lx, T, S, D):
    """Both paths of the decode read give the same result, so the tuner may choose either."""
    lib = _lib(lx)
    torch.manual_seed(1)
    streams = torch.randn(T, S, D, device="cuda")
    matrix = torch.randn(S, S, device="cuda")
    read = torch.randn(S, device="cuda")
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    outs = []
    for variant in (0, 1):
        lib.lx_tune_clear()
        lib.lx_tune_set(b"mhc_read", T, D, S, variant)
        mixed = torch.empty(T, S, D, device="cuda")
        h = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)
        lx("mhc_read", streams, matrix, read, norm_w, 1e-6, mixed, h, T, S, D)
        outs.append((mixed, h))
    lib.lx_tune_clear()
    close(outs[0][0], outs[1][0], atol=1e-6, rtol=1e-5)
    close(outs[0][1].float(), outs[1][1].float(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("T,S,D", [(1, 4, 2048), (8, 4, 2560), (3, 3, 37)])
def test_memory_fuse_table_paths_agree(lx, T, S, D):
    lib = _lib(lx)
    torch.manual_seed(2)
    projected = torch.randn(T, S, D, device="cuda")
    gate = (torch.randn(S, D, device="cuda") * 0.1).to(torch.bfloat16)
    gate_bias = (torch.randn(S, device="cuda") * 0.1).to(torch.bfloat16)
    outs = []
    for variant in (0, 1):
        lib.lx_tune_clear()
        lib.lx_tune_set(b"memory_fuse", T, D, S, variant)
        streams = torch.randn(T, S, D, device="cuda", generator=torch.Generator("cuda").manual_seed(3))
        summary = torch.empty(T, D, device="cuda")
        lx("memory_fuse", streams, projected, gate, gate_bias, summary, T, S, D)
        outs.append((streams, summary))
    lib.lx_tune_clear()
    close(outs[0][0], outs[1][0], atol=1e-5, rtol=1e-4)
    close(outs[0][1], outs[1][1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "T,H,KVH,Dh,p0",
    [(1, 32, 8, 64, 0), (1, 32, 8, 64, 100), (4, 32, 8, 64, 40), (8, 8, 2, 64, 200), (1, 40, 10, 64, 255), (64, 32, 8, 64, 0)],
)
def test_attention_split_and_whole_items_agree(lx, T, H, KVH, Dh, p0):
    """Decode attention splits a token's cache over a cluster on sm_90 and up; below that each block takes
    whole items. Same math, so the outputs agree to bf16 rounding of the cache and the output."""
    lib = _lib(lx)
    session = lib.lx_arch()
    torch.manual_seed(4)
    capacity, slots = 512, 1
    qkv = torch.randn(T, (H + 2 * KVH) * Dh, device="cuda")
    k0 = torch.randn(slots, capacity, KVH, Dh, device="cuda").to(torch.bfloat16)
    v0 = torch.randn(slots, capacity, KVH, Dh, device="cuda").to(torch.bfloat16)
    slot_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    start = torch.tensor([p0], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    scale = 1.0 / Dh**0.5
    outs = []
    try:
        for arch in (0, 80):  # 0: the card as it is, 80: the whole-items path
            lib.lx_set_arch_override(arch)
            kc, vc = k0.clone(), v0.clone()
            out = torch.empty(T, H * Dh, device="cuda", dtype=torch.bfloat16)
            lx("attention_varlen", qkv, (H + 2 * KVH) * Dh, kc, vc, slot_ids, start, cu, 1, capacity, out, H, KVH, Dh,
               scale)
            outs.append((out, kc))
    finally:
        real = lib.lx_set_arch_override(0)
        lib.lx_set_arch_override(session if session != real else 0)
    close(outs[0][0].float(), outs[1][0].float(), atol=2e-2, rtol=2e-2)
    close(outs[0][1].float(), outs[1][1].float(), atol=0, rtol=0)  # the cache is written the same either way
