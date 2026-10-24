"""Weights in each format (kernels.h, "Weight formats") for the kernel tests.

A Weight owns the device tensors of one [N, K] matrix (or an [E, N, K] stack) in one format, the
lx_weight that names them, and the dense float64 values they stand for, computed from the stored
bytes. The dense values are the reference: a kernel that reads the format must agree with them.
"""

import ctypes

import torch

W_BF16, W_INT8, W_FP8, W_NVFP4, W_BF16X = 0, 1, 2, 3, 4
FORMATS = {"bf16": W_BF16, "int8": W_INT8, "fp8": W_FP8, "nvfp4": W_NVFP4}
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


class LxWeight(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("scale", ctypes.c_void_p),
        ("global_", ctypes.c_float),
        ("expert_global", ctypes.c_void_p),
    ]


def _ptr(t):
    return None if t is None else t.data_ptr()


class Weight:
    """One weight: its format, device tensors, the lx_weight and the dense reference [..., N, K]."""

    def __init__(self, fmt, data, scale=None, glob=1.0, expert_global=None, dense=None):
        self.fmt = fmt
        self.data = data
        self.scale = scale
        self.glob = float(glob)
        self.expert_global = expert_global
        self.dense = dense

    def lx(self):
        return LxWeight(
            type=self.fmt,
            data=_ptr(self.data),
            scale=_ptr(self.scale),
            global_=self.glob,
            expert_global=_ptr(self.expert_global),
        )


def _e4m3_bytes(x):
    """Round-to-nearest e4m3 (saturating) of a float tensor, as uint8 bytes."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_values(b):
    return b.view(torch.float8_e4m3fn).double()


def _e2m1_values(codes):
    """codes uint8 [..., K] with one code per element (low 4 bits): the signed E2M1 values, float64."""
    grid = E2M1_GRID.to(codes.device)
    mag = grid[(codes & 7).long()]
    return torch.where((codes & 8) > 0, -mag, mag)


def random_weight(fmt, N, K, seed, device="cuda", glob=0.0371):
    """A random [N, K] weight in format `fmt` (a W_* code), with a reference of its exact values."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if fmt == W_BF16:
        w = (torch.randn(N, K, generator=g) * 0.5).to(torch.bfloat16).to(device)
        return Weight(fmt, w, dense=w.double())
    if fmt == W_INT8:
        q = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int32).to(torch.int8).to(device)
        scale = (torch.rand(N, generator=g) * 0.02 + 0.001).float().to(device)
        return Weight(fmt, q, scale, dense=q.double() * scale.double()[:, None])
    if fmt == W_FP8:
        b = _e4m3_bytes(torch.randn(N, K, generator=g) * 3.0).to(device)
        scale = (torch.rand(N, generator=g) * 0.02 + 0.001).float().to(device)
        return Weight(fmt, b, scale, dense=_e4m3_values(b) * scale.double()[:, None])
    if fmt == W_NVFP4:
        assert K % 16 == 0
        codes = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.int32).to(torch.uint8).to(device)
        blocks = _e4m3_bytes(torch.rand(N, K // 16, generator=g) * 2.0).to(device)
        vals = _nvfp4_dense(codes, blocks, glob)
        return Weight(fmt, codes, blocks, glob=glob, dense=vals)
    raise ValueError(fmt)


def _nvfp4_dense(codes, blocks, glob):
    """Dense values of packed NVFP4 [..., N, K/2] with block scales [..., N, K/16] and a global."""
    lo, hi = codes & 15, codes >> 4
    inter = torch.stack((lo, hi), dim=-1).flatten(-2)  # low nibble is the even k
    vals = _e2m1_values(inter)
    scales = _e4m3_values(blocks).repeat_interleave(16, dim=-1)
    return vals * scales * float(glob)


def stack_weight(fmt, E, N, K, seed, device="cuda"):
    """An [E, N, K] stack in one format. NVFP4 experts each get their own global, as checkpoints carry."""
    parts = []
    for e in range(E):
        glob = 0.01 + 0.004 * e  # distinct per expert, so a kernel that uses the wrong one fails
        parts.append(random_weight(fmt, N, K, seed * 1000 + e, device, glob=glob))
    data = torch.stack([p.data for p in parts]).contiguous()
    scale = None if parts[0].scale is None else torch.stack([p.scale for p in parts]).contiguous()
    dense = torch.stack([p.dense for p in parts])
    if fmt == W_NVFP4:
        globals_ = torch.tensor([p.glob for p in parts], dtype=torch.float32, device=device)
        return Weight(fmt, data, scale, glob=parts[0].glob, expert_global=globals_, dense=dense)
    return Weight(fmt, data, scale, dense=dense)
