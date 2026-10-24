"""Reference quantizers for the weight formats and activation maths (kernels.h, "Weight formats" and
lx_quantize_act), in PyTorch, written to reproduce the kernels' roundings bit for bit, plus the
lx_weight struct for ctypes. Shared by the GEMM, expert and quantization tests and the benches."""

import ctypes

import torch

W_BF16, W_INT8, W_FP8, W_NVFP4 = 0, 1, 2, 3
MATH_BF16, MATH_INT8, MATH_FP8, MATH_FP4 = 0, 1, 2, 3

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class LxWeight(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("scale", ctypes.c_void_p),
        ("glob", ctypes.c_float),
        ("expert_global", ctypes.c_void_p),  # kernels.h's last field: null, so stacks use glob
    ]


class Weight:
    """A weight in one format: data, scale and global, as kernels.h lays them out. Keeps the tensors
    alive for as long as the ctypes struct points at them."""

    def __init__(self, wtype, data, scale=None, glob=1.0):
        self.wtype, self.data, self.scale, self.glob = wtype, data.contiguous(), scale, float(glob)
        if self.scale is not None:
            self.scale = self.scale.contiguous()
        self.struct = LxWeight(
            wtype, self.data.data_ptr(), self.scale.data_ptr() if self.scale is not None else None, self.glob, None
        )

    def ref(self):
        """The weight's exact values as float64."""
        return dequant_weight(self)


def e2m1_rne(v):
    """Round to the nearest e2m1 value (ties to even, saturating at 6); returns the 4-bit codes."""
    a = v.abs()
    m = torch.zeros_like(a, dtype=torch.int32)
    m = torch.where(a > 0.25, 1, m)
    m = torch.where(a >= 0.75, 2, m)
    m = torch.where(a > 1.25, 3, m)
    m = torch.where(a >= 1.75, 4, m)
    m = torch.where(a > 2.5, 5, m)
    m = torch.where(a >= 3.5, 6, m)
    m = torch.where(a > 5.0, 7, m)
    sign = ((v < 0) & (m != 0)).to(torch.int32) * 8
    return (m | sign).to(torch.uint8)


def e2m1_decode(codes):
    vals = E2M1_VALUES.to(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -vals, vals)


def pack_nibbles(codes):
    """[..., K] 4-bit codes to [..., K / 2] bytes, the even k in the low nibble."""
    c = codes.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed):
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def fdiv(a, b):
    """float32 a / b with IEEE division. PyTorch divides by a Python scalar as a multiply by its
    reciprocal, which rounds differently from the kernels' division, so the divisor is made a tensor."""
    if not isinstance(b, torch.Tensor):
        b = torch.full_like(a, b)
    if not isinstance(a, torch.Tensor):
        a = torch.full_like(b, a)
    return a / b


def e4m3(v):
    """float32 -> e4m3 bytes (round to nearest even, saturating), as uint8."""
    return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(b):
    return b.view(torch.float8_e4m3fn).float()


# ---- weights --------------------------------------------------------------------------------------


def quantize_weight(w, wtype):
    """w float/bf16 [..., N, K] -> Weight in the format, round to nearest (kernels.h lx_quantize_weight)."""
    w = w.float()
    if wtype == W_BF16:
        return Weight(W_BF16, w.to(torch.bfloat16))
    if wtype in (W_INT8, W_FP8):
        qmax = 127.0 if wtype == W_INT8 else 448.0
        absmax = w.abs().amax(-1)
        scale = torch.where(absmax > 0, fdiv(absmax, qmax), torch.ones_like(absmax))
        v = fdiv(w, scale.unsqueeze(-1).expand_as(w))
        if wtype == W_INT8:
            q = torch.round(v).clamp(-127, 127).to(torch.int8)
        else:
            q = e4m3(v)
        return Weight(wtype, q, scale.float())
    # NVFP4: one global for the whole tensor (stack), an e4m3 scale per 16 values
    K = w.shape[-1]
    assert K % 16 == 0
    amax = w.abs().max().reshape(1)
    glob_t = torch.where(amax > 0, fdiv(amax, 448.0 * 6.0), torch.ones_like(amax))
    glob = glob_t.item()
    blocks = w.reshape(*w.shape[:-1], K // 16, 16)
    bmax = blocks.abs().amax(-1)
    s8 = e4m3(fdiv(bmax, (6.0 * glob_t).expand_as(bmax)))
    s = e4m3_decode(s8)
    d = (s * glob_t).unsqueeze(-1)
    codes = torch.where(s.unsqueeze(-1) > 0, e2m1_rne(fdiv(blocks, d.expand_as(blocks))).int(), 0).to(torch.uint8)
    return Weight(W_NVFP4, pack_nibbles(codes.reshape(*w.shape)), s8, glob)


def dequant_weight(wt):
    if wt.wtype == W_BF16:
        return wt.data.double()
    if wt.wtype == W_INT8:
        return wt.data.double() * wt.scale.double().unsqueeze(-1)
    if wt.wtype == W_FP8:
        return e4m3_decode(wt.data).double() * wt.scale.double().unsqueeze(-1)
    codes = unpack_nibbles(wt.data)
    vals = e2m1_decode(codes).double()
    K = vals.shape[-1]
    s = e4m3_decode(wt.scale).double()
    return (vals.reshape(*vals.shape[:-1], K // 16, 16) * s.unsqueeze(-1)).reshape(vals.shape) * wt.glob


# ---- activations ----------------------------------------------------------------------------------


def quantize_act(x, math):
    """x [M, K] (f32 or bf16) -> (q, scale, block_scale) exactly as lx_quantize_act writes them."""
    xf = x.float()
    a = xf.abs().amax(-1)
    if math in (MATH_INT8, MATH_FP8):
        qmax = 127.0 if math == MATH_INT8 else 448.0
        inv = torch.where(a > 0, fdiv(qmax, a), torch.zeros_like(a))
        v = xf * inv.unsqueeze(-1)
        if math == MATH_INT8:
            q = torch.round(v).clamp(-127, 127).to(torch.int8).view(torch.uint8)
        else:
            q = e4m3(v)
        scale = torch.where(a > 0, fdiv(a, qmax), torch.ones_like(a))
        return q, scale, None
    M, K = xf.shape
    g = torch.where(a > 0, fdiv(a, 448.0 * 6.0), torch.ones_like(a))
    blocks = xf.view(M, K // 16, 16)
    bmax = blocks.abs().amax(-1)
    s8 = e4m3(fdiv(bmax, 6.0 * g.unsqueeze(-1)))
    s = e4m3_decode(s8)
    r = fdiv(1.0, s * g.unsqueeze(-1)).unsqueeze(-1)  # the kernel multiplies by the block's reciprocal
    codes = torch.where(s.unsqueeze(-1) > 0, e2m1_rne(blocks * r.expand_as(blocks)).int(), 0).to(torch.uint8)
    return pack_nibbles(codes.view(M, K)), g, s8


def dequant_act(q, scale, block_scale, math, K):
    if math == MATH_INT8:
        return q[:, :K].view(torch.int8).double() * scale.double().unsqueeze(-1)
    if math == MATH_FP8:
        return e4m3_decode(q[:, :K].contiguous()).double() * scale.double().unsqueeze(-1)
    M = q.shape[0]
    vals = e2m1_decode(unpack_nibbles(q[:, : K // 2])).double()
    s = e4m3_decode(block_scale[:, : K // 16].contiguous()).double()
    return (vals.view(M, K // 16, 16) * s.unsqueeze(-1)).view(M, K) * scale.double().unsqueeze(-1)


def act_operand(x, math):
    """The activations as the GEMM's math sees them (float64): bf16-rounded for BF16 math, else the
    dequantized per-token quantization."""
    if math == MATH_BF16:
        return x.to(torch.bfloat16).double()
    q, s, bs = quantize_act(x, math)
    return dequant_act(q, s, bs, math, x.shape[1])
