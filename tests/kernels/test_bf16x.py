"""Lossless compressed BF16 (engine/cuda/bf16x.cu, LX_W_BF16X) against PyTorch.

The format must reproduce every bf16 bit, so nothing here has a tolerance:
  - encoding matches a PyTorch encoder written from the layout in bf16x.cuh, byte for byte
  - decoding gives back the input bits, including zeros, subnormals, infinities and NaNs, rows with
    more patches than their slots (overflow records), and K that is not a multiple of 128
  - lx_gemv_bf16x equals lx_gemv on the original bf16 weights bit for bit, for every path (vec with
    each split, small, strided x, every M, every dtype), and is deterministic
  - lx_moe_experts_bf16x and lx_moe_experts_decode_bf16 equal lx_moe_experts bit for bit
"""

import ctypes

import pytest
import torch

from conftest import DTYPE, F32

W_BF16 = 0
METIS_SHAPES = [  # (N, K) of every Praxis decode product
    (10304, 2048),
    (3072, 2048),
    (2048, 4096),
    (65536, 2048),
    (1024, 2048),
    (2048, 1024),
    (1024, 512),
    (256, 6400),
    (256, 2048),
    (2048, 256),
    (256, 256),
]


# ---- the layout, written out in PyTorch -------------------------------------------------------


def layout(K):
    B = (K + 127) // 128
    H = (4 * (2 + K // 2048) + B + 7) // 8 * 8  # at least 2 + K / 2048 slots, more if the padding fits them
    P = (H - B) // 4
    bo = 4 * P
    return B, P, bo, H, H + 192 * B


def ref_encode(w):
    """bf16 [rows, K] to the bytes bf16x.cuh specifies (uint8, rows * row_bytes + overflow)."""
    rows, K = w.shape
    B, P, bo, H, rb = layout(K)
    u = (w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).cpu()
    Kp = 128 * B
    up = torch.zeros(rows, Kp, dtype=torch.int32)
    up[:, :K] = u
    valid = torch.zeros(rows, Kp, dtype=torch.bool)
    valid[:, :K] = True
    e = (up >> 7) & 0xFF
    e0 = torch.where(valid, e, 0).view(rows, B, 128).amax(-1)  # [rows, B]
    e0k = e0.repeat_interleave(128, dim=1)
    patched = valid & (e0k - e > 15)
    code = torch.where(valid & ~patched, e0k - e, 0)
    sm = torch.where(valid, ((up >> 8) & 0x80) | (up & 0x7F), 0)

    out = torch.zeros(rows, rb, dtype=torch.uint8)
    out[:, bo : bo + B] = e0.to(torch.uint8)
    blocks = torch.zeros(rows, B, 192, dtype=torch.uint8)
    blocks[:, :, :128] = sm.view(rows, B, 128).to(torch.uint8)
    c = code.view(rows, B, 16, 8)
    blocks[:, :, 128:] = (c[..., :4] | (c[..., 4:] << 4)).reshape(rows, B, 64).to(torch.uint8)
    out[:, H:] = blocks.view(rows, B * 192)

    slots = torch.full((rows, P), 0xFFFFFFFF, dtype=torch.int64)
    overflow = []
    region = rows * rb
    offset = 0
    for r in patched.any(1).nonzero().flatten().tolist():
        ks = patched[r].nonzero().flatten()
        entries = [(int(k) << 8) | int(e[r, k]) for k in ks]
        if len(entries) <= P:
            slots[r, : len(entries)] = torch.tensor(entries)
        else:
            size = (4 * (len(entries) + 2) + 7) // 8 * 8
            rec = torch.zeros(size // 4, dtype=torch.int64)
            rec[0] = len(entries)
            rec[1 : 1 + len(entries)] = torch.tensor(entries)
            rec[1 + len(entries)] = 0xFFFFFFFF
            slots[r, 0] = 0x80000000 | ((region + offset - r * rb) // 8)
            overflow.append(rec)
            offset += size
    out[:, :bo] = torch.from_numpy(slots.numpy().astype("<u4").view("u1").reshape(rows, bo))
    flat = out.view(-1)
    if overflow:
        recs = torch.cat(overflow)
        flat = torch.cat([flat, torch.from_numpy(recs.numpy().astype("<u4").view("u1"))])
    return flat


# ---- calls --------------------------------------------------------------------------------------


def nbytes(lx, w):
    fn = lx.lib.lx_bf16x_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    n = fn(w.data_ptr(), w.numel() // w.shape[-1], w.shape[-1], torch.cuda.current_stream().cuda_stream)
    assert n > 0, f"lx_bf16x_bytes returned {n}"
    return n


def row_bytes(lx, K):
    fn = lx.lib.lx_bf16x_row_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int]
    return fn(K)


def encode(lx, w):
    """w bf16 [..., K] (contiguous) to its encoding (uint8 on the GPU)."""
    rows, K = w.numel() // w.shape[-1], w.shape[-1]
    n = nbytes(lx, w)
    out = torch.full((n,), 0xAA, dtype=torch.uint8, device="cuda")  # poison: every byte must be written
    lx("bf16x_encode", w, ctypes.c_int64(rows), K, out, ctypes.c_int64(n))
    return out


def decode(lx, data, rows, K):
    w = torch.empty(rows, K, dtype=torch.bfloat16, device="cuda")
    lx("bf16x_decode", data, ctypes.c_int64(rows), K, w)
    return w


def bits(t):
    return t.contiguous().view(torch.int16)


def gemv(lx, name, x, w, bias, y, alpha, beta, M, N, K, ld_x, ld_y):
    bias_dtype = DTYPE[bias.dtype] if bias is not None else F32
    if name == "bf16":
        lx("gemv", x, DTYPE[x.dtype], ld_x, w, W_BF16, None, bias, bias_dtype, y, DTYPE[y.dtype], ld_y, M, N, K,
           float(alpha), float(beta))
    else:
        lx("gemv_bf16x", x, DTYPE[x.dtype], ld_x, w, bias, bias_dtype, y, DTYPE[y.dtype], ld_y, M, N, K,
           float(alpha), float(beta))


# ---- weights ------------------------------------------------------------------------------------


def trained_like(*shape):
    """Gaussian weights: like a trained matrix, a few values per row sit more than 15 binades below
    their block's largest, so rows with and without patches both occur."""
    return (torch.randn(*shape, device="cuda") * 0.02).to(torch.bfloat16)


def wide_range(*shape, lo=-60.0, hi=4.0):
    """Magnitudes log-uniform over 2^lo .. 2^hi with random signs: most blocks patch, many rows
    overflow their slots."""
    mag = torch.exp2(torch.empty(*shape, device="cuda").uniform_(lo, hi))
    sign = torch.where(torch.rand(*shape, device="cuda") < 0.5, -1.0, 1.0)
    return (mag * sign).to(torch.bfloat16)


def specials(rows, K):
    """Every class of bf16: zeros of both signs, subnormals, normals across the range, infinities, NaNs."""
    g = torch.Generator(device="cpu").manual_seed(rows * 7 + K)
    raw = torch.randint(0, 1 << 16, (rows, K), generator=g, dtype=torch.int32)
    pick = torch.randint(0, 8, (rows, K), generator=g)
    raw = torch.where(pick == 0, torch.tensor(0x0000), raw)       # +0
    raw = torch.where(pick == 1, torch.tensor(0x8000), raw)       # -0
    raw = torch.where(pick == 2, raw & 0x807F, raw)               # subnormal (or zero)
    raw = torch.where(pick == 3, (raw & 0x807F) | 0x7F80, raw)    # inf / NaN
    raw = torch.where(raw >= 0x8000, raw - 0x10000, raw).to(torch.int16)
    return raw.view(torch.bfloat16).cuda()


# ---- format --------------------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 5, 64, 127, 128, 129, 300, 512, 528, 1024, 2047, 2048, 4096, 6400])
def test_row_bytes_matches_layout(lx, K):
    assert row_bytes(lx, K) == layout(K)[4]
    assert row_bytes(lx, K) % 8 == 0


@pytest.mark.parametrize("make", ["trained", "wide", "specials"])
@pytest.mark.parametrize("rows,K", [(3, 1), (5, 7), (9, 128), (17, 300), (33, 528), (64, 2048), (8, 4096), (4, 6400)])
def test_encoding_matches_reference_layout(lx, make, rows, K):
    torch.manual_seed(rows + K)
    w = {"trained": trained_like, "wide": wide_range, "specials": specials}[make](rows, K)
    got = encode(lx, w).cpu()
    want = ref_encode(w)
    assert got.numel() == want.numel(), f"size {got.numel()} != {want.numel()}"
    diff = (got != want).nonzero()
    assert diff.numel() == 0, f"{diff.numel()} bytes differ, first at {diff[0].item()}"


@pytest.mark.parametrize("make", ["trained", "wide", "specials"])
@pytest.mark.parametrize("rows,K", [(1, 1), (7, 13), (100, 256), (37, 1000), (512, 2048), (64, 4096), (16, 6400),
                                    (3, 20000), (2, 70000)])
def test_round_trip_is_exact(lx, make, rows, K):
    torch.manual_seed(rows * 3 + K)
    w = {"trained": trained_like, "wide": wide_range, "specials": specials}[make](rows, K)
    data = encode(lx, w)
    assert torch.equal(bits(decode(lx, data, rows, K)), bits(w))


def test_overflow_rows_round_trip(lx):
    # Row 1 has 300 values 2^-100 below a block max of 1: far more patches than its 4 slots.
    rows, K = 4, 2048
    w = trained_like(rows, K)
    w[1, ::7] = 2.0**-100
    w[1, 0::128] = 1.0
    w[3, 5] = -(2.0**-120)
    data = encode(lx, w)
    assert data.numel() > rows * row_bytes(lx, K), "expected an overflow record"
    assert torch.equal(bits(decode(lx, data, rows, K)), bits(w))


def test_stacked_experts_are_rows(lx):
    torch.manual_seed(2)
    w = trained_like(6, 64, 512)
    data = encode(lx, w)
    assert torch.equal(bits(decode(lx, data, 6 * 64, 512)), bits(w.view(-1, 512)))
    assert torch.equal(data.cpu(), ref_encode(w.view(-1, 512)))


def test_encoding_is_deterministic(lx):
    torch.manual_seed(4)
    w = wide_range(300, 2048, lo=-30.0)
    assert torch.equal(encode(lx, w), encode(lx, w))


def test_trained_like_size(lx):
    # Gaussian weights land where Praxis does: 12.1 to 12.2 bits a value at K = 2048.
    torch.manual_seed(5)
    w = trained_like(4096, 2048)
    bits_per_value = 8 * nbytes(lx, w) / w.numel()
    assert 12.0 < bits_per_value < 12.2, bits_per_value


# ---- decode GEMV: bit-identical to lx_gemv on bf16 ------------------------------------------------


def both_gemv(lx, M, N, K, x_dtype, y_dtype, bias_dtype, alpha, beta, make=trained_like, ld_pad=0):
    w = make(N, K)
    data = encode(lx, w)
    x_buf = torch.randn(M, K + ld_pad, device="cuda").to(x_dtype)
    bias = None if bias_dtype is None else torch.randn(N, device="cuda").to(bias_dtype)
    y0 = torch.randn(M, N + ld_pad, device="cuda").to(y_dtype)
    ys = {}
    for name, weight in (("bf16", w), ("bf16x", data)):
        y = y0.clone() if beta != 0.0 else torch.full((M, N + ld_pad), float("nan"), device="cuda", dtype=y_dtype)
        gemv(lx, name, x_buf, weight, bias, y, alpha, beta, M, N, K, ld_x=K + ld_pad, ld_y=N + ld_pad)
        ys[name] = y
    return ys["bf16"], ys["bf16x"]


@pytest.mark.parametrize("K", [40, 528])  # 40 takes the small kernel, 528 the vec kernel
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, 0.0), (-1.5, 1.0)])
@pytest.mark.parametrize("y_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 8])
def test_gemv_dtype_matrix(lx, M, x_dtype, bias_dtype, y_dtype, alpha, beta, K):
    torch.manual_seed(M * 101 + K)
    ref, got = both_gemv(lx, M, 37, K, x_dtype, y_dtype, bias_dtype, alpha, beta)
    assert same(got, ref)


def same(a, b):
    return torch.equal(a.contiguous().view(torch.int32 if a.dtype == torch.float32 else torch.int16),
                       b.contiguous().view(torch.int32 if b.dtype == torch.float32 else torch.int16))


@pytest.mark.parametrize("M", list(range(1, 9)))
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_every_m(lx, x_dtype, M):
    torch.manual_seed(M)
    ref, got = both_gemv(lx, M, 300, 1024, x_dtype, torch.float32, torch.float32, 1.0, 0.0)
    assert same(got, ref)


@pytest.mark.parametrize("N,K", METIS_SHAPES)
@pytest.mark.parametrize("M", [1, 4, 8])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_metis_shapes(lx, N, K, M, x_dtype):
    torch.manual_seed(N * 31 + K + M)
    ref, got = both_gemv(lx, M, N, K, x_dtype, torch.float32, None, 1.0, 0.0)
    assert same(got, ref)


@pytest.mark.parametrize("N", [1, 8, 37, 130, 256, 1024, 4096])  # every split S from 8 down to 1
@pytest.mark.parametrize("K", [264, 1000, 2048, 6400])
def test_gemv_every_split(lx, N, K):
    torch.manual_seed(N + K)
    ref, got = both_gemv(lx, 3, N, K, torch.bfloat16, torch.float32, torch.bfloat16, 1.0, 0.0)
    assert same(got, ref)


@pytest.mark.parametrize("pad", [0, 3, 8])  # 3 breaks 16-byte alignment of x (small kernel), 8 keeps it
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 5])
def test_gemv_strided(lx, M, x_dtype, pad):
    torch.manual_seed(M * 7 + pad)
    ref, got = both_gemv(lx, M, 300, 1024, x_dtype, torch.float32, torch.bfloat16, 2.0, 0.5, ld_pad=pad)
    assert same(got, ref)


@pytest.mark.parametrize("K", [1, 5, 8, 13, 24, 64, 100, 128, 256, 257, 300])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_odd_and_small_k(lx, x_dtype, K):
    torch.manual_seed(K)
    ref, got = both_gemv(lx, 3, 21, K, x_dtype, torch.float32, None, 1.0, 0.0)
    assert same(got, ref)


@pytest.mark.parametrize("N,K", [(64, 2048), (300, 4096), (21, 300), (40, 40)])
@pytest.mark.parametrize("M", [1, 8])
def test_gemv_patches_and_overflow(lx, N, K, M):
    # Wide-range weights: most blocks carry patches and many rows overflow into records.
    torch.manual_seed(N + K + M)
    ref, got = both_gemv(lx, M, N, K, torch.float32, torch.float32, None, 1.0, 0.0,
                         make=lambda n, k: wide_range(n, k, lo=-40.0, hi=2.0))
    assert same(got, ref)


@pytest.mark.parametrize("M,N,K", [(1, 256, 2048), (8, 8, 2048), (8, 10304, 2048), (3, 128, 1280)])
def test_gemv_is_deterministic(lx, M, N, K):
    torch.manual_seed(M + N)
    w = wide_range(N, K, lo=-30.0)
    data = encode(lx, w)
    x = torch.randn(M, K, device="cuda")
    first = torch.empty(M, N, device="cuda")
    second = torch.empty(M, N, device="cuda")
    gemv(lx, "bf16x", x, data, None, first, 1.0, 0.0, M, N, K, K, N)
    gemv(lx, "bf16x", x, data, None, second, 1.0, 0.0, M, N, K, K, N)
    assert same(first, second)


# ---- decode experts: bit-identical to lx_moe_experts ---------------------------------------------


def make_experts(T, E, L, I, top_k, make=trained_like):
    latent = torch.randn(T, L, device="cuda").to(torch.bfloat16).contiguous()
    gate_up = make(E, 2 * I, L).contiguous()
    down = make(E, L, I).contiguous()
    ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
    return latent, ids, weights, gate_up, down


def workspace(lx, T, I, top_k):
    fn = lx.lib.lx_moe_bf16x_workspace_bytes
    fn.restype = ctypes.c_int64
    return torch.empty(fn(T, I, top_k), dtype=torch.uint8, device="cuda")


def run_experts(lx, name, latent, ids, weights, gate_up, down, T, L, I, top_k):
    out = torch.full((T, L), float("nan"), device="cuda")
    if name == "bf16":
        lx("moe_experts", latent, ids, weights, gate_up, down, out, T, L, I, top_k)
    else:
        ws = workspace(lx, T, I, top_k)
        fn = "moe_experts_bf16x" if name == "bf16x" else "moe_experts_decode_bf16"
        lx(fn, latent, ids, weights, gate_up, down, out, ws, T, L, I, top_k)
    return out


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_experts_metis_shape_bit_identical(lx, T):
    E, L, I, top_k = 128, 1024, 512, 4
    torch.manual_seed(T)
    latent, ids, weights, gate_up, down = make_experts(T, E, L, I, top_k)
    ref = run_experts(lx, "bf16", latent, ids, weights, gate_up, down, T, L, I, top_k)
    gu_x, dn_x = encode(lx, gate_up), encode(lx, down)
    got = run_experts(lx, "bf16x", latent, ids, weights, gu_x, dn_x, T, L, I, top_k)
    plain = run_experts(lx, "decode_bf16", latent, ids, weights, gate_up, down, T, L, I, top_k)
    assert same(got, ref), "bf16x experts differ from lx_moe_experts"
    assert same(plain, ref), "decode_bf16 experts differ from lx_moe_experts"


@pytest.mark.parametrize("E,L,I,top_k,T", [
    (8, 1024, 512, 4, 3),     # repeated experts across tokens
    (16, 2048, 512, 2, 2),    # L > 1024: rows loaded in two rounds of chunks
    (8, 512, 1000, 4, 1),     # I % 256 != 0: partial chunks; four down chunks
    (8, 256, 1536, 3, 2),     # I > 1024: down chunks past the prefetched ones
    (12, 1024, 512, 8, 2),    # top_k 8
    (6, 300, 200, 4, 2),      # L, I % 8 == 0 but not % 128
    (6, 100, 52, 2, 3),       # small rows
    (5, 70, 36, 2, 2),        # L % 8 != 0: scalar gate/up
    (5, 64, 30, 3, 2),        # I % 8 != 0: scalar down
])
@pytest.mark.parametrize("make", ["trained", "wide"])
def test_experts_shapes_bit_identical(lx, E, L, I, top_k, T, make):
    torch.manual_seed(E + L + I + top_k + T)
    mk = trained_like if make == "trained" else (lambda *s: wide_range(*s, lo=-40.0, hi=1.0))
    latent, ids, weights, gate_up, down = make_experts(T, E, L, I, top_k, mk)
    if E == 8 and T == 3:
        ids = ids[:1].repeat(T, 1).contiguous()
    ref = run_experts(lx, "bf16", latent, ids, weights, gate_up, down, T, L, I, top_k)
    got = run_experts(lx, "bf16x", latent, ids, weights, encode(lx, gate_up), encode(lx, down), T, L, I, top_k)
    plain = run_experts(lx, "decode_bf16", latent, ids, weights, gate_up, down, T, L, I, top_k)
    assert same(got, ref)
    assert same(plain, ref)
