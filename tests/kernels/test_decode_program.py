"""Decode programs (engine/cuda/decode_*.cu, lx_program_*) against PyTorch references.

Each test builds a program of one or a few ops, runs it on the persistent kernel, and compares with a
float64 reference of what the op means in a Metis step (kernels.h documents every op). Programs run
twice: the counters a step leaves behind must be reset for the next one, and ops that only overwrite
must give the same result again.
"""

import ctypes
import math

import pytest
import torch
import torch.nn.functional as F

from conftest import BF16, F32, close

W_BF16, W_INT8, W_F32 = 0, 1, 16
FX_PLAIN, FX_RMS, FX_GATED, FX_CONCAT = 0, 1, 2, 3
FY_STORE, FY_WRITE = 0, 1
c_int, c_float, c_void_p = ctypes.c_int, ctypes.c_float, ctypes.c_void_p


def ptr(t):
    return c_void_p(t.data_ptr()) if t is not None else c_void_p(None)


class Weight(ctypes.Structure):  # lx_weight
    _fields_ = [("type", c_int), ("data", c_void_p), ("scale", c_void_p), ("global_", c_float),
                ("expert_global", c_void_p)]


class Fx(ctypes.Structure):
    _fields_ = [("kind", c_int), ("x", c_void_p), ("dtype", c_int), ("ld", c_int), ("parts", c_void_p),
                ("n_parts", c_int), ("norm_w", c_void_p), ("eps", c_float), ("group", c_int),
                ("seg", c_void_p * 4), ("width", c_int * 4), ("round_bf16", c_int)]


class Fy(ctypes.Structure):
    _fields_ = [("kind", c_int), ("out", c_void_p), ("dtype", c_int), ("ld", c_int), ("bias", c_void_p),
                ("bias_dtype", c_int), ("alpha", c_float), ("beta", c_float), ("act", c_int), ("out_bf", c_void_p),
                ("ld_bf", c_int), ("lora_up", c_void_p), ("lora", c_void_p), ("rank", c_int), ("lora_gate", c_float),
                ("streams", c_void_p), ("matrix", c_void_p), ("write", c_void_p), ("S", c_int), ("state", c_void_p),
                ("next_read", c_void_p), ("next_src", c_void_p), ("next_parts", c_void_p)]


class MemReadArgs(ctypes.Structure):
    _fields_ = [("streams", c_void_p), ("bank", c_void_p), ("n_entries", c_int), ("entry_stride", c_int),
                ("token_stride", c_int), ("mem_dim", c_int), ("gate", c_void_p), ("gate_bias", c_void_p),
                ("state", c_void_p), ("summary", c_void_p), ("read", c_void_p), ("norm_w", c_void_p), ("eps", c_float),
                ("h", c_void_p), ("lora_down", c_void_p), ("rank", c_int), ("lora", c_void_p), ("previous", c_void_p),
                ("difference", c_void_p), ("rms", c_void_p), ("S", c_int), ("D", c_int)]


class SsmArgs(ctypes.Structure):
    _fields_ = [("zxbcdt", c_void_p), ("ld", c_int), ("conv_w", c_void_p), ("conv_b", c_void_p),
                ("conv_state", c_void_p), ("ssm_state", c_void_p), ("A_log", c_void_p), ("D", c_void_p),
                ("dt_bias", c_void_p), ("v", c_void_p), ("head_ss", c_void_p), ("I", c_int), ("H", c_int),
                ("P", c_int), ("G", c_int), ("N", c_int), ("W", c_int)]


class SelectArgs(ctypes.Structure):
    _fields_ = [("logits", c_void_p), ("selection_bias", c_void_p), ("ids", c_void_p), ("weights", c_void_p),
                ("entropy", c_void_p), ("confidence", c_void_p), ("expert_emb", c_void_p), ("summary", c_void_p),
                ("history", c_void_p), ("E", c_int), ("top_k", c_int), ("R", c_int)]


class ExpertsArgs(ctypes.Structure):
    _fields_ = [("latent", c_void_p), ("ids", c_void_p), ("weights", c_void_p), ("gate_up", Weight),
                ("down", Weight), ("shared_gate_up", Weight), ("shared_down", Weight), ("out", c_void_p),
                ("L", c_int), ("I", c_int), ("Is", c_int), ("top_k", c_int)]


class AttentionArgs(ctypes.Structure):
    _fields_ = [("qkv", c_void_p), ("ld_qkv", c_int), ("k_cache", c_void_p), ("v_cache", c_void_p),
                ("capacity", c_int), ("out", c_void_p), ("H", c_int), ("KVH", c_int), ("Dh", c_int),
                ("scale", c_float)]


class EntryArgs(ctypes.Structure):
    _fields_ = [("state", c_void_p), ("w_state", Weight), ("b_state", c_void_p), ("summary", c_void_p),
                ("entropy", c_void_p), ("confidence", c_void_p), ("continuation", c_void_p), ("mean_k", c_float),
                ("inv_log_e", c_float), ("w_meta", c_void_p), ("b_meta", c_void_p), ("pass_emb", c_void_p),
                ("anchor_emb", c_void_p), ("entry", c_void_p), ("D", c_int), ("R", c_int), ("mem_dim", c_int)]


class Program:
    """One lx_program: add ops, build, run."""

    def __init__(self, lx, M, tokens=None, slots=None, has_state=None, start_pos=None):
        self.lib = lx.lib
        if not hasattr(self.lib, "lx_program_create"):
            # Metal (and CPU) export no decode program: a caller falls back to the per-kernel path (kernels.h).
            pytest.skip("this backend has no decode program")
        self.lib.lx_program_create.restype = c_void_p
        self.lib.lx_program_create.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_void_p]
        for name in ["gemv", "embed", "memory_read", "ssm", "select", "experts", "attention", "memory_entry",
                     "continuation", "build", "run", "tasks", "trace"]:
            getattr(self.lib, "lx_program_" + name).restype = c_int
        self.keep = [tokens, slots, has_state, start_pos]
        self.p = c_void_p(self.lib.lx_program_create(M, ptr(tokens), ptr(slots), ptr(has_state), ptr(start_pos)))
        assert self.p.value

    def add(self, name, *args):
        op = getattr(self.lib, "lx_program_" + name)(self.p, *args)
        assert op >= 0, f"lx_program_{name} failed"
        return op

    def gemv(self, fx, w, fy, N, K):
        self.keep += [fx, w, fy]
        return self.add("gemv", ctypes.byref(fx), ctypes.byref(w), ctypes.byref(fy), c_int(N), c_int(K))

    def tasks(self, op):
        return self.lib.lx_program_tasks(self.p, c_int(op))

    def build(self):
        status = self.lib.lx_program_build(self.p)
        assert status == 0, f"lx_program_build returned {status}"

    def run(self):
        status = self.lib.lx_program_run(self.p, c_void_p(torch.cuda.current_stream().cuda_stream))
        assert status == 0, f"lx_program_run returned {status}"
        torch.cuda.synchronize()

    def __del__(self):
        if getattr(self, "p", None) and self.p.value:
            self.lib.lx_program_destroy.argtypes = [c_void_p]
            self.lib.lx_program_destroy(self.p)


def weight(N, K, kind):
    """(lx_weight, dequantized float64 [N, K], tensors to keep)."""
    if kind == "bf16":
        w = (torch.randn(N, K, device="cuda") / math.sqrt(K)).to(torch.bfloat16)
        return Weight(W_BF16, w.data_ptr(), None, 1.0), w.double(), [w]
    if kind == "f32":
        w = torch.randn(N, K, device="cuda") / math.sqrt(K)
        return Weight(W_F32, w.data_ptr(), None, 1.0), w.double(), [w]
    q = torch.randint(-127, 128, (N, K), device="cuda").to(torch.int8)
    s = (torch.rand(N, device="cuda") + 0.5) / (127 * math.sqrt(K))
    return Weight(W_INT8, q.data_ptr(), s.data_ptr(), 1.0), q.double() * s.double()[:, None], [q, s]


def fx_plain(x):
    fx = Fx()
    fx.kind, fx.x, fx.dtype, fx.ld = FX_PLAIN, x.data_ptr(), BF16 if x.dtype == torch.bfloat16 else F32, x.shape[1]
    return fx


def fy_store(out, bias=None, alpha=1.0, beta=0.0, act=0, out_bf=None):
    fy = Fy()
    fy.kind, fy.out, fy.dtype, fy.ld = FY_STORE, out.data_ptr(), BF16 if out.dtype == torch.bfloat16 else F32, out.shape[1]
    if bias is not None:
        fy.bias, fy.bias_dtype = bias.data_ptr(), BF16 if bias.dtype == torch.bfloat16 else F32
    fy.alpha, fy.beta, fy.act = alpha, beta, act
    if out_bf is not None:
        fy.out_bf, fy.ld_bf = out_bf.data_ptr(), out_bf.shape[1]
    return fy


def bf(t):
    return t.to(torch.bfloat16).double()


def check(got, want, rel=2e-4):
    scale = want.abs().max().item()
    close(got, want, atol=rel * scale + 1e-30)


# ---- Products -----------------------------------------------------------------------------------

GEMV_CASES = [  # N, K, weight kinds
    (10304, 2048, ["int8", "bf16"]),  # in_proj
    (2048, 4096, ["bf16"]),           # out_proj, kept bf16
    (65536, 2048, ["int8"]),          # the head
    (1024, 512, ["int8", "bf16"]),    # shared expert down
    (256, 6400, ["f32"]),             # the depth head's hidden layer: rows wider than a page
    (128, 1280, ["f32"]),             # a router
    (4100, 256, ["int8", "bf16"]),    # the memory bank projection
    (40, 72, ["int8", "bf16", "f32"]),
]


@pytest.mark.parametrize("N,K,kinds", GEMV_CASES)
@pytest.mark.parametrize("M", [1, 3, 8])
def test_gemv_store(lx, N, K, kinds, M):
    torch.manual_seed(N + K + M)
    for kind in kinds:
        w, wd, keep = weight(N, K, kind)
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        out = torch.zeros(M, N, device="cuda")
        prog = Program(lx, M)
        prog.gemv(fx_plain(x), w, fy_store(out), N, K)
        prog.build()
        want = x.double() @ wd.T
        for _ in range(2):
            prog.run()
            check(out, want)


@pytest.mark.parametrize("M", [1, 5])
def test_gemv_epilogue(lx, M):
    """bias (bf16 and f32), alpha / beta, tanh and silu, a bf16 output and its copy, and the LoRA term."""
    torch.manual_seed(7)
    N, K, R = 3072, 2048, 8
    w, wd, keep = weight(N, K, "int8")
    x = torch.randn(M, K, device="cuda")  # f32 x
    bias16 = torch.randn(N, device="cuda").to(torch.bfloat16)
    bias32 = torch.randn(N, device="cuda")
    lora_up = (torch.randn(N, R, device="cuda") * 0.3).to(torch.bfloat16)
    lora = bf(torch.randn(M, R, device="cuda")).float()
    y0 = torch.randn(M, N, device="cuda")
    out_a = y0.clone()
    out_b = torch.zeros(M, N, device="cuda").to(torch.bfloat16)
    copy_b = torch.zeros(M, N, device="cuda").to(torch.bfloat16)
    out_c = torch.zeros(M, N, device="cuda")
    prog = Program(lx, M)
    prog.gemv(fx_plain(x), w, fy_store(out_a, bias16, alpha=0.5, beta=-1.5), N, K)
    prog.gemv(fx_plain(x), w, fy_store(out_b, bias32, act=1, out_bf=copy_b), N, K)
    fy = fy_store(out_c, act=2)
    fy.lora_up, fy.lora, fy.rank, fy.lora_gate = lora_up.data_ptr(), lora.data_ptr(), R, 0.75
    prog.gemv(fx_plain(x), w, fy, N, K)
    prog.build()
    prog.run()
    y = x.double() @ wd.T
    check(out_a, 0.5 * (y + bias16.double()) - 1.5 * y0.double())
    check(out_b.float(), torch.tanh(y + bias32.double()), rel=2**-8)
    check(copy_b.float(), torch.tanh(y + bias32.double()), rel=2**-8)
    v = y + 0.75 * (lora.double() @ lora_up.double().T)
    check(out_c, v * torch.sigmoid(v))


def rms_x(src, parts_sum, K, w, eps):
    """The bf16 row an RMS prologue stages, computed in f32 the way the kernel does."""
    scale = torch.rsqrt(parts_sum / K + eps)
    return ((src * scale[:, None]) * w.float()[None, :]).to(torch.bfloat16)


@pytest.mark.parametrize("M", [1, 4, 8])
def test_gemv_rms_and_gated(lx, M):
    torch.manual_seed(11 + M)
    K, N, n_parts, eps = 2048, 1024, 37, 1e-6
    src = torch.randn(M, K, device="cuda")
    sq = src.square()
    # partial sums of squares over arbitrary slices, as a WRITE epilogue leaves them
    cuts = torch.linspace(0, K, n_parts + 1).long().tolist()
    parts = torch.stack([sq[:, a:b].sum(1) for a, b in zip(cuts[:-1], cuts[1:])], 1).contiguous()
    norm_w = (torch.rand(K, device="cuda") + 0.5).to(torch.bfloat16)
    w, wd, keep = weight(N, K, "int8")
    out = torch.zeros(M, N, device="cuda")
    fx = Fx()
    fx.kind, fx.x, fx.dtype, fx.ld = FX_RMS, src.data_ptr(), F32, K
    fx.parts, fx.n_parts, fx.norm_w, fx.eps = parts.data_ptr(), n_parts, norm_w.data_ptr(), eps
    # gated: 8 groups of 512, the sums of squares per head of 64
    I, G = 4096, 8
    v = torch.randn(M, I, device="cuda")
    head_ss = v.square().view(M, 64, 64).sum(2).contiguous()
    gw = (torch.rand(I, device="cuda") + 0.5).to(torch.bfloat16)
    w2, wd2, keep2 = weight(2048, I, "bf16")
    out2 = torch.zeros(M, 2048, device="cuda")
    fx2 = Fx()
    fx2.kind, fx2.x, fx2.dtype, fx2.ld = FX_GATED, v.data_ptr(), F32, I
    fx2.parts, fx2.n_parts, fx2.norm_w, fx2.eps, fx2.group = head_ss.data_ptr(), 64, gw.data_ptr(), 1e-5, I // G
    prog = Program(lx, M)
    prog.gemv(fx, w, fy_store(out), N, K)
    prog.gemv(fx2, w2, fy_store(out2), 2048, I)
    prog.build()
    prog.run()
    xr = rms_x(src, parts.sum(1), K, norm_w, eps)
    check(out, xr.double() @ wd.T, rel=1e-3)
    vg = v.view(M, G, I // G)
    scale = torch.rsqrt(vg.square().sum(2) / (I // G) + 1e-5)
    xg = ((vg * scale[:, :, None]).view(M, I) * gw.float()[None, :]).to(torch.bfloat16)
    check(out2, xg.double() @ wd2.T, rel=1e-3)


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("round_bf16", [0, 1])
def test_gemv_concat(lx, M, round_bf16):
    torch.manual_seed(3 + M)
    D, R = 2048, 256
    a, b, c = torch.randn(M, D, device="cuda"), torch.randn(M, D, device="cuda"), torch.randn(M, R, device="cuda")
    K, N = 2 * D + R, 256
    w, wd, keep = weight(N, K, "int8" if round_bf16 else "f32")
    out = torch.zeros(M, N, device="cuda")
    fx = Fx()
    fx.kind, fx.round_bf16 = FX_CONCAT, round_bf16
    for i, (t, wid) in enumerate([(a, D), (b, D), (c, R)]):
        fx.seg[i], fx.width[i] = t.data_ptr(), wid
    prog = Program(lx, M)
    prog.gemv(fx, w, fy_store(out, act=1), N, K)
    prog.build()
    prog.run()
    x = torch.cat([a, b, c], 1).double()
    if round_bf16:
        x = bf(x)
    check(out, torch.tanh(x @ wd.T))


@pytest.mark.parametrize("M", [1, 3, 8])
def test_write_then_rms_chain(lx, M):
    """A mixer's hyper-connection write, then the MoE read normalizing what it left (next_src and the
    per-tile sums of squares), then a second write: three dependent products in one program."""
    torch.manual_seed(5 + M)
    S, D, I = 4, 2048, 4096
    streams = torch.randn(M, S, D, device="cuda")
    s0 = streams.clone()
    matrix = torch.softmax(torch.randn(S, S, device="cuda"), 1).contiguous()
    write = torch.softmax(torch.randn(S, device="cuda"), 0)
    read2 = torch.softmax(torch.randn(S, device="cuda"), 0)
    matrix2 = torch.softmax(torch.randn(S, S, device="cuda"), 1).contiguous()
    write2 = torch.softmax(torch.randn(S, device="cuda"), 0)
    v = torch.randn(M, I, device="cuda").to(torch.bfloat16)
    w1, wd1, k1 = weight(D, I, "bf16")
    w2, wd2, k2 = weight(1024, D, "int8")
    w3, wd3, k3 = weight(D, 1024, "int8")
    next_src = torch.zeros(M, D, device="cuda")
    state = torch.zeros(M, D, device="cuda")
    latent = torch.zeros(M, 1024, device="cuda")
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    prog = Program(lx, M)
    n_tiles = 1
    parts = torch.zeros(M, 4096, device="cuda")  # big enough for any tiling
    fy = Fy()
    fy.kind, fy.streams, fy.matrix, fy.write, fy.S = FY_WRITE, streams.data_ptr(), matrix.data_ptr(), write.data_ptr(), S
    fy.next_read, fy.next_src, fy.next_parts = read2.data_ptr(), next_src.data_ptr(), parts.data_ptr()
    op1 = prog.gemv(fx_plain(v), w1, fy, D, I)
    n_tiles = prog.tasks(op1)
    # next_parts is [M, tiles]; the reader takes them as n_parts per token
    fx = Fx()
    fx.kind, fx.x, fx.dtype, fx.ld = FX_RMS, next_src.data_ptr(), F32, D
    fx.parts, fx.n_parts, fx.norm_w, fx.eps = parts.data_ptr(), n_tiles, norm_w.data_ptr(), 1e-6
    prog.gemv(fx, w2, fy_store(latent), 1024, D)
    fy3 = Fy()
    fy3.kind, fy3.streams, fy3.matrix, fy3.write, fy3.S = FY_WRITE, streams.data_ptr(), matrix2.data_ptr(), write2.data_ptr(), S
    fy3.state = state.data_ptr()
    prog.gemv(fx_plain(latent), w3, fy3, D, 1024)
    prog.build()
    prog.run()

    y1 = v.double() @ wd1.T
    new1 = torch.einsum("si,mid->msd", matrix.double(), s0.double()) + write.double()[None, :, None] * y1[:, None, :]
    src = torch.einsum("s,msd->md", read2.double(), new1)
    xr = rms_x(src.float(), src.square().sum(1).float(), D, norm_w, 1e-6)
    lat = xr.double() @ wd2.T
    check(latent, lat, rel=1e-3)
    y3 = latent.double() @ wd3.T  # from the kernel's own latent, so the last step is checked tightly
    new3 = torch.einsum("si,mid->msd", matrix2.double(), new1) + write2.double()[None, :, None] * y3[:, None, :]
    check(streams, new3, rel=1e-3)
    check(state, new3.mean(1), rel=1e-3)


# ---- Embedding and the depth memory -------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 8])
def test_embed(lx, M):
    torch.manual_seed(1)
    V, S, D, R = 1000, 4, 2048, 256
    table = torch.randn(V, D, device="cuda").to(torch.bfloat16)
    semb = torch.randn(S, D, device="cuda").to(torch.bfloat16)
    tokens = torch.randint(0, V, (M,), device="cuda", dtype=torch.int32)
    streams = torch.full((M, S, D), float("nan"), device="cuda")
    prev = torch.full((M, D), float("nan"), device="cuda")
    hist = torch.full((M, R), float("nan"), device="cuda")
    cont = torch.full((M,), float("nan"), device="cuda")
    prog = Program(lx, M, tokens=tokens)
    prog.add("embed", ptr(table), ptr(semb), ptr(streams), ptr(prev), ptr(hist), ptr(cont), S, D, R)
    prog.build()
    prog.run()
    want = table.float()[tokens.long()][:, None, :] + semb.float()[None]
    close(streams, want, atol=0)
    close(prev, want.mean(1), atol=1e-6)
    assert bool((hist == 0).all()) and bool((cont == 0).all())


def bank_entries(M, Emax, ne, D, Md):
    """A bank as the program reads it, from random keys and values and the read's projections:
    U = Wq^T k, P = Wout v + bout, c = bq . k. Returns the bank and what the original read needs."""
    Wq = torch.randn(Md, D, device="cuda", dtype=torch.float64) / math.sqrt(D)
    bq = torch.randn(Md, device="cuda", dtype=torch.float64) * 0.1
    Wout = torch.randn(D, Md, device="cuda", dtype=torch.float64) / math.sqrt(Md)
    bout = torch.randn(D, device="cuda", dtype=torch.float64) * 0.1
    k = torch.randn(M, Emax, Md, device="cuda", dtype=torch.float64) * 2
    v = torch.randn(M, Emax, Md, device="cuda", dtype=torch.float64)
    stride = 2 * D + 4
    bank = torch.full((M, Emax, stride), float("nan"), device="cuda", dtype=torch.float64)
    bank[:, :, :D] = k @ Wq
    bank[:, :, D : 2 * D] = v @ Wout.T + bout
    bank[:, :, 2 * D] = k @ bq
    return bank.float().contiguous(), (Wq, bq, Wout, bout, k, v)


@pytest.mark.parametrize("M,ne", [(1, 0), (1, 5), (3, 1), (8, 6)])
@pytest.mark.parametrize("end_of_pass", [False, True])
def test_memory_read(lx, M, ne, end_of_pass):
    torch.manual_seed(M * 10 + ne)
    S, D, Md, Emax, rank = 4, 2048, 256, 6, 8
    bank, (Wq, bq, Wout, bout, k, v) = bank_entries(M, Emax, ne, D, Md)
    streams = torch.randn(M, S, D, device="cuda")
    s0 = streams.double().clone()
    gate = (torch.randn(S, D, device="cuda") * 0.05).to(torch.bfloat16)
    gate_b = torch.randn(S, device="cuda").to(torch.bfloat16)
    read = torch.softmax(torch.randn(S, device="cuda"), 0)
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    lora_down = (torch.randn(rank, D, device="cuda") / math.sqrt(D)).to(torch.bfloat16)
    state, summary = torch.zeros(M, D, device="cuda"), torch.zeros(M, D, device="cuda")
    h = torch.zeros(M, D, device="cuda").to(torch.bfloat16)
    lora = torch.zeros(M, rank, device="cuda")
    prev = torch.randn(M, D, device="cuda")
    prev0 = prev.double().clone()
    diff = torch.zeros(M, D, device="cuda")
    rms = torch.zeros(2 * M, device="cuda")
    a = MemReadArgs()
    a.streams, a.bank, a.n_entries, a.entry_stride, a.token_stride, a.mem_dim = (
        streams.data_ptr(), bank.data_ptr(), ne, 2 * D + 4, Emax * (2 * D + 4), Md)
    a.gate, a.gate_bias, a.state, a.summary = gate.data_ptr(), gate_b.data_ptr(), state.data_ptr(), summary.data_ptr()
    a.read, a.norm_w, a.eps, a.h = read.data_ptr(), norm_w.data_ptr(), 1e-6, h.data_ptr()
    a.lora_down, a.rank, a.lora = lora_down.data_ptr(), rank, lora.data_ptr()
    a.rms = rms.data_ptr()
    if end_of_pass:
        a.previous, a.difference = prev.data_ptr(), diff.data_ptr()
    a.S, a.D = S, D
    prog = Program(lx, M)
    prog.add("memory_read", ctypes.byref(a))
    prog.build()
    prog.run()

    # the original read: per-stream query, attention over the entries, output projection, gated fuse
    x = s0
    if ne > 0:
        q = x @ Wq.T + bq                                            # [M, S, Md]
        scores = torch.einsum("msj,mej->mse", q, k[:, :ne]) / math.sqrt(Md)
        p = torch.softmax(scores, -1)
        retrieved = torch.einsum("mse,mej->msj", p, v[:, :ne])
        proj = retrieved @ Wout.T + bout
        g = torch.sigmoid((x * gate.double()).sum(-1) / math.sqrt(D) + gate_b.double())
        x = x + g[..., None] * proj
        want_summary = proj.mean(1)
    else:
        want_summary = torch.zeros(M, D, device="cuda", dtype=torch.float64)
    check(streams, x, rel=1e-5)
    check(summary, want_summary, rel=1e-5) if ne > 0 else close(summary, want_summary, atol=0)
    st = x.mean(1)
    check(state, st, rel=1e-5)
    close(rms[:M], st.square().sum(1), atol=0, rtol=1e-4)
    if end_of_pass:
        check(diff, st - prev0, rel=1e-5)
        check(prev, st, rel=1e-5)
        close(rms[M:], (st - prev0).square().sum(1), atol=0, rtol=1e-4)
    src = torch.einsum("s,msd->md", read.double(), x)
    hr = rms_x(src.float(), src.square().sum(1).float(), D, norm_w, 1e-6)
    close(h.float(), hr.float(), atol=0, rtol=2**-7)
    check(lora, bf(h.double() @ lora_down.double().T), rel=2**-7)


# ---- Mamba-2 ------------------------------------------------------------------------------------


def ssm_reference(zx, conv_w, conv_b, conv_state, ssm_state, A_log, Dp, dt_bias, slots, has, I, H, P, G, N, W):
    M = zx.shape[0]
    conv_dim = I + 2 * G * N
    zx, conv_w, conv_b = zx.double(), conv_w.double().view(conv_dim, W), conv_b.double()
    cs, ss = conv_state.double().clone(), ssm_state.double().clone()
    v = torch.zeros(M, I, dtype=torch.float64, device="cuda")
    for m in range(M):
        sl = slots[m]
        hist = cs[sl] if has[m] else torch.zeros(W - 1, conv_dim, dtype=torch.float64, device="cuda")
        xin = zx[m, I : I + conv_dim]
        window = torch.cat([hist, xin[None]], 0)                 # [W, conv_dim]
        out = F.silu(conv_b + (window.T * conv_w).sum(1))
        cs[sl] = window[1:]
        x, B, C = out[:I].view(H, P), out[I : I + G * N].view(G, N), out[I + G * N :].view(G, N)
        dt = F.softplus(zx[m, I + conv_dim : I + conv_dim + H] + dt_bias.double())
        dA = torch.exp(dt * -torch.exp(A_log.double()))
        g = torch.arange(H, device="cuda") // (H // G)
        S0 = ss[sl] if has[m] else torch.zeros(H, P, N, dtype=torch.float64, device="cuda")
        S1 = dA[:, None, None] * S0 + (dt[:, None] * x)[:, :, None] * B[g][:, None, :]
        y = (S1 * C[g][:, None, :]).sum(-1) + Dp.double()[:, None] * x
        ss[sl] = S1
        v[m] = (y * F.silu(zx[m, :I].view(H, P))).flatten()
    return v, cs, ss


@pytest.mark.parametrize("M,slots,has", [(1, [2], [1]), (1, [0], [0]), (4, [3, 0, 2, 1], [1, 0, 1, 1])])
def test_ssm(lx, M, slots, has):
    torch.manual_seed(M + sum(slots))
    I, H, P, G, N, W = 4096, 64, 64, 8, 128, 4
    conv_dim, proj = I + 2 * G * N, 2 * I + 2 * G * N + H
    n_slots = 4
    zx = torch.randn(M, proj, device="cuda")
    conv_w = (torch.randn(conv_dim, 1, W, device="cuda") * 0.5).to(torch.bfloat16)
    conv_b = (torch.randn(conv_dim, device="cuda") * 0.1).to(torch.bfloat16)
    conv_state = torch.randn(n_slots, W - 1, conv_dim, device="cuda")
    ssm_state = torch.randn(n_slots, H, P, N, device="cuda")
    A_log = torch.rand(H, device="cuda")
    Dp = torch.randn(H, device="cuda")
    dt_bias = torch.randn(H, device="cuda") * 0.5
    v = torch.zeros(M, I, device="cuda")
    head_ss = torch.zeros(M, H, device="cuda")
    want_v, want_cs, want_ss = ssm_reference(zx, conv_w, conv_b, conv_state, ssm_state, A_log, Dp, dt_bias, slots, has,
                                             I, H, P, G, N, W)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")
    a = SsmArgs(zx.data_ptr(), proj, conv_w.data_ptr(), conv_b.data_ptr(), conv_state.data_ptr(), ssm_state.data_ptr(),
                A_log.data_ptr(), Dp.data_ptr(), dt_bias.data_ptr(), v.data_ptr(), head_ss.data_ptr(), I, H, P, G, N, W)
    prog = Program(lx, M, slots=i32(slots), has_state=i32(has))
    prog.add("ssm", ctypes.byref(a))
    prog.build()
    prog.run()
    check(v, want_v, rel=1e-4)
    close(head_ss, want_v.view(M, H, P).square().sum(2), atol=0, rtol=1e-3)
    for m in range(M):
        check(conv_state[slots[m]], want_cs[slots[m]], rel=1e-6)
        check(ssm_state[slots[m]], want_ss[slots[m]], rel=1e-5)


# ---- Routing, experts ---------------------------------------------------------------------------


def route_reference(logits, bias, top_k):
    p = torch.softmax(logits.double(), -1)
    entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)
    confidence = p.max(-1).values
    ids = torch.topk(logits.double() + bias.double(), top_k, -1).indices
    w = torch.softmax(torch.gather(logits.double(), 1, ids), -1)
    return ids, w, entropy, confidence


@pytest.mark.parametrize("M", [1, 8])
def test_select(lx, M):
    torch.manual_seed(M)
    E, K, R = 128, 4, 256
    logits = torch.randn(M, E, device="cuda") * 2
    bias = torch.randn(E, device="cuda") * 0.1
    emb = torch.randn(E, R, device="cuda").to(torch.bfloat16)
    ids = torch.zeros(M, K, dtype=torch.int32, device="cuda")
    w = torch.zeros(M, K, device="cuda")
    ent, conf = torch.zeros(M, device="cuda"), torch.zeros(M, device="cuda")
    summary = torch.zeros(M, R, device="cuda")
    hist = torch.randn(M, R, device="cuda")
    h0 = hist.double().clone()
    a = SelectArgs(logits.data_ptr(), bias.data_ptr(), ids.data_ptr(), w.data_ptr(), ent.data_ptr(), conf.data_ptr(),
                   emb.data_ptr(), summary.data_ptr(), hist.data_ptr(), E, K, R)
    prog = Program(lx, M)
    prog.add("select", ctypes.byref(a))
    prog.build()
    prog.run()
    rid, rw, rent, rconf = route_reference(logits, bias, K)
    assert torch.equal(ids.long(), rid)
    check(w, rw, rel=1e-5)
    check(ent, rent, rel=1e-5)
    check(conf, rconf, rel=1e-5)
    want_sum = torch.einsum("mk,mkr->mr", rw, emb.double()[rid])
    check(summary, want_sum, rel=1e-5)
    check(hist, 0.5 * h0 + 0.5 * want_sum, rel=1e-5)


def expert_stack(E, rows, cols, kind):
    """A stacked expert weight [E, rows, cols] as lx_weight and dequantized float64."""
    w, wd, keep = weight(E * rows, cols, kind)
    return w, wd.view(E, rows, cols), keep


@pytest.mark.parametrize("M", [1, 3, 8])
@pytest.mark.parametrize("kind", ["int8", "bf16"])
def test_experts(lx, M, kind):
    """Routing then the experts, so the expert tiles wait on the routing op inside the program."""
    torch.manual_seed(M)
    E, K, L, I, Is, R = 128, 4, 1024, 512, 512, 256
    latent = torch.randn(M, L, device="cuda").to(torch.bfloat16)
    logits = torch.randn(M, E, device="cuda") * 2
    bias = torch.zeros(E, device="cuda")
    emb = torch.randn(E, R, device="cuda").to(torch.bfloat16)
    ids = torch.zeros(M, K, dtype=torch.int32, device="cuda")
    rw = torch.zeros(M, K, device="cuda")
    scratch = [torch.zeros(M, device="cuda") for _ in range(2)] + [torch.zeros(M, R, device="cuda") for _ in range(2)]
    gu, gud, k1 = expert_stack(E, 2 * I, L, kind)
    dn, dnd, k2 = expert_stack(E, L, I, kind)
    sgu, sgud, k3 = weight(2 * Is, L, kind)
    sdn, sdnd, k4 = weight(L, Is, kind)
    out = torch.zeros(M, L, device="cuda")
    prog = Program(lx, M)
    sel = SelectArgs(logits.data_ptr(), bias.data_ptr(), ids.data_ptr(), rw.data_ptr(), scratch[0].data_ptr(),
                     scratch[1].data_ptr(), emb.data_ptr(), scratch[2].data_ptr(), scratch[3].data_ptr(), E, K, R)
    prog.add("select", ctypes.byref(sel))
    a = ExpertsArgs(latent.data_ptr(), ids.data_ptr(), rw.data_ptr(), gu, dn, sgu, sdn, out.data_ptr(), L, I, Is, K)
    prog.add("experts", ctypes.byref(a))
    prog.build()
    for _ in range(2):
        prog.run()
        rid, rwt, _, _ = route_reference(logits, bias, K)
        x = latent.double()
        want = torch.zeros(M, L, dtype=torch.float64, device="cuda")
        for m in range(M):
            for j in range(K):
                e = rid[m, j]
                g = gud[e, :I] @ x[m]
                u = gud[e, I:] @ x[m]
                h = bf(F.silu(g) * u)
                want[m] += rwt[m, j] * (dnd[e] @ h)
        sg = x @ sgud[:Is].T
        su = x @ sgud[Is:].T
        hs = bf(F.silu(sg) * su)
        want += hs @ sdnd.T
        check(out, want, rel=2e-3)


# ---- Attention ----------------------------------------------------------------------------------


@pytest.mark.parametrize("M,start", [(1, [0]), (1, [700]), (3, [5, 0, 1300])])
def test_attention(lx, M, start):
    torch.manual_seed(M + sum(start))
    H, KVH, Dh, cap, n_slots = 32, 8, 64, 1536, 4
    slots = list(range(M))[::-1]
    qkv_w = (H + 2 * KVH) * Dh
    qkv = torch.randn(M, qkv_w, device="cuda")
    kc = torch.randn(n_slots, cap, KVH, Dh, device="cuda").to(torch.bfloat16)
    vc = torch.randn(n_slots, cap, KVH, Dh, device="cuda").to(torch.bfloat16)
    k0, v0 = kc.clone(), vc.clone()
    out = torch.zeros(M, H * Dh, device="cuda").to(torch.bfloat16)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(Dh)
    a = AttentionArgs(qkv.data_ptr(), qkv_w, kc.data_ptr(), vc.data_ptr(), cap, out.data_ptr(), H, KVH, Dh, scale)
    prog = Program(lx, M, slots=i32(slots), start_pos=i32(start))
    prog.add("attention", ctypes.byref(a))
    prog.build()
    prog.run()
    for m in range(M):
        s, p = slots[m], start[m]
        knew = qkv[m, H * Dh : (H + KVH) * Dh].view(KVH, Dh).to(torch.bfloat16)
        vnew = qkv[m, (H + KVH) * Dh :].view(KVH, Dh).to(torch.bfloat16)
        assert torch.equal(kc[s, p], knew) and torch.equal(vc[s, p], vnew)
        keys = torch.cat([k0[s, :p], knew[None]], 0).double()     # [L, KVH, Dh]
        vals = torch.cat([v0[s, :p], vnew[None]], 0).double()
        q = qkv[m, : H * Dh].view(KVH, H // KVH, Dh).double()
        sc = torch.einsum("kgd,lkd->kgl", q, keys) * scale
        o = torch.einsum("kgl,lkd->kgd", torch.softmax(sc, -1), vals).reshape(H * Dh)
        close(out[m].float(), o.float(), atol=2e-3, rtol=2**-7)


# ---- The memory entry and the depth head --------------------------------------------------------


@pytest.mark.parametrize("M", [1, 3, 8])
@pytest.mark.parametrize("kind", ["int8", "bf16"])
def test_memory_entry(lx, M, kind):
    torch.manual_seed(M)
    D, R, Md = 2048, 256, 256
    state = torch.randn(M, D, device="cuda")
    ws, wsd, keep = weight(Md, D, kind)
    bs = torch.randn(Md, device="cuda").to(torch.bfloat16)
    summary = torch.randn(M, R, device="cuda")
    ent, conf, cont = torch.rand(M, device="cuda") * 4, torch.rand(M, device="cuda"), torch.rand(M, device="cuda")
    wm = (torch.randn(Md, R + 4, device="cuda") / 16).to(torch.bfloat16)
    bm = torch.randn(Md, device="cuda").to(torch.bfloat16)
    pe, ae = torch.randn(Md, device="cuda").to(torch.bfloat16), torch.randn(Md, device="cuda").to(torch.bfloat16)
    entry = torch.zeros(M, Md, device="cuda")
    mean_k, inv_log_e = 0.5, 1.0 / math.log(128)
    a = EntryArgs(state.data_ptr(), ws, bs.data_ptr(), summary.data_ptr(), ent.data_ptr(), conf.data_ptr(),
                  cont.data_ptr(), mean_k, inv_log_e, wm.data_ptr(), bm.data_ptr(), pe.data_ptr(), ae.data_ptr(),
                  entry.data_ptr(), D, R, Md)
    prog = Program(lx, M)
    prog.add("memory_entry", ctypes.byref(a))
    prog.build()
    for _ in range(2):
        prog.run()
        meta = torch.cat([summary, torch.full((M, 1), mean_k, device="cuda"), (ent * inv_log_e)[:, None],
                          conf[:, None], cont[:, None]], 1)
        want = (state.double() @ wsd.T + bs.double()) + (bf(meta) @ wm.double().T + bm.double()) + pe.double() + ae.double()
        check(entry, want, rel=1e-4)


@pytest.mark.parametrize("M", [1, 8])
def test_continuation(lx, M):
    torch.manual_seed(M)
    R, D, limit = 256, 2048, 8.0
    hidden = torch.tanh(torch.randn(M, R, device="cuda"))
    w = torch.randn(1, R, device="cuda") * 0.1
    b = torch.randn(1, device="cuda")
    rms = torch.cat([torch.rand(M, device="cuda") * D, torch.rand(M, device="cuda") * D * 0.1])
    out = torch.zeros(M, device="cuda")
    prog = Program(lx, M)
    prog.add("continuation", ptr(hidden), ptr(w), ptr(b), ptr(rms), c_float(limit), ptr(out), R, D)
    prog.build()
    prog.run()
    logit = hidden.double() @ w.double()[0] + b.double()
    scale = torch.sqrt(rms[:M].double() / D).clamp_min(1e-6)
    innov = torch.sqrt(rms[M:].double() / D)
    want = torch.sigmoid((logit + torch.tanh(innov / scale) - 0.5).clamp(-limit, limit))
    check(out, want, rel=1e-5)


# ---- Many ops ---------------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 8])
def test_long_chain(lx, M):
    """Forty dependent products, each reading the one before: the weights stream ahead of the
    dependencies, and repeated runs must match (the counters reset between steps). The outputs shrink
    by four orders of magnitude down the chain, so the tensor cores' accumulation (less careful than a
    chain of FMAs) shows at 1e-4; 1e-3 still catches any real mistake."""
    torch.manual_seed(M)
    D = 1024
    ws = [weight(D, D, ["int8", "bf16"][i % 2]) for i in range(40)]
    bufs = [torch.randn(M, D, device="cuda")] + [torch.zeros(M, D, device="cuda") for _ in range(40)]
    prog = Program(lx, M)
    for i in range(40):
        prog.gemv(fx_plain(bufs[i]), ws[i][0], fy_store(bufs[i + 1], act=1), D, D)
    prog.build()
    want = bufs[0].double()
    for i in range(40):
        want = torch.tanh(want @ ws[i][1].T)
    for _ in range(3):
        prog.run()
        check(bufs[-1], want, rel=1e-3)


# ---- Batch invariance ---------------------------------------------------------------------------

INVARIANCE_CASES = [  # N, K, weight kind, x dtype
    (10304, 2048, "int8", torch.bfloat16),  # in_proj
    (2048, 4096, "bf16", torch.bfloat16),   # out_proj: x near the buffer's size at eight tokens
    (2048, 1024, "int8", torch.float32),    # latent_up: f32 x, read as a 16-bit pair
    (2048, 1024, "bf16", torch.float32),
    (256, 6400, "f32", torch.float32),      # K split into chunks
    (128, 1280, "f32", torch.float32),      # a router
    (4100, 256, "bf16", torch.bfloat16),
    (40, 72, "int8", torch.bfloat16),       # the FMA reader
]


@pytest.mark.parametrize("N,K,kind,xdt", INVARIANCE_CASES)
def test_rows_do_not_depend_on_m(lx, N, K, kind, xdt):
    """A token's output is the same to the bit whether it is decoded alone or among eight: speculative
    verification checks a draft against the bits a single step would give."""
    torch.manual_seed(N + K)
    w, _, keep = weight(N, K, kind)
    x8 = torch.randn(8, K, device="cuda").to(xdt)
    outs = {}
    for M, x in [(8, x8), (1, x8[5:6].clone())]:
        out = torch.zeros(M, N, device="cuda")
        prog = Program(lx, M)
        prog.gemv(fx_plain(x), w, fy_store(out), N, K)
        prog.build()
        prog.run()
        outs[M] = out
    assert torch.equal(outs[8][5], outs[1][0])


@pytest.mark.parametrize("kind", ["int8", "bf16"])
def test_chain_and_experts_do_not_depend_on_m(lx, kind):
    """A write, the RMS read of what it left, routing and the experts: token 3 among eight against
    token 3 alone, to the bit."""
    torch.manual_seed(17)
    S, D, I, E, K, L, Ie, R = 4, 2048, 4096, 128, 4, 1024, 512, 256
    matrix = torch.softmax(torch.randn(S, S, device="cuda"), 1).contiguous()
    write = torch.softmax(torch.randn(S, device="cuda"), 0)
    read2 = torch.softmax(torch.randn(S, device="cuda"), 0)
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    w1, _, k1 = weight(D, I, kind)
    w2, _, k2 = weight(L, D, kind)
    wr, _, k3 = weight(E, L, "f32")
    gu, _, k4 = expert_stack(E, 2 * Ie, L, kind)
    dn, _, k5 = expert_stack(E, L, Ie, kind)
    sgu, _, k6 = weight(2 * Ie, L, kind)
    sdn, _, k7 = weight(L, Ie, kind)
    bias = torch.zeros(E, device="cuda")
    emb = torch.randn(E, R, device="cuda").to(torch.bfloat16)
    v8 = torch.randn(8, I, device="cuda").to(torch.bfloat16)
    s8 = torch.randn(8, S, D, device="cuda")
    res = {}
    for M, sl in [(8, slice(0, 8)), (1, slice(3, 4))]:
        streams = s8[sl].clone()
        v = v8[sl].clone()
        next_src = torch.zeros(M, D, device="cuda")
        parts = torch.zeros(M, 4096, device="cuda")
        latent = torch.zeros(M, L, device="cuda")
        latent_bf = torch.zeros(M, L, device="cuda").to(torch.bfloat16)
        logits = torch.zeros(M, E, device="cuda")
        ids = torch.zeros(M, K, dtype=torch.int32, device="cuda")
        rw = torch.zeros(M, K, device="cuda")
        scratch = [torch.zeros(M, device="cuda") for _ in range(2)] + [torch.zeros(M, R, device="cuda") for _ in range(2)]
        out = torch.zeros(M, L, device="cuda")
        prog = Program(lx, M)
        fy = Fy()
        fy.kind, fy.streams, fy.matrix, fy.write, fy.S = FY_WRITE, streams.data_ptr(), matrix.data_ptr(), write.data_ptr(), S
        fy.next_read, fy.next_src, fy.next_parts = read2.data_ptr(), next_src.data_ptr(), parts.data_ptr()
        op1 = prog.gemv(fx_plain(v), w1, fy, D, I)
        fx = Fx()
        fx.kind, fx.x, fx.dtype, fx.ld = FX_RMS, next_src.data_ptr(), F32, D
        fx.parts, fx.n_parts, fx.norm_w, fx.eps = parts.data_ptr(), prog.tasks(op1), norm_w.data_ptr(), 1e-6
        prog.gemv(fx, w2, fy_store(latent, out_bf=latent_bf), L, D)
        prog.gemv(fx_plain(latent), wr, fy_store(logits), E, L)
        sel = SelectArgs(logits.data_ptr(), bias.data_ptr(), ids.data_ptr(), rw.data_ptr(), scratch[0].data_ptr(),
                         scratch[1].data_ptr(), emb.data_ptr(), scratch[2].data_ptr(), scratch[3].data_ptr(), E, K, R)
        prog.add("select", ctypes.byref(sel))
        a = ExpertsArgs(latent_bf.data_ptr(), ids.data_ptr(), rw.data_ptr(), gu, dn, sgu, sdn, out.data_ptr(), L, Ie,
                        Ie, K)
        prog.add("experts", ctypes.byref(a))
        prog.build()
        prog.run()
        res[M] = [t[3 if M == 8 else 0] for t in (streams, latent, logits, rw, out)]
    for a, b in zip(res[8], res[1]):
        assert torch.equal(a, b)
