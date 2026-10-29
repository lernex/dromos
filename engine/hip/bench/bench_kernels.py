"""Time the decode-path kernels of one library on the current GPU, with CUDA/HIP events.

    $PY engine/hip/bench/bench_kernels.py path/to/libdromos_kernels.so [iterations]

Each row is the median over the iterations after a warm-up, in microseconds. The shapes are decode
shapes (one to eight tokens) and one long-context attention step. Numbers mean something only on an
idle GPU: the script prints the device so a run on a shared card is not mistaken for a benchmark.
"""

import ctypes
import math
import statistics
import sys

import torch

lib = ctypes.CDLL(sys.argv[1])
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
LX_F32, LX_BF16 = 0, 1
LX_W_INT8 = 1


def bind(name, argtypes):
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


def ptr(t):
    return vp(t.data_ptr()) if t is not None else vp(None)


gemv = bind("lx_gemv", [vp, ci, ci, vp, ci, vp, vp, ci, vp, ci, ci, ci, ci, ci, cf, cf, vp])
quantize = bind("lx_quantize_w8", [vp, vp, vp, ctypes.c_int64, ci, vp])
moe_w8 = bind("lx_moe_experts_w8", [vp, vp, vp, vp, vp, vp, vp, vp, vp, ci, ci, ci, ci, vp])
moe_ws = lib.lx_moe_workspace_bytes
moe_ws.argtypes = [ci, ci, ci]
moe_ws.restype = ctypes.c_int64
attention = bind("lx_attention_varlen", [vp, ci, vp, vp, vp, vp, vp, ci, ci, vp, ci, ci, ci, cf, vp])
fuse = bind("lx_memory_fuse", [vp, vp, vp, vp, vp, ci, ci, ci, vp])
mhc = bind("lx_mhc_read", [vp, vp, vp, vp, cf, vp, vp, ci, ci, ci, vp])
memory_attend = bind("lx_memory_attend", [vp, vp, vp, ci, ci, vp, ci, ci, ci, vp])


def timed(label, fn, rows):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    med = statistics.median(samples)
    rows.append((label, med, min(samples)))


def check(rc):
    assert rc == 0, f"library returned {rc}"


rows = []
device = torch.cuda.get_device_name(0)
print(f"library {sys.argv[1]}\ndevice  {device}\niterations {ITERS}\n")

# int8 GEMV, the decode projection: N x K weight, M tokens.
for N, K in [(6144, 2048), (2048, 6144)]:
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    q = torch.empty(N, K, device="cuda", dtype=torch.int8)
    s = torch.empty(N, device="cuda")
    check(quantize(ptr(w), ptr(q), ptr(s), N, K, None))
    for M in (1, 8):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        y = torch.empty(M, N, device="cuda")
        timed(f"gemv int8 N{N} K{K} M{M}",
              lambda: check(gemv(ptr(x), LX_BF16, K, ptr(q), LX_W_INT8, ptr(s), None, LX_BF16, ptr(y), LX_F32, N, M,
                                 N, K, 1.0, 0.0, None)),
              rows)

# int8 MoE, decode: 8 experts, top-2.
E, top_k, L, I = 8, 2, 2048, 512
torch.manual_seed(0)
gu = (torch.randn(E * 2 * I, L, device="cuda") * 0.05).to(torch.bfloat16)
dn = (torch.randn(E * L, I, device="cuda") * 0.05).to(torch.bfloat16)
gu_q = torch.empty(E * 2 * I, L, device="cuda", dtype=torch.int8)
gu_s = torch.empty(E * 2 * I, device="cuda")
dn_q = torch.empty(E * L, I, device="cuda", dtype=torch.int8)
dn_s = torch.empty(E * L, device="cuda")
check(quantize(ptr(gu), ptr(gu_q), ptr(gu_s), E * 2 * I, L, None))
check(quantize(ptr(dn), ptr(dn_q), ptr(dn_s), E * L, I, None))
for T in (1, 8):
    latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]).to(torch.int32)
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), dim=-1)
    ws = torch.empty(max(1, moe_ws(T, I, top_k)), device="cuda", dtype=torch.uint8)
    out = torch.empty(T, L, device="cuda")
    timed(f"moe int8 T{T} L{L} I{I} E{E} top{top_k}",
          lambda: check(moe_w8(ptr(latent), ptr(ids), ptr(weights), ptr(gu_q.view(E, 2 * I, L)), ptr(gu_s.view(E, 2 * I)),
                               ptr(dn_q.view(E, L, I)), ptr(dn_s.view(E, L)), ptr(out), ptr(ws), T, L, I, top_k, None)),
          rows)

# Attention, decode over a long cache, one token per sequence at position 3000. One sequence is the
# few-items case (the CUDA library splits each item across a cluster there); eight is the batched case.
H, KVH, Dh, capacity = 16, 8, 128, 4096
ld = (H + 2 * KVH) * Dh
for n_seq in (1, 8):
    torch.manual_seed(0)
    qkv = torch.randn(n_seq, ld, device="cuda")
    kc = (torch.randn(n_seq, capacity, KVH, Dh, device="cuda") * 0.5).to(torch.bfloat16)
    vc = (torch.randn(n_seq, capacity, KVH, Dh, device="cuda") * 0.5).to(torch.bfloat16)
    slots = torch.arange(n_seq, dtype=torch.int32, device="cuda")
    starts = torch.full((n_seq,), 3000, dtype=torch.int32, device="cuda")
    cu = torch.arange(n_seq + 1, dtype=torch.int32, device="cuda")
    out_att = torch.empty(n_seq, H * Dh, device="cuda", dtype=torch.bfloat16)
    timed(f"attention decode {n_seq} seq @3000",
          lambda: check(attention(ptr(qkv), ld, ptr(kc), ptr(vc), ptr(slots), ptr(starts), ptr(cu), n_seq, capacity,
                                  ptr(out_att), H, KVH, Dh, 1.0 / math.sqrt(Dh), None)),
          rows)

# Depth memory and hyper-connections, decode: 8 tokens.
T, S, D, M, Ebank = 8, 4, 2048, 256, 8
torch.manual_seed(0)
streams = torch.randn(T, S, D, device="cuda")
projected = torch.randn(T, S, D, device="cuda") * 0.5
gate = torch.randn(S, D, device="cuda").to(torch.bfloat16)
gate_bias = torch.randn(S, device="cuda").to(torch.bfloat16)
summary = torch.empty(T, D, device="cuda")
timed(f"memory fuse T{T} S{S} D{D}",
      lambda: check(fuse(ptr(streams), ptr(projected), ptr(gate), ptr(gate_bias), ptr(summary), T, S, D, None)),
      rows)

Dm = 2560
mstreams = torch.randn(T, S, Dm, device="cuda")
matrix = torch.randn(S, S, device="cuda")
read = torch.randn(S, device="cuda")
norm_w = (torch.rand(Dm, device="cuda") + 0.5).to(torch.bfloat16)
mixed = torch.empty(T, S, Dm, device="cuda")
h = torch.empty(T, Dm, device="cuda", dtype=torch.bfloat16)
timed(f"mhc read T{T} S{S} D{Dm}",
      lambda: check(mhc(ptr(mstreams), ptr(matrix), ptr(read), ptr(norm_w), 1e-6, ptr(mixed), ptr(h), T, S, Dm, None)),
      rows)

bank_q = torch.randn(T, S, M, device="cuda")
bank_k = torch.randn(T, Ebank, M, device="cuda")
bank_v = torch.randn(T, Ebank, M, device="cuda")
out_mem = torch.empty(T, S, M, device="cuda", dtype=torch.bfloat16)
timed(f"memory attend T{T} S{S} M{M} E{Ebank}",
      lambda: check(memory_attend(ptr(bank_q), ptr(bank_k), ptr(bank_v), Ebank, Ebank, ptr(out_mem), T, S, M, None)),
      rows)

print(f"{'kernel':40s} {'median us':>12s} {'min us':>12s}")
for label, med, low in rows:
    print(f"{label:40s} {med:12.1f} {low:12.1f}")
