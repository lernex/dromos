"""The offload kernels (engine/cuda/offload.cu) against PyTorch.

The expert kernels read each pair's expert through a pointer, so the tests place experts in device
memory and in pinned host memory (read across PCIe), mix both in one call, skip pairs, and check that
fills copy the exact bytes. The reference dequantizes the weights and computes the experts in float64,
rounding the SwiGLU intermediate to bf16 as the kernels do.

The cache tests build an lx_offload_state through ctypes and drive lx_offload_plan / lx_offload_join
with a Python thread standing in for the runtime's CPU worker, so the mailbox protocol (system-scope
fences, the doorbell, the done flag) is exercised end to end.
"""

import ctypes
import threading
import time

import pytest
import torch
import torch.nn.functional as F

from conftest import close

BF16, INT8, FP8 = 0, 1, 2
FORMATS = {"bf16": BF16, "int8": INT8, "fp8": FP8}
OG_BEGIN, OG_END = 1, 2
MAX_PAIRS = 64
OC = 8  # counters per layer
OC_HITS, OC_GPU, OC_CPU, OC_FILLS, OC_PREFETCHES, OC_PREFETCH_HITS, OC_CALLS = range(7)


class ExpertRef(ctypes.Structure):
    _fields_ = [("gate_up", ctypes.c_void_p), ("gate_up_scale", ctypes.c_void_p), ("down", ctypes.c_void_p),
                ("down_scale", ctypes.c_void_p)]


class Mailbox(ctypes.Structure):
    _fields_ = [("doorbell", ctypes.c_uint32), ("pad0", ctypes.c_uint32 * 15), ("done", ctypes.c_uint32),
                ("pad1", ctypes.c_uint32 * 15), ("layer", ctypes.c_int32), ("T", ctypes.c_int32), ("n", ctypes.c_int32),
                ("token", ctypes.c_int32 * MAX_PAIRS), ("expert", ctypes.c_int32 * MAX_PAIRS),
                ("weight", ctypes.c_float * MAX_PAIRS)]


class State(ctypes.Structure):
    _fields_ = [("n_layers", ctypes.c_int), ("n_experts", ctypes.c_int), ("top_k", ctypes.c_int), ("L", ctypes.c_int),
                ("I", ctypes.c_int), ("type", ctypes.c_int), ("vram_slot", ctypes.c_void_p),
                ("host_slot", ctypes.c_void_p), ("cpu_first", ctypes.c_void_p), ("layer_first", ctypes.c_void_p),
                ("layer_pinned", ctypes.c_void_p), ("slot_expert", ctypes.c_void_p), ("slot_stamp", ctypes.c_void_p),
                ("slot_prefetched", ctypes.c_void_p), ("clock", ctypes.c_void_p), ("vram_pool", ExpertRef),
                ("host_pool", ExpertRef), ("gate_up_stride", ctypes.c_int64), ("gate_up_scale_stride", ctypes.c_int64),
                ("down_stride", ctypes.c_int64), ("down_scale_stride", ctypes.c_int64), ("max_cpu_pairs", ctypes.c_int),
                ("cpu_fraction", ctypes.c_float), ("fill", ctypes.c_int), ("mailbox", ctypes.c_void_p),
                ("cpu_latent", ctypes.c_void_p), ("cpu_result", ctypes.c_void_p), ("job_seq", ctypes.c_void_p),
                ("job_pending", ctypes.c_void_p), ("error", ctypes.c_void_p), ("counters", ctypes.c_void_p)]


# ---- experts in every format ------------------------------------------------------------------


def quantize_rows(w, fmt):
    """w [..., rows, cols] f32 -> (data, scale f32 [..., rows] or None) in the format."""
    if fmt == BF16:
        return w.to(torch.bfloat16).contiguous(), None
    absmax = w.abs().amax(-1)
    if fmt == INT8:
        scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        q = torch.round(w / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.float().contiguous()
    scale = torch.where(absmax > 0, absmax / 448.0, torch.ones_like(absmax))
    q = (w / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.float().contiguous()


def dequant(data, scale):
    if scale is None:
        return data.double()
    return data.double() * scale.double().unsqueeze(-1)


class Experts:
    """E experts with gate_up [E, 2I, L] and down [E, L, I] in a format, on a device ('cuda' or pinned 'cpu')."""

    def __init__(self, E, L, I, fmt, seed):
        g = torch.Generator().manual_seed(seed)
        gu = torch.randn(E, 2 * I, L, generator=g) / L**0.5
        dn = torch.randn(E, L, I, generator=g) / I**0.5
        self.E, self.L, self.I, self.fmt = E, L, I, fmt
        self.gu, self.gus = quantize_rows(gu, fmt)
        self.dn, self.dns = quantize_rows(dn, fmt)

    def placed(self, where):
        """A copy in device memory ('cuda') or pinned host memory ('host')."""
        out = Experts.__new__(Experts)
        out.E, out.L, out.I, out.fmt = self.E, self.L, self.I, self.fmt
        for name in ("gu", "gus", "dn", "dns"):
            t = getattr(self, name)
            if t is not None:
                t = t.cuda() if where == "cuda" else t.cpu().pin_memory()
            setattr(out, name, t)
        return out

    def ref(self, e):
        def ptr(t, row_elems):
            if t is None:
                return None
            return t.data_ptr() + e * row_elems * t.element_size()

        I, L = self.I, self.L
        return (ptr(self.gu, 2 * I * L), ptr(self.gus, 2 * I), ptr(self.dn, L * I), ptr(self.dns, L))


def refs_tensor(refs):
    """A device array of lx_expert_ref from tuples of 4 addresses (None for null)."""
    flat = [0 if a is None else a for r in refs for a in r]
    return torch.tensor(flat, dtype=torch.int64, device="cuda") if flat else torch.zeros(4, dtype=torch.int64, device="cuda")


NULL_REF = (None, None, None, None)


def reference(latent, weights, experts, ids, skip=None):
    """float64 experts; pairs with skip[p] contribute nothing."""
    T, K = ids.shape
    I = experts.I
    x = latent.double().cpu()
    out = torch.zeros(T, experts.L, dtype=torch.float64)
    for t in range(T):
        for k in range(K):
            p = t * K + k
            if skip is not None and skip[p]:
                continue
            e = int(ids[t, k])
            gu = dequant(experts.gu[e].cpu(), None if experts.gus is None else experts.gus[e].cpu())
            dn = dequant(experts.dn[e].cpu(), None if experts.dns is None else experts.dns[e].cpu())
            mid = (F.silu(x[t] @ gu[:I].T) * (x[t] @ gu[I:].T)).to(torch.bfloat16).double()
            out[t] += float(weights[t, k]) * (mid @ dn.T)
    return out


def workspace(lx, T, L, I, K):
    fn = lx.lib.lx_offload_workspace_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int] * 4
    return torch.full((fn(T, L, I, K),), 255, dtype=torch.uint8, device="cuda")


def make_routing(T, E, K, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.rand(T, E, generator=g).argsort(-1)[:, :K].to(torch.int32)
    weights = torch.softmax(torch.randn(T, K, generator=g), -1)
    latent = torch.randn(T, 1, generator=g)  # placeholder, replaced per test
    return ids, weights


@pytest.mark.parametrize("fmt", ["bf16", "int8", "fp8"])
@pytest.mark.parametrize("T,L,I,K", [(1, 1024, 768, 4), (2, 1024, 512, 4), (8, 1024, 768, 4), (3, 256, 64, 8),
                                     (1, 3072, 768, 4), (4, 64, 2048, 2)])
def test_experts_decode_device_and_host(lx, fmt, T, L, I, K):
    torch.manual_seed(T * 7 + L + I + K)
    E = max(8, K + 2)
    base = Experts(E, L, I, FORMATS[fmt], seed=T + L + I)
    dev, host = base.placed("cuda"), base.placed("host")
    ids, weights = make_routing(T, E, K, seed=3 + T)
    latent = torch.randn(T, L).to(torch.bfloat16)
    want = reference(latent, weights, base, ids)
    scale = want.abs().max().item()
    for mode in ("device", "host", "mixed"):
        refs = []
        for p in range(T * K):
            e = int(ids.view(-1)[p])
            on_host = mode == "host" or (mode == "mixed" and p % 2 == 1)
            refs.append((host if on_host else dev).ref(e))
        src = refs_tensor(refs)
        out = torch.full((T, L), float("nan"), device="cuda")
        lx("offload_experts", latent.cuda(), weights.cuda().contiguous(), src, None, FORMATS[fmt], out,
           workspace(lx, T, L, I, K), T, L, I, K)
        close(out.cpu(), want.float(), atol=1e-3 * scale, rtol=1e-3)


@pytest.mark.parametrize("fmt", ["bf16", "int8", "fp8"])
def test_experts_decode_skip_and_fill(lx, fmt):
    T, L, I, K, E = 2, 1024, 768, 4, 12
    base = Experts(E, L, I, FORMATS[fmt], seed=11)
    host = base.placed("host")
    ids, weights = make_routing(T, E, K, seed=5)
    latent = torch.randn(T, L).to(torch.bfloat16)
    skip = [p % 3 == 2 for p in range(T * K)]
    # every non-skipped pair reads host memory; pairs 0 and 4 also fill a device slot
    slots = base.placed("cuda")
    for name in ("gu", "gus", "dn", "dns"):
        t = getattr(slots, name)
        if t is not None:
            t.zero_()
    refs, fills = [], []
    for p in range(T * K):
        e = int(ids.view(-1)[p])
        refs.append(NULL_REF if skip[p] else host.ref(e))
        fills.append(slots.ref(p) if (p in (0, 4) and not skip[p]) else NULL_REF)
    out = torch.full((T, L), float("nan"), device="cuda")
    lx("offload_experts", latent.cuda(), weights.cuda().contiguous(), refs_tensor(refs), refs_tensor(fills),
       FORMATS[fmt], out, workspace(lx, T, L, I, K), T, L, I, K)
    want = reference(latent, weights, base, ids, skip)
    close(out.cpu(), want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)
    for p in (0, 4):
        e = int(ids.view(-1)[p])
        assert torch.equal(slots.gu[p].cpu().view(torch.uint8), base.gu[e].view(torch.uint8))
        assert torch.equal(slots.dn[p].cpu().view(torch.uint8), base.dn[e].view(torch.uint8))
        if base.gus is not None:
            assert torch.equal(slots.gus[p].cpu(), base.gus[e])
            assert torch.equal(slots.dns[p].cpu(), base.dns[e])
    untouched = [s for s in range(E) if s not in (0, 4)]
    assert int(slots.gu[untouched].view(torch.uint8).count_nonzero()) == 0


@pytest.mark.parametrize("fmt", ["bf16", "int8", "fp8"])
@pytest.mark.parametrize("T", [1, 9, 64, 300])
def test_experts_grouped(lx, fmt, T):
    L, I, K, E = 1024, 768, 4, 24
    base = Experts(E, L, I, FORMATS[fmt], seed=T)
    dev, host = base.placed("cuda"), base.placed("host")
    ids, weights = make_routing(T, E, K, seed=T + 1)
    latent = torch.randn(T, L).to(torch.bfloat16)
    want = reference(latent, weights, base, ids)
    # groups in two batches (two calls), half from host memory
    flat = ids.view(-1).tolist()
    used = sorted(set(flat))
    groups, offsets, pairs = [], [0], []
    for g, e in enumerate(used):
        groups.append((host if g % 2 else dev).ref(e))
        pairs += [p for p, x in enumerate(flat) if x == e]
        offsets.append(len(pairs))
    groups_t = refs_tensor(groups)
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    pairs_t = torch.tensor(pairs, dtype=torch.int32, device="cuda")
    ws = workspace(lx, T, L, I, K)
    out = torch.full((T, L), float("nan"), device="cuda")
    half = len(used) // 2
    lx.lib.lx_offload_experts_grouped.restype = ctypes.c_int
    lx("offload_experts_grouped", latent.cuda(), weights.cuda().contiguous(), groups_t, offsets_t, pairs_t, half,
       FORMATS[fmt], out, ws, T, L, I, K, OG_BEGIN)
    rest_groups = groups_t[half * 4:].contiguous()
    rest_offsets = offsets_t[half:].contiguous()
    lx("offload_experts_grouped", latent.cuda(), weights.cuda().contiguous(), rest_groups, rest_offsets, pairs_t,
       len(used) - half, FORMATS[fmt], out, ws, T, L, I, K, OG_END)
    close(out.cpu(), want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)


@pytest.mark.parametrize("fmt", ["bf16", "fp8"])
def test_copy(lx, fmt):
    L, I, E = 1024, 768, 6
    base = Experts(E, L, I, FORMATS[fmt], seed=2)
    host = base.placed("host")
    dst = base.placed("cuda")
    for name in ("gu", "gus", "dn", "dns"):
        t = getattr(dst, name)
        if t is not None:
            t.zero_()
    src = refs_tensor([host.ref(4), host.ref(1), host.ref(0)])
    dsts = refs_tensor([dst.ref(0), dst.ref(1), dst.ref(5)])
    count = torch.tensor([2], dtype=torch.int32, device="cuda")  # only the first two
    lx("offload_copy", src, dsts, count, 3, FORMATS[fmt], L, I)
    for d, s in ((0, 4), (1, 1)):
        assert torch.equal(dst.gu[d].cpu().view(torch.uint8), base.gu[s].view(torch.uint8))
        assert torch.equal(dst.dn[d].cpu().view(torch.uint8), base.dn[s].view(torch.uint8))
        if base.gus is not None:
            assert torch.equal(dst.gus[d].cpu(), base.gus[s])
    assert int(dst.gu[5].view(torch.uint8).count_nonzero()) == 0


def test_read_probe(lx):
    buf = torch.ones(1 << 22, dtype=torch.uint8).pin_memory()
    sink = torch.zeros(1, dtype=torch.int64, device="cuda")
    lx("offload_read_probe", ctypes.c_void_p(buf.data_ptr()), buf.numel(), sink, 16)


# ---- the cache: plan, join, prefetch ----------------------------------------------------------


class Cache:
    """A two-layer cache over E experts per layer, built in device memory through ctypes."""

    def __init__(self, lx, fmt=BF16, E=16, K=4, L=64, I=32, slots=(5, 5), pinned=(2, 0), on_disk=()):
        self.lx, self.E, self.K, self.L, self.I, self.fmt = lx, E, K, L, I, fmt
        layers = 2
        self.layers = layers
        self.experts = [Experts(E, L, I, fmt, seed=40 + l) for l in range(layers)]
        # host pool: every expert, slot = l * E + e (one stacked copy, pinned)
        cat = lambda name: None if getattr(self.experts[0], name) is None else torch.cat(
            [getattr(x, name) for x in self.experts]).pin_memory()
        self.h_gu, self.h_gus, self.h_dn, self.h_dns = cat("gu"), cat("gus"), cat("dn"), cat("dns")
        n_slots = sum(slots)
        el = self.h_gu.element_size()
        self.v_gu = torch.zeros(n_slots, 2 * I, L, dtype=self.h_gu.dtype, device="cuda")
        self.v_dn = torch.zeros(n_slots, L, I, dtype=self.h_dn.dtype, device="cuda")
        self.v_gus = torch.zeros(n_slots, 2 * I, device="cuda") if self.h_gus is not None else None
        self.v_dns = torch.zeros(n_slots, L, device="cuda") if self.h_dns is not None else None
        first = [0, slots[0], n_slots]
        self.layer_first = torch.tensor(first, dtype=torch.int32, device="cuda")
        self.layer_pinned = torch.tensor(pinned, dtype=torch.int32, device="cuda")
        # initial contents: layer l's slots hold experts 0.. in order
        vram_slot = torch.full((layers * E,), -1, dtype=torch.int32)
        slot_expert = torch.full((n_slots,), -1, dtype=torch.int32)
        for l in range(layers):
            for j in range(slots[l]):
                s = first[l] + j
                vram_slot[l * E + j] = s
                slot_expert[s] = j
                self.v_gu[s] = self.experts[l].gu[j].cuda()
                self.v_dn[s] = self.experts[l].dn[j].cuda()
                if self.v_gus is not None:
                    self.v_gus[s] = self.experts[l].gus[j].cuda()
                    self.v_dns[s] = self.experts[l].dns[j].cuda()
        self.vram_slot = vram_slot.cuda()
        self.slot_expert = slot_expert.cuda()
        host_slot = torch.arange(layers * E, dtype=torch.int32)
        for l, e in on_disk:
            host_slot[l * E + e] = -1
        self.host_slot = host_slot.cuda()
        self.cpu_first = torch.zeros(layers * E, dtype=torch.uint8, device="cuda")
        self.slot_stamp = torch.zeros(n_slots, dtype=torch.int32, device="cuda")
        self.slot_prefetched = torch.zeros(n_slots, dtype=torch.uint8, device="cuda")
        self.clock = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.job_seq = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.job_pending = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.error = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.counters = torch.zeros(layers * OC, dtype=torch.int64, device="cuda")
        # mailbox and CPU buffers in pinned host memory (mapped: unified addressing)
        self.mb_buf = torch.zeros(ctypes.sizeof(Mailbox), dtype=torch.uint8).pin_memory()
        self.mailbox = Mailbox.from_address(self.mb_buf.data_ptr())
        self.cpu_latent = torch.zeros(8, L, dtype=torch.bfloat16).pin_memory()
        self.cpu_result = torch.zeros(8, L).pin_memory()

        st = State()
        st.n_layers, st.n_experts, st.top_k, st.L, st.I, st.type = layers, E, K, L, I, fmt
        st.vram_slot, st.host_slot, st.cpu_first = self.vram_slot.data_ptr(), self.host_slot.data_ptr(), self.cpu_first.data_ptr()
        st.layer_first, st.layer_pinned = self.layer_first.data_ptr(), self.layer_pinned.data_ptr()
        st.slot_expert, st.slot_stamp = self.slot_expert.data_ptr(), self.slot_stamp.data_ptr()
        st.slot_prefetched, st.clock = self.slot_prefetched.data_ptr(), self.clock.data_ptr()
        st.vram_pool = ExpertRef(self.v_gu.data_ptr(), self.v_gus.data_ptr() if self.v_gus is not None else None,
                                 self.v_dn.data_ptr(), self.v_dns.data_ptr() if self.v_dns is not None else None)
        st.host_pool = ExpertRef(self.h_gu.data_ptr(), self.h_gus.data_ptr() if self.h_gus is not None else None,
                                 self.h_dn.data_ptr(), self.h_dns.data_ptr() if self.h_dns is not None else None)
        st.gate_up_stride = 2 * I * L * el
        st.down_stride = L * I * el
        st.gate_up_scale_stride = 2 * I * 4
        st.down_scale_stride = L * 4
        st.max_cpu_pairs, st.cpu_fraction, st.fill = 0, 0.0, 1
        st.mailbox, st.cpu_latent, st.cpu_result = self.mb_buf.data_ptr(), self.cpu_latent.data_ptr(), self.cpu_result.data_ptr()
        st.job_seq, st.job_pending, st.error = self.job_seq.data_ptr(), self.job_pending.data_ptr(), self.error.data_ptr()
        st.counters = self.counters.data_ptr()
        self.state_host = st
        self.state = torch.zeros(ctypes.sizeof(State), dtype=torch.uint8, device="cuda")
        self.upload()
        self.src = torch.zeros(MAX_PAIRS * 4, dtype=torch.int64, device="cuda")
        self.fill = torch.zeros(MAX_PAIRS * 4, dtype=torch.int64, device="cuda")
        self.served = []

    def upload(self):
        raw = bytes(self.state_host)
        self.state.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))

    def set_policy(self, cpu_fraction=0.0, max_cpu=0, fill=1):
        self.state_host.cpu_fraction, self.state_host.max_cpu_pairs, self.state_host.fill = cpu_fraction, max_cpu, fill
        self.upload()

    def cpu_worker(self, stop):
        """Serves the mailbox like the runtime's CPU worker, in float64 then f32."""
        seen = 0
        while not stop.is_set():
            bell = self.mailbox.doorbell
            if bell == seen:
                time.sleep(0.0002)
                continue
            seen = bell
            mb = self.mailbox
            T, L, I = mb.T, self.L, self.I
            x = self.cpu_latent[:T].double()
            res = torch.zeros(T, L, dtype=torch.float64)
            ex = self.experts[mb.layer]
            for j in range(mb.n):
                t, e, w = mb.token[j], mb.expert[j], mb.weight[j]
                gu = dequant(ex.gu[e], ex.gus[e] if ex.gus is not None else None)
                dn = dequant(ex.dn[e], ex.dns[e] if ex.dns is not None else None)
                mid = (F.silu(x[t] @ gu[:I].T) * (x[t] @ gu[I:].T)).to(torch.bfloat16).double()
                res[t] += w * (mid @ dn.T)
                self.served.append((mb.layer, e))
            self.cpu_result[:T] = res.float()
            ctypes.c_uint32.from_address(self.mb_buf.data_ptr() + Mailbox.done.offset).value = bell

    def step(self, layer, ids, weights, latent):
        T = ids.shape[0]
        lx = self.lx
        ids_d, w_d, lat_d = ids.cuda().contiguous(), weights.cuda().contiguous(), latent.cuda().contiguous()
        lx("offload_plan", ids_d, w_d, lat_d, layer, self.state, self.src, self.fill, T)
        src = self.src[: T * self.K * 4].view(T * self.K, 4).cpu()
        fill = self.fill[: T * self.K * 4].view(T * self.K, 4).cpu()
        out = torch.full((T, self.L), float("nan"), device="cuda")
        lx("offload_experts", lat_d, w_d, self.src, self.fill, self.fmt, out, workspace(lx, T, self.L, self.I, self.K),
           T, self.L, self.I, self.K)
        lx("offload_join", self.state, out, T, self.L)
        return out.cpu(), src, fill

    def counter(self, layer, i):
        return int(self.counters[layer * OC + i])

    def vram_expert_of(self, layer, slot_range=None):
        vs = self.vram_slot.cpu().view(self.layers, self.E)[layer]
        return {int(e): int(s) for e, s in enumerate(vs) if s >= 0}


def classify(src, cache):
    """Per pair: 'vram', 'host' or 'skip' from the address it reads."""
    v0, v1 = cache.v_gu.data_ptr(), cache.v_gu.data_ptr() + cache.v_gu.numel() * cache.v_gu.element_size()
    out = []
    for r in src.tolist():
        a = r[0]
        out.append("skip" if a == 0 else ("vram" if v0 <= a < v1 else "host"))
    return out


def test_plan_hits_misses_and_fills(lx):
    c = Cache(lx)
    K, E, L = c.K, c.E, c.L
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, L, generator=g).to(torch.bfloat16)
    weights = torch.softmax(torch.randn(1, K, generator=g), -1)
    # layer 0 holds experts 0..4 (0, 1 pinned): 1 and 3 hit, 7 and 9 miss
    ids = torch.tensor([[1, 7, 3, 9]], dtype=torch.int32)
    out, src, fill = c.step(0, ids, weights, latent)
    assert classify(src, c) == ["vram", "host", "vram", "host"]
    want = reference(latent, weights, c.experts[0], ids)
    close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)
    # the misses were written into the two least recently used dynamic slots (2 and 4: 3 was just used)
    resident = c.vram_expert_of(0)
    assert resident[7] in (2, 4) and resident[9] in (2, 4) and resident[1] == 1 and resident[3] == 3
    assert 2 not in resident and 4 not in resident
    s7 = resident[7]
    assert torch.equal(c.v_gu[s7].cpu().view(torch.uint8), c.experts[0].gu[7].view(torch.uint8))
    assert c.counter(0, OC_HITS) == 2 and c.counter(0, OC_GPU) == 2 and c.counter(0, OC_FILLS) == 2
    # the same experts again: all hits now
    out2, src2, _ = c.step(0, ids, weights, latent)
    assert classify(src2, c) == ["vram"] * 4
    assert torch.equal(out2, out)  # the same bytes, the same kernel: bit-identical
    # layer 1 was untouched
    assert c.vram_expert_of(1) == {e: 5 + e for e in range(5)}


def test_plan_pinned_never_evicted_and_lru(lx):
    c = Cache(lx)
    K, L = c.K, c.L
    latent = torch.randn(1, L).to(torch.bfloat16)
    weights = torch.full((1, K), 0.25)
    rows = [[8, 9, 15, 0], [10, 11, 14, 0], [12, 13, 2, 1], [5, 6, 7, 1], [8, 2, 3, 4], [14, 15, 9, 0]]
    for row in rows:
        ids = torch.tensor([row], dtype=torch.int32)
        c.step(0, ids, weights, latent)
        resident = c.vram_expert_of(0)
        assert resident[0] == 0 and resident[1] == 1  # pinned
        assert len(resident) == 5


def test_plan_no_slot_free_reads_without_fill(lx):
    # layer 1 has 5 dynamic slots; a step of 2 tokens x 4 distinct misses needs 8: 5 fill, 3 do not
    c = Cache(lx)
    L = c.L
    ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)
    latent = torch.randn(2, L).to(torch.bfloat16)
    weights = torch.softmax(torch.randn(2, 4), -1)
    out, src, fill = c.step(1, ids, weights, latent)
    assert classify(src, c) == ["host"] * 8
    assert sum(1 for r in fill.tolist() if r[0] != 0) == 5
    want = reference(latent, weights, c.experts[1], ids)
    close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)


def test_plan_shared_miss_fills_once(lx):
    c = Cache(lx)
    L = c.L
    ids = torch.tensor([[10, 1, 2, 0], [0, 10, 1, 2]], dtype=torch.int32)  # slots 3 and 4 are free
    latent = torch.randn(2, L).to(torch.bfloat16)
    weights = torch.softmax(torch.randn(2, 4), -1)
    out, src, fill = c.step(0, ids, weights, latent)
    fills = [r[0] != 0 for r in fill.tolist()]
    assert fills == [True] + [False] * 7  # pair 0 fills expert 10; pair 5 reads it from host
    want = reference(latent, weights, c.experts[0], ids)
    close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)


@pytest.mark.parametrize("fmt", [BF16, FP8])
def test_cpu_lane_through_the_mailbox(lx, fmt):
    c = Cache(lx, fmt=fmt)
    c.set_policy(cpu_fraction=0.5, max_cpu=4)
    c.cpu_first[0 * c.E + 9] = 1  # a cold expert: the CPU takes it first
    stop = threading.Event()
    worker = threading.Thread(target=c.cpu_worker, args=(stop,), daemon=True)
    worker.start()
    try:
        L = c.L
        g = torch.Generator().manual_seed(7)
        for step in range(3):
            ids = torch.tensor([[1, 7, 9, 11], [9, 12, 0, 3]], dtype=torch.int32)
            latent = torch.randn(2, L, generator=g).to(torch.bfloat16)
            weights = torch.softmax(torch.randn(2, 4, generator=g), -1)
            out, src, fill = c.step(0, ids, weights, latent)
            want = reference(latent, weights, c.experts[0], ids)
            close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)
            if step == 0:
                kinds = classify(src, c)
                # misses 7, 9, 11, 12 (distinct); the CPU takes half: 9 (cold) and one more
                assert kinds[2] == "skip" and kinds[4] == "skip"
                assert kinds.count("skip") == 3  # 9 twice, plus one other miss
        assert int(c.error) == 0
        assert c.counter(0, OC_CPU) > 0
        assert (0, 9) in c.served
    finally:
        stop.set()
        worker.join()


def test_disk_only_experts_go_to_the_cpu(lx):
    c = Cache(lx, on_disk=[(1, 13)])
    c.set_policy(cpu_fraction=0.0, max_cpu=0)
    stop = threading.Event()
    worker = threading.Thread(target=c.cpu_worker, args=(stop,), daemon=True)
    worker.start()
    try:
        ids = torch.tensor([[13, 6, 2, 3]], dtype=torch.int32)
        latent = torch.randn(1, c.L).to(torch.bfloat16)
        weights = torch.softmax(torch.randn(1, 4), -1)
        out, src, _ = c.step(1, ids, weights, latent)
        assert classify(src, c) == ["skip", "host", "vram", "vram"]
        want = reference(latent, weights, c.experts[1], ids)
        close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)
        assert 13 not in c.vram_expert_of(1)
    finally:
        stop.set()
        worker.join()


def test_join_without_job_is_a_no_op(lx):
    c = Cache(lx)
    out = torch.ones(1, c.L, device="cuda")
    lx("offload_join", c.state, out, 1, c.L)
    assert bool((out == 1).all())


def test_prefetch_then_hit(lx):
    c = Cache(lx)
    L, K = c.L, c.K
    predicted = torch.tensor([12, 2, 14], dtype=torch.int32, device="cuda")  # 2 is resident
    psrc = torch.zeros(16 * 4, dtype=torch.int64, device="cuda")
    pdst = torch.zeros(16 * 4, dtype=torch.int64, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    lx("offload_prefetch", predicted, 3, 0, c.state, psrc, pdst, count)
    assert int(count) == 2
    lx("offload_copy", psrc, pdst, count, 16, c.fmt, L, c.I)
    resident = c.vram_expert_of(0)
    assert 12 in resident and 14 in resident and 2 in resident and resident[0] == 0 and resident[1] == 1
    assert torch.equal(c.v_gu[resident[12]].cpu().view(torch.uint8), c.experts[0].gu[12].view(torch.uint8))
    ids = torch.tensor([[12, 14, 2, 0]], dtype=torch.int32)
    latent = torch.randn(1, L).to(torch.bfloat16)
    weights = torch.softmax(torch.randn(1, 4), -1)
    out, src, _ = c.step(0, ids, weights, latent)
    assert classify(src, c) == ["vram"] * 4
    assert c.counter(0, OC_PREFETCH_HITS) == 2 and c.counter(0, OC_PREFETCHES) == 2
    want = reference(latent, weights, c.experts[0], ids)
    close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)
