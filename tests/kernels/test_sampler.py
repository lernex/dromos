"""The batched sampler (engine/cuda/sampler.cu) against PyTorch.

Greedy picks, penalties, top-p / top-k / min-p membership and the log-probabilities are checked
against float64 PyTorch references. The draw itself is checked on its distribution: the empirical
frequencies over many positions match the nucleus each request defines, and a seed reproduces its
tokens wherever its row sits in the batch.
"""

import ctypes

import pytest
import torch

from conftest import close


class Sampling(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64),
        ("position", ctypes.c_uint64),
        ("slot", ctypes.c_int32),
        ("row", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("min_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
    ]


def make_rows(specs):
    """specs: dicts of Sampling fields. Returns the struct array as a device byte tensor."""
    arr = (Sampling * len(specs))()
    for i, spec in enumerate(specs):
        values = {
            "seed": 1,
            "position": 0,
            "slot": 0,
            "row": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        values.update(spec)
        for name, value in values.items():
            setattr(arr[i], name, value)
    raw = bytes(arr)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()


class Sampler:
    """Runs lx_sample on device buffers for a fixed vocabulary and slot count."""

    def __init__(self, lx, vocab, slots):
        self.lx = lx
        self.V = vocab
        self.seen = torch.zeros(slots, vocab, dtype=torch.int32, device="cuda")
        self.out = torch.zeros(slots, vocab, dtype=torch.int32, device="cuda")

    def reset(self, slot):
        self.lx("sampler_reset", self.seen, self.out, slot, self.V)

    def observe(self, slot, tokens, is_output):
        t = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        self.lx("sampler_observe", self.seen, self.out, slot, self.V, t, len(tokens), int(is_output))

    def sample(self, logits, specs, top_n=0):
        n = len(specs)
        rows = make_rows(specs)
        work = torch.empty(n, self.V, dtype=torch.float32, device="cuda")
        tokens = torch.empty(n, dtype=torch.int32, device="cuda")
        lp = torch.empty(n, dtype=torch.float32, device="cuda")
        top_ids = torch.empty(max(n * top_n, 1), dtype=torch.int32, device="cuda")
        top_lp = torch.empty(max(n * top_n, 1), dtype=torch.float32, device="cuda")
        self.lx(
            "sample",
            logits,
            self.V,
            rows,
            n,
            self.seen,
            self.out,
            work,
            tokens,
            lp,
            top_n,
            top_ids,
            top_lp,
        )
        return (
            tokens.cpu().tolist(),
            lp.cpu(),
            top_ids[: n * top_n].view(n, top_n).cpu(),
            top_lp[: n * top_n].view(n, top_n).cpu(),
        )


def ref_penalized(logits, seen, out, rep, pres, freq):
    x = logits.double()
    seen_mask = seen > 0
    scaled = torch.where(x > 0, x / rep, x * rep)
    x = torch.where(seen_mask & (rep != 1.0), scaled, x)
    count = out.double()
    x = x - (count > 0).double() * (freq * count + pres)
    return x


def nucleus(scaled, top_k=0, top_p=1.0, min_p=0.0):
    """The set of token ids a request keeps, in float64: top-k, then top-p of what remains, then min-p."""
    s = scaled.double()
    keep = torch.ones_like(s, dtype=torch.bool)
    if 0 < top_k < s.numel():
        kth = torch.topk(s, top_k).values[-1]
        keep &= s >= kth
    if 0.0 < top_p < 1.0:
        masked = torch.where(keep, s, torch.full_like(s, float("-inf")))
        probs = torch.softmax(masked, dim=0)
        order = torch.argsort(probs, descending=True)
        cum_before = torch.cumsum(probs[order], 0) - probs[order]
        kept_sorted = order[cum_before < top_p]
        keep2 = torch.zeros_like(keep)
        keep2[kept_sorted] = True
        keep &= keep2
    if min_p > 0.0:
        probs = torch.softmax(s, dim=0)
        keep &= probs >= min_p * probs.max()
    return keep


V = 65536


@pytest.mark.parametrize("vocab", [V, 1000])
def test_greedy_with_penalties(lx, vocab):
    torch.manual_seed(7)
    n, slots = 6, 6
    logits = (torch.randn(n, vocab, device="cuda") * 3).float()
    sampler = Sampler(lx, vocab, slots)
    specs, expected = [], []
    for i in range(n):
        sampler.reset(i)
        prompt = torch.randint(0, vocab, (40,)).tolist()
        output = torch.randint(0, vocab, (25,)).tolist()
        sampler.observe(i, prompt, is_output=False)
        sampler.observe(i, output, is_output=True)
        spec = dict(
            slot=i,
            row=i,
            temperature=0.0,
            repetition_penalty=1.0 + 0.1 * i,
            presence_penalty=0.2 * (i % 3),
            frequency_penalty=0.05 * i,
        )
        specs.append(spec)
        # the expected pick, from the counts before this step's draw
        x = ref_penalized(logits[i], sampler.seen[i], sampler.out[i], spec["repetition_penalty"],
                          spec["presence_penalty"], spec["frequency_penalty"])
        expected.append(int(torch.argmax(x)))
    tokens, lp, _, _ = sampler.sample(logits, specs)
    for i in range(n):
        assert tokens[i] == expected[i], f"row {i}"
        want = torch.log_softmax(logits[i].double(), 0)[tokens[i]].item()
        assert abs(lp[i].item() - want) < 1e-4


def test_sampled_tokens_respect_top_k_and_top_p(lx):
    torch.manual_seed(11)
    logits = (torch.randn(1, V, device="cuda") * 2.5).float()
    sampler = Sampler(lx, V, 1)
    sampler.reset(0)
    for top_k, top_p in [(20, 1.0), (0, 0.5), (50, 0.9), (5, 0.99)]:
        allowed = nucleus(logits[0], top_k=top_k, top_p=top_p)
        ids = []
        for launch in range(4):
            specs = [dict(slot=0, row=0, position=launch * 64 + j, temperature=0.8, top_k=top_k, top_p=top_p,
                          seed=12345) for j in range(64)]
            tokens, _, _, _ = sampler.sample(logits, specs)
            ids += tokens
        bad = [t for t in ids if not bool(allowed[t])]
        assert not bad, f"top_k={top_k} top_p={top_p}: tokens outside the nucleus {bad[:5]}"


def test_min_p_membership(lx):
    torch.manual_seed(5)
    logits = (torch.randn(1, V, device="cuda") * 2.0).float()
    sampler = Sampler(lx, V, 1)
    sampler.reset(0)
    allowed = nucleus(logits[0], min_p=0.05)
    specs = [dict(slot=0, row=0, position=j, temperature=1.0, min_p=0.05, seed=9) for j in range(200)]
    tokens, _, _, _ = sampler.sample(logits, specs)
    assert all(bool(allowed[t]) for t in tokens)


def test_empirical_distribution_matches_nucleus(lx):
    """Draws at 4000 positions follow softmax over the kept set (top-k 12, top-p 0.9, min-p 0.01)."""
    torch.manual_seed(3)
    base = torch.randn(V, device="cuda")
    base[:8] += torch.tensor([6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5], device="cuda")
    logits = base.float().view(1, V)
    sampler = Sampler(lx, V, 1)
    sampler.reset(0)
    keep = nucleus(logits[0], top_k=12, top_p=0.9, min_p=0.01)
    probs = torch.softmax(torch.where(keep.cuda(), logits[0].double(), torch.tensor(float("-inf"), device="cuda", dtype=torch.float64)), 0)
    draws = []
    for launch in range(8):
        specs = [dict(slot=0, row=0, position=launch * 500 + j, temperature=1.0, top_k=12, top_p=0.9,
                      min_p=0.01, seed=777) for j in range(500)]
        tokens, _, _, _ = sampler.sample(logits, specs)
        draws += tokens
    counts = torch.bincount(torch.tensor(draws), minlength=V).double()
    freq = counts / counts.sum()
    assert (freq[~keep.cpu()] == 0).all()
    worst = (freq - probs.cpu()).abs().max().item()
    assert worst < 0.035, f"empirical frequency off by {worst:.3f}"


def test_seed_reproduces_wherever_the_row_sits(lx):
    """The same request (seed, position, logits, penalties) draws the same token at any batch index."""
    torch.manual_seed(2)
    logits = (torch.randn(4, V, device="cuda") * 2).float()
    sampler = Sampler(lx, V, 4)
    target = dict(slot=0, row=2, position=5, seed=42, top_p=0.95, top_k=40)
    filler = [dict(slot=s, row=s, position=s, seed=7 + s) for s in (1, 3, 2)]
    sampler.reset(0)
    first, _, _, _ = sampler.sample(logits, [target])
    sampler.reset(0)
    second, _, _, _ = sampler.sample(logits, filler + [target])
    assert first == second[3:], "same seed and position must draw the same token"
    # different seeds draw different tokens over many positions of a flat distribution
    flat = torch.zeros(1, V, device="cuda")
    sampler.reset(0)
    a, _, _, _ = sampler.sample(flat, [dict(slot=0, row=0, position=j, seed=1, temperature=1.0) for j in range(64)])
    sampler.reset(0)
    b, _, _, _ = sampler.sample(flat, [dict(slot=0, row=0, position=j, seed=2, temperature=1.0) for j in range(64)])
    assert a != b


def test_top_n_logprobs_match_log_softmax(lx):
    torch.manual_seed(8)
    logits = (torch.randn(3, V, device="cuda") * 3).float()
    sampler = Sampler(lx, V, 3)
    for s in range(3):
        sampler.reset(s)
    specs = [dict(slot=s, row=s, temperature=0.0) for s in range(3)]
    tokens, lp, top_ids, top_lp = sampler.sample(logits, specs, top_n=5)
    ref = torch.log_softmax(logits.double(), 1)
    want_v, want_i = torch.topk(ref, 5, dim=1)
    assert top_ids.tolist() == want_i.cpu().tolist()
    close(top_lp, want_v.cpu(), atol=1e-4)
    for i in range(3):
        assert tokens[i] == top_ids[i, 0].item()


def test_chosen_token_counts_once(lx):
    logits = torch.zeros(1, 64, device="cuda")
    logits[0, 7] = 10.0
    sampler = Sampler(lx, 64, 1)
    sampler.reset(0)
    tokens, _, _, _ = sampler.sample(logits, [dict(slot=0, row=0, temperature=0.0)])
    assert tokens == [7]
    assert sampler.seen[0, 7].item() == 1 and sampler.out[0, 7].item() == 1
    sampler.reset(0)
    assert sampler.seen.sum().item() == 0 and sampler.out.sum().item() == 0
