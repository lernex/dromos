"""lx_mask_logits (the constrained-decoding mask, engine/cuda/sampler.cu) against PyTorch.

Entry j applies its mask bits to logits row rows[j]: a token whose bit is clear becomes -inf, a set bit leaves the
logit alone. Rows nothing names are untouched. Checked on a vocabulary that is not a multiple of 32 and on a
production-sized one, with random masks plus an all-clear and an all-set row.
"""

import pytest
import torch

NEG_INF = float("-inf")


def reference(logits, rows, bits, vocab):
    """The masked logits, computed with the bits read as unsigned 32-bit words."""
    out = logits.clone()
    idx = torch.arange(vocab, device=logits.device)
    words = bits.to(torch.int64) & 0xFFFFFFFF
    for j, r in enumerate(rows):
        allowed = ((words[j][idx // 32] >> (idx % 32)) & 1).bool()
        out[r] = torch.where(allowed, logits[r], torch.full_like(logits[r], NEG_INF))
    return out


def random_bits(entries, vocab):
    words = (vocab + 31) // 32
    raw = torch.randint(0, 2**32, (entries, words), dtype=torch.int64)
    signed = raw - (raw >= 2**31).to(torch.int64) * 2**32
    return signed.to(torch.int32)


@pytest.mark.parametrize("vocab", [1000, 65536])
def test_listed_rows_are_masked_and_the_rest_untouched(lx, vocab):
    torch.manual_seed(20261028)
    slots = 8
    logits = torch.randn(slots, vocab, dtype=torch.float32, device="cuda")
    rows_list = [5, 0, 3]
    bits = random_bits(len(rows_list), vocab)
    bits[0].zero_()  # every token ruled out: the row is all -inf
    bits[1].fill_(-1)  # every token allowed: the row is unchanged
    rows = torch.tensor(rows_list, dtype=torch.int32, device="cuda")
    bits = bits.to("cuda")
    expected = reference(logits, rows_list, bits, vocab)

    got = logits.clone()
    lx("mask_logits", got, vocab, rows, bits, len(rows_list))
    assert torch.equal(got, expected)
    assert torch.isinf(got[5]).all() and (got[5] == NEG_INF).all()
    assert torch.equal(got[0], logits[0])  # all allowed
    untouched = [r for r in range(slots) if r not in rows_list]
    assert torch.equal(got[untouched], logits[untouched])


def test_zero_entries_is_a_no_op(lx):
    logits = torch.randn(2, 96, dtype=torch.float32, device="cuda")
    before = logits.clone()
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    bits = torch.zeros(1, 3, dtype=torch.int32, device="cuda")
    lx("mask_logits", logits, 96, rows, bits, 0)
    assert torch.equal(logits, before)
