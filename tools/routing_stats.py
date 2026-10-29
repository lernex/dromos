"""Routing statistics for expert placement, from LX_ROUTE_TRACE files, and a simulation of the cache.

    python tools/routing_stats.py TRACE [--heldout TRACE2] [--stats OUT] [--experts E] [--layers L]

A trace is what `LX_ROUTE_TRACE=FILE dromos offload eval ...` writes: records of int32
(pass, layer, T) followed by T * 4 expert ids, in execution order; records whose pass has bit 16 set
hold the next layer's routing predicted from the streams as they leave the previous layer (8 ids per
token). Reported:
  skew        how much of the routing mass the hottest experts carry (per layer and overall)
  reuse       how many of a token's pass-2 experts it already used in pass 1 (same layer)
  locality    how many of a token's experts the previous token used (same pass and layer)
  prediction  recall of the next-layer prediction (actual top-4 within the predicted top-4 / top-8)
  cache       VRAM hit rate per policy and capacity, replaying the (held-out) trace in decode order
              (token by token: pass 1 layers 0..L-1, then pass 2), with the runtime's policies:
              static (hottest experts by calibration frequency), LRU per layer, and static + LRU
              (the runtime's default, dynamic share d, dynamic slots per layer in proportion to the
              mass the static set leaves uncovered), with and without next-layer prefetch
--stats writes per-layer expert counts over both passes (the file LX_ROUTING_STATS reads).
"""

import argparse
import collections

import numpy as np


def read_trace(path, K=4, P=8):
    raw = np.fromfile(path, dtype=np.int32)
    actual = collections.defaultdict(list)    # (pass, layer) -> list of [T, K]
    predicted = collections.defaultdict(list)  # (pass, layer) -> list of [T, P]
    i = 0
    while i < len(raw):
        p, layer, T = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
        i += 3
        if p & 0x10000:
            predicted[(p & 0xFFFF, layer)].append(raw[i:i + T * P].reshape(T, P))
            i += T * P
        else:
            actual[(p, layer)].append(raw[i:i + T * K].reshape(T, K))
            i += T * K
    act = {k: np.concatenate(v) for k, v in actual.items()}
    pred = {k: np.concatenate(v) for k, v in predicted.items()}
    passes = 1 + max(k[0] for k in act)
    layers = 1 + max(k[1] for k in act)
    return act, pred, passes, layers


def frequencies(act, passes, layers, E):
    f = np.zeros((layers, E))
    for (p, l), ids in act.items():
        f[l] += np.bincount(ids.ravel(), minlength=E)
    return f


def overlap(a, b):
    """Per row, |a ∩ b| (a [T, k1], b [T, k2])."""
    return (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)


def simulate(act, passes, layers, E, slots, pinned_sets, prefetch=None, pred=None):
    """Hits and misses replaying decode order with per-layer caches.

    slots[l]: VRAM slots of layer l; pinned_sets[l]: the experts pinned there (the rest of the slots
    start with the next entries of init order and follow LRU). prefetch: experts of the prediction to
    bring in before each step (None: no prefetch). Returns (hits, misses, prefetched, prefetch_used)."""
    T = act[(0, 0)].shape[0]
    hits = misses = fetched = used = 0
    for l in range(layers):
        pinned = set(pinned_sets[l][0])
        dyn_init = list(pinned_sets[l][1])
        n_dyn = slots[l] - len(pinned)
        # dynamic slots: expert -> last use stamp
        dyn = collections.OrderedDict((e, 0) for e in dyn_init[:n_dyn])
        fresh = set()  # prefetched, not yet used
        clock = 0
        for t in range(T):
            for p in range(passes):
                clock += 1
                if prefetch and l > 0 and (p, l) in pred and n_dyn > 0:
                    for e in pred[(p, l)][t, :prefetch]:
                        e = int(e)
                        if e in pinned or e in dyn:
                            if e in dyn:
                                dyn.move_to_end(e)
                            continue
                        dyn.popitem(last=False) if len(dyn) >= n_dyn else None
                        dyn[e] = clock
                        fresh.add(e)
                        fetched += 1
                step = [int(e) for e in act[(p, l)][t]]
                miss = []
                for e in step:
                    if e in pinned:
                        hits += 1
                    elif e in dyn:
                        hits += 1
                        dyn.move_to_end(e)
                        if e in fresh:
                            used += 1
                            fresh.discard(e)
                    else:
                        miss.append(e)
                misses += len(miss)
                # fills: the least recently used dynamic slots not used by this step
                in_step = set(step)
                for e in miss:
                    if n_dyn <= 0:
                        break
                    if len(dyn) < n_dyn:
                        dyn[e] = clock
                        continue
                    victim = next((v for v in dyn if v not in in_step), None)
                    if victim is None:
                        break
                    del dyn[victim]
                    fresh.discard(victim)
                    dyn[e] = clock
    return hits, misses, fetched, used


def placement(freq, capacity, dynamic):
    """The runtime's placement (engine/runtime/offload.cpp finalize): returns slots[l], (pinned, rest)."""
    layers, E = freq.shape
    C = int(capacity)
    n_static = int(C * (1 - dynamic))
    order = np.argsort(-freq.ravel(), kind="stable")
    placed = np.zeros(layers * E, bool)
    placed[order[:n_static]] = True
    pinned = [sorted(np.nonzero(placed.reshape(layers, E)[l])[0], key=lambda e: -freq[l, e]) for l in range(layers)]
    residual = np.array([freq[l][~placed.reshape(layers, E)[l]].sum() + 1e-9 for l in range(layers)])
    extra = np.minimum(E - np.array([len(p) for p in pinned]), np.floor((C - n_static) * residual / residual.sum())).astype(int)
    left = C - n_static - extra.sum()
    while left > 0:
        progressed = False
        for l in range(layers):
            if left and len(pinned[l]) + extra[l] < E:
                extra[l] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    sets, slots = [], []
    for l in range(layers):
        rest = [e for e in np.argsort(-freq[l], kind="stable") if not placed[l * E + e]]
        sets.append((pinned[l], rest))
        slots.append(len(pinned[l]) + extra[l])
    return slots, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--heldout")
    ap.add_argument("--stats")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--capacities", default="0.3,0.45,0.6,0.7,0.8,0.9")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    act, pred, passes, layers = read_trace(a.trace)
    E = a.experts or int(max(v.max() for v in act.values())) + 1
    T = act[(0, 0)].shape[0]
    freq = frequencies(act, passes, layers, E)
    print(f"{a.trace}: {T} tokens, {passes} passes, {layers} layers, {E} experts")
    if a.stats:
        with open(a.stats, "w") as f:
            f.write(f"# expert counts per layer over both passes, {T} tokens\n")
            for l in range(layers):
                f.write(" ".join(str(int(x)) for x in freq[l]) + "\n")

    # skew
    print("\nskew: share of the routing mass on the hottest experts of each layer (mean over layers, min..max)")
    shares = {}
    for frac in (0.1, 0.25, 0.5, 0.75):
        n = int(round(frac * E))
        s = [np.sort(freq[l])[::-1][:n].sum() / freq[l].sum() for l in range(layers)]
        shares[frac] = s
        print(f"  top {frac:4.0%} of experts: {np.mean(s):6.1%}  ({min(s):.1%} .. {max(s):.1%})")
    allf = np.sort(freq.ravel())[::-1]
    print("  over all layers at once (what a static cache of that share of every expert covers):")
    for frac in (0.25, 0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"    {frac:4.0%}: {allf[: int(frac * len(allf))].sum() / allf.sum():6.1%}")
    unused = (freq == 0).sum()
    print(f"  experts never chosen in the calibration set: {unused} of {layers * E}")
    per_layer = " ".join(f"{np.sort(freq[l])[::-1][: E // 4].sum() / freq[l].sum():.2f}" for l in range(layers))
    print(f"  per layer, top-25% share: {per_layer}")

    # reuse between the passes
    if passes > 1:
        r = [overlap(act[(1, l)], act[(0, l)]).mean() / 4 for l in range(layers)]
        print(f"\npass reuse: {np.mean(r):.1%} of a token's pass-2 experts it used in pass 1 at the same layer"
              f" ({min(r):.1%} .. {max(r):.1%})")
        print("  per layer: " + " ".join(f"{x:.2f}" for x in r))

    # temporal locality
    loc = []
    for p in range(passes):
        for l in range(layers):
            ids = act[(p, l)]
            loc.append(overlap(ids[1:], ids[:-1]).mean() / 4)
    win = []
    for w in (4, 16):
        acc = []
        for p in range(passes):
            for l in range(layers):
                ids = act[(p, l)]
                prev = np.stack([ids[w - d:len(ids) - d] for d in range(1, w + 1)], 1).reshape(len(ids) - w, -1)
                acc.append(overlap(ids[w:], prev).mean() / 4)
        win.append((w, np.mean(acc)))
    print(f"\nlocality: {np.mean(loc):.1%} of a token's experts the previous token used (same pass and layer); "
          + ", ".join(f"within the last {w}: {x:.1%}" for w, x in win))

    # next-layer prediction
    if pred:
        r4, r8 = [], []
        for (p, l), pids in pred.items():
            r4.append(overlap(act[(p, l)], pids[:, :4]).mean() / 4)
            r8.append(overlap(act[(p, l)], pids[:, :8]).mean() / 4)
        print(f"\nnext-layer prediction (streams leaving layer l-1, layer l's MoE read and router): recall of the"
              f" actual top-4 in the predicted top-4 {np.mean(r4):.1%}, top-8 {np.mean(r8):.1%}")
        if passes > 1:
            pr = [overlap(act[(1, l)], act[(0, l)]).mean() / 4 for l in range(layers)]
            print(f"  (pass 1's routing as the prediction for pass 2: {np.mean(pr):.1%})")

    # cache simulation
    if a.quick:
        return
    held = act
    hpred = pred
    if a.heldout:
        held, hpred, _, _ = read_trace(a.heldout)
        print(f"\ncache simulation on the held-out trace ({held[(0, 0)].shape[0]} tokens), placement from the calibration trace")
    else:
        print("\ncache simulation on the calibration trace itself")
    caps = [float(x) for x in a.capacities.split(",")]
    Th = held[(0, 0)].shape[0]
    print(f"  {'capacity':>8} {'policy':<26} {'hit rate':>8} {'misses/token':>12} {'prefetch/token':>14} {'used':>6}")
    for cap in caps:
        C = int(cap * layers * E)
        configs = [("static", 0.0, None), ("static + LRU d=0.25", 0.25, None), ("static + LRU d=0.50", 0.50, None),
                   ("LRU", 1.0, None), ("LRU + prefetch 1", 1.0, 1), ("LRU + prefetch 2", 1.0, 2),
                   ("LRU + prefetch 3", 1.0, 3), ("LRU + prefetch 4", 1.0, 4)]
        for name, d, pf in configs:
            if pf and not hpred:
                continue
            if d == 1.0:
                slots = [C // layers + (1 if l < C % layers else 0) for l in range(layers)]
                sets = [([], list(np.argsort(-freq[l], kind="stable"))) for l in range(layers)]
            else:
                slots, sets = placement(freq, C, d)
            h, m, f, u = simulate(held, passes, layers, E, slots, sets, pf, hpred)
            print(f"  {cap:8.0%} {name:<26} {h / (h + m):8.1%} {m / Th:12.2f} {f / Th:14.2f} {(u / f if f else 0):6.1%}")


if __name__ == "__main__":
    main()
