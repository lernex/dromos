"""Per-kernel table of a decode step, each kernel's bytes against the DRAM floor (896 GB/s).

Two nsys runs of the same model, one eager and serial, one as the engine runs it (graph replay):

  LX_NO_GRAPHS=1 LX_SERIAL_DECODE=1 LX_LINEAR_TRACE=out/prof/linear.txt \\
    nsys profile --cuda-graph-trace=node -o out/prof/eager --force-overwrite=true \\
    build/dromos bench MODEL --streams 1 --prompt-tokens 64 --gen-tokens 16
  nsys export --type sqlite -o out/prof/eager.sqlite --force-overwrite=true out/prof/eager.nsys-rep
  nsys profile --cuda-graph-trace=node -o out/prof/graph --force-overwrite=true \\
    build/dromos bench MODEL --streams 1 --prompt-tokens 64 --gen-tokens 128
  nsys export --type sqlite -o out/prof/graph.sqlite --force-overwrite=true out/prof/graph.nsys-rep

  python bench/profile_decode.py out/prof/linear.txt out/prof/eager.sqlite [out/prof/graph.sqlite]

LX_LINEAR_TRACE (metis.cpp) writes one "step T" line per run() and one "gemv N K FORMAT M xdtype ydtype beta"
line per decode-sized product, in issue order. run() starts with one embed_kernel, so the kernels of the eager
export split into steps at those, and the GEMV kernels of a step pair with its gemv lines in order (LX_SERIAL_DECODE
puts every kernel on one stream, so issue order is stream order). Each GEMV's bytes are its weights, scales, x and y.
Other kernels are listed by name with their time only. The graph export gives the step's span as the engine
replays it: the kernel times above are the same kernels run back to back, so the span is the number to compare
with tok/s.
"""

import argparse
import collections
import re
import sqlite3

DRAM_GBPS = 896.0


def base(demangled):
    """The kernel's base name: 'gemv_vec_kernel' for 'void gemv_vec_kernel<...>(...)'."""
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[<(]", demangled.replace("void ", "", 1))
    return m.group(1) if m else demangled[:40]


TRACED = ("gemv_vec_kernel", "gemv_small_kernel")  # the kernels linear() launches, one per traced product


def is_gemv(name):
    """A kernel that pairs with a LX_LINEAR_TRACE gemv line. The f32 GEMV (routers, continuation head) is not traced."""
    return name in TRACED


def gemv_bytes(n, k, fmt, m, xdt, ydt, acc):
    """Bytes a GEMV must move: weights, their scales, x read, y written (and read when beta != 0)."""
    if fmt == "BF16":
        w = 2 * n * k
    elif fmt == "BF16X":
        w = 1.53 * n * k  # the lossless encoding: ~12.2 bits a value (bf16x.cuh)
    elif fmt in ("INT8", "FP8"):
        w = n * k + 4 * n  # one f32 scale per row
    elif fmt == "NVFP4":
        w = n * k / 2 + n * k / 16  # two e2m1 a byte, one e4m3 scale per 16 values (the global is one float)
    else:
        w = 0
    x = m * k * (4 if xdt == "f32" else 2)
    y = m * n * (4 if ydt == "f32" else 2) * (2 if acc else 1)
    return w + x + y


def exclusive_us(seg):
    """Each kernel's share of the covered time, in us: kernels in start order, each charged only for the part of its
    run that no earlier-started kernel covers. Programmatic launch starts a kernel while its predecessor drains, so a
    kernel's raw duration includes that wait; these shares add up to the step's covered time instead."""
    order = sorted(range(len(seg)), key=lambda i: seg[i][0])
    out = [0.0] * len(seg)
    covered = 0
    for i in order:
        s, e = seg[i][0], seg[i][1]
        out[i] = max(0, e - max(s, covered)) / 1e3
        covered = max(covered, e)
    return out


def read_trace(path):
    """[(T, [(n, k, fmt, m, xdt, ydt, acc), ...]), ...]: one entry per run()."""
    steps = []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "step":
            steps.append((int(p[1]), []))
        elif p[0] == "gemv" and steps:
            n, k, fmt, m, xdt, ydt, acc = p[1:8]
            steps[-1][1].append((int(n), int(k), fmt, int(m), xdt, ydt, acc == "1"))
    return steps


def load_kernels(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    rows = db.execute(
        "select start, end, demangledName, gridX, blockX, graphNodeId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        "order by start"
    ).fetchall()
    return [(s, e, names.get(n, "?"), gx, bx, g, c) for s, e, n, gx, bx, g, c in rows]


def eager_steps(kernels):
    """Kernels split at each embed_kernel: one list per run() of the engine."""
    cuts = [i for i, k in enumerate(kernels) if base(k[2]) == "embed_kernel"]
    return [kernels[a:b] for a, b in zip(cuts, cuts[1:] + [len(kernels)])]


def graph_span(path, drop=2):
    """The engine's decode step as replayed: span, kernel busy time, and kernels per step from graph kernels."""
    rows = [k for k in load_kernels(path) if k[5] is not None]
    by = collections.OrderedDict()
    for k in rows:
        by.setdefault(k[6], []).append(k)
    steps = list(by.values())
    if not steps:
        return None
    mode = collections.Counter(len(s) for s in steps).most_common(1)[0][0]
    steps = [s for s in steps if len(s) == mode][drop:]
    if not steps:
        return None
    n = len(steps)
    span = sum(s[-1][1] - s[0][0] for s in steps) / n
    busy = sum(sum(k[1] - k[0] for k in s) for s in steps) / n
    return {"steps": n, "kernels": mode, "span_us": span / 1e3, "busy_us": busy / 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("eager")
    ap.add_argument("graph", nargs="?")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--drop", type=int, default=2, help="warm-up decode steps to skip")
    args = ap.parse_args()

    trace = read_trace(args.trace)
    segs = eager_steps(load_kernels(args.eager))
    if len(segs) != len(trace):
        print(f"warning: {len(segs)} runs in the export, {len(trace)} in the trace; pairing from the start")

    decode = []  # (segment kernels, gemv lines) for every one-token step
    bad = 0
    for (t, gemvs), seg in zip(trace, segs):
        if t != 1:
            continue
        seq = [i for i, k in enumerate(seg) if is_gemv(base(k[2]))]
        if len(seq) != len(gemvs):
            bad += 1
            continue
        decode.append((seg, gemvs))
    if bad:
        print(f"warning: {bad} decode steps had a different GEMV count from their trace and were skipped")
    if not decode:
        raise SystemExit("no decode steps to read")

    sigs = collections.Counter(tuple(base(k[2]) for k in seg) for seg, _ in decode)
    sig, _ = sigs.most_common(1)[0]
    use = [(seg, g) for seg, g in decode if tuple(base(k[2]) for k in seg) == sig][args.drop :]
    if not use:
        use = [(seg, g) for seg, g in decode if tuple(base(k[2]) for k in seg) == sig]
    n = len(use)
    print(f"decode steps used {n}, kernels per step {len(sig)}, GEMVs per step "
          f"{sum(1 for k in use[0][0] if is_gemv(base(k[2])))}")

    agg = collections.OrderedDict()  # label -> [count, us, bytes, kind]
    for seg, gemvs in use:
        gi = 0
        excl = exclusive_us(seg)
        for i, k in enumerate(seg):
            us = excl[i]
            if is_gemv(base(k[2])):
                nn, kk, fmt, m, xdt, ydt, acc = gemvs[gi]
                gi += 1
                label = f"{fmt:6s} {nn:>6} x {kk:<5} x{xdt} M{m}"
                nbytes = gemv_bytes(nn, kk, fmt, m, xdt, ydt, acc)
                kind = "gemv"
            else:
                label = base(k[2])
                nbytes = None
                kind = "other"
            row = agg.setdefault((kind, label), [0, 0.0, 0.0 if nbytes is not None else None])
            row[0] += 1
            row[1] += us
            if nbytes is not None:
                row[2] += nbytes
    total_us = sum(v[1] for v in agg.values()) / n
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"{'us/step':>8} {'%':>5} {'n/step':>6} {'us each':>8} {'MB/step':>8} {'GB/s':>6} {'%896':>5} {'floor us':>8}  kernel")
    for (kind, label), (cnt, us, nbytes) in rows[: args.top]:
        us_step = us / n
        each = us_step / (cnt / n)
        if nbytes is not None:
            mb = nbytes / n / 1e6
            floor = nbytes / n / (DRAM_GBPS * 1e3)
            gbs = nbytes / n / (us_step * 1e3) if us_step > 0 else 0.0
            print(f"{us_step:8.2f} {100 * us_step / total_us:5.1f} {cnt / n:6.1f} {each:8.2f} {mb:8.3f} {gbs:6.0f} "
                  f"{100 * gbs / DRAM_GBPS:5.1f} {floor:8.2f}  {label}")
        else:
            print(f"{us_step:8.2f} {100 * us_step / total_us:5.1f} {cnt / n:6.1f} {each:8.2f} {'':>8} {'':>6} {'':>5} "
                  f"{'':>8}  [{label}]")
    gemv_us = sum(v[1] for (kind, _), v in agg.items() if kind == "gemv") / n
    gemv_bytes_total = sum(v[2] for (kind, _), v in agg.items() if kind == "gemv" and v[2] is not None) / n
    print(f"\ncovered GPU time: {total_us:.1f} us/step (GEMVs {gemv_us:.1f} us, {100 * gemv_us / total_us:.1f}%)")
    floor_us = gemv_bytes_total / (DRAM_GBPS * 1e3)
    print(f"GEMV bytes {gemv_bytes_total / 1e6:.1f} MB/step: floor {floor_us:.1f} us, achieved {gemv_us:.1f} us "
          f"({100 * floor_us / gemv_us:.1f}% of 896 GB/s)")
    if args.graph:
        g = graph_span(args.graph, args.drop)
        if g:
            print(f"graph replay: {g['steps']} steps of {g['kernels']} kernels, span {g['span_us']:.1f} us "
                  f"({1e6 / g['span_us']:.0f} tok/s at 1 stream), kernel busy {g['busy_us']:.1f} us")


if __name__ == "__main__":
    main()
