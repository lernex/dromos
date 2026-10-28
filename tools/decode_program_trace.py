"""Where a decode program's step time goes: per op, per block, against the DRAM floor.

Run one step of the program with the trace on (the 32nd decode step, or LX_PROGRAM_TRACE_STEP):

    LX_PROGRAM=1 LX_PROGRAM_TRACE=1 LX_PROGRAM_TRACE_OUT=DIR build/lernex bench MODEL --streams 1 \\
        --prompt-tokens 512 --gen-tokens 64

DIR then holds trace.bin (the timestamps of the traced step, from lx_program_trace: per list entry, then per run),
ops.txt (the op names), and opinfo_M<M>.txt (kind, shape, tasks, blocks, dependencies and bytes per op, written by
the builder). Then:

    python3 tools/decode_program_trace.py DIR [--step-ms 3.58] [--gbps 896] [--bin-us 20] [--top 25]

Every block runs its list in order; a run is one op's consecutive tasks on that block. Per run the trace has the
time it started (before its dependencies), the time its dependencies were met, and per task the time it started, the
time its inputs were staged and the time it ended. So a block's time splits into waiting for dependencies (the gap
between the run's start and its dependencies being met), computing (deps met to the task's end), and gaps between
runs. The sync latency of a run is its dependencies being met minus the moment the last of its producers finished on
every block (the producers' last task end), which is the cost of the counter round trip plus the scheduling of the
consumer block.

DRAM busy is estimated from bytes: each run's bytes are spread over the run's window (start to end) and compared
with the bandwidth. That is a proxy: prefetched bytes arrive before their run starts, so early windows understate.
"""

import argparse
import collections
import os
import statistics
import struct

REC = 4  # int64 per record


def read_opinfo(path):
    ops = {}
    with open(path) as f:
        next(f)
        for line in f:
            p = line.split()
            if not p:
                continue
            op = int(p[0])
            ops[op] = {
                "kind": p[1],
                "ntasks": int(p[2]),
                "nblocks": int(p[3]),
                "pool": int(p[4]),
                "waited": int(p[5]),
                "N": int(p[6]),
                "K": int(p[7]),
                "bytes": float(p[8]),
                "wbytes": float(p[9]),
                "deps": [int(x) for x in p[10:]],
            }
    return ops


def read_names(path):
    names = collections.defaultdict(list)
    if not os.path.exists(path):
        return names
    for line in open(path):
        p = line.split(maxsplit=1)
        if len(p) == 2:
            names[int(p[0])].append(p[1].strip())
    return names


def read_trace(path, total_tasks):
    """Per list entry: (op, task, block, start, staged, end) or None; per run head: (run start, deps met) or None."""
    raw = open(path, "rb").read()
    n = len(raw) // (REC * 8)
    recs = struct.iter_unpack("<qqqq", raw[: n * REC * 8])
    rows = list(recs)
    tasks = rows[:total_tasks]
    runs = rows[total_tasks : 2 * total_tasks]
    entries = []
    for pos, r in enumerate(tasks):
        if r[0] == 0 and r[1] == 0:
            entries.append(None)
            continue
        op = r[0] >> 32
        task = (r[0] >> 16) & 0xFFFF
        block = r[0] & 0xFFFF
        entries.append((op, task, block, r[1], r[2], r[3]))
    heads = []
    for pos, r in enumerate(runs):
        heads.append((r[0], r[1], r[2], r[3]) if r[0] != 0 else None)
    return entries, heads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--step-ms", type=float, default=None, help="the bench's step time, for the ratios")
    ap.add_argument("--gbps", type=float, default=896.0)
    ap.add_argument("--bin-us", type=float, default=20.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--timeline", type=int, default=None, metavar="BLOCK",
                    help="print one block's runs in order (times in us from the step's first task)")
    ap.add_argument("--from-run", type=int, default=0, help="the first run of --timeline to print")
    ap.add_argument("--runs", type=int, default=60, help="how many runs --timeline prints")
    a = ap.parse_args()

    ops = read_opinfo(os.path.join(a.dir, f"opinfo_M{a.M}.txt"))
    names = read_names(os.path.join(a.dir, "ops.txt"))
    total_tasks = os.path.getsize(os.path.join(a.dir, "trace.bin")) // (REC * 8) // 2
    entries, heads = read_trace(os.path.join(a.dir, "trace.bin"), total_tasks)

    # per op: its tasks' ends, its runs (per block)
    op_end = collections.defaultdict(int)
    op_tasks = collections.Counter()
    op_blocks = collections.defaultdict(set)
    per_block = collections.defaultdict(list)  # block -> list of run dicts in order
    for pos, e in enumerate(entries):
        if e is None:
            continue
        op, task, block, start, staged, end = e
        op_end[op] = max(op_end[op], end)
        op_tasks[op] += 1
        op_blocks[op].add(block)
        h = heads[pos]
        if h is not None:
            # a new run on this block (or a further piece of the same run)
            run = {"op": op, "start": h[0], "deps": h[1], "end": end, "tasks": 1, "staged": staged,
                   "t0": start, "pf_done": h[2], "desc": h[3]}
            per_block[block].append(run)
        else:
            run = per_block[block][-1]
            run["end"] = max(run["end"], end)
            run["tasks"] += 1
            run["staged"] = max(run["staged"], staged)

    if not per_block:
        raise SystemExit("no trace records: was the traced step run with LX_PROGRAM_TRACE set?")

    t_first = min(r["start"] for runs in per_block.values() for r in runs)
    t_last = max(r["end"] for runs in per_block.values() for r in runs)
    span_ns = t_last - t_first
    blocks = sorted(per_block)

    if a.timeline is not None:
        us = lambda t: f"{(t - t_first) / 1e3:9.2f}" if t else "        -"
        dl = lambda t, base: f"{(t - base) / 1e3:6.2f}" if t and base else "     -"
        print(f"block {a.timeline}: run, op, kind, shape, tasks | start  pf-issued  desc-ready  deps-met  staged  end (us; "
              "deltas in us from the run start in brackets)")
        runs = per_block[a.timeline]
        for k in range(a.from_run, min(len(runs), a.from_run + a.runs)):
            r = runs[k]
            info = ops.get(r["op"], {})
            shape = f"{info.get('N', 0)}x{info.get('K', 0)}" if info.get("kind") == "gemv" else ""
            nm = "; ".join(names.get(r["op"], []))
            print(f"{k:4d} {r['op']:4d} {info.get('kind', '?'):10s} {shape:>11s} {r['tasks']:3d} | "
                  f"{us(r['start'])} {dl(r.get('pf_done'), r['start'])} {dl(r.get('desc'), r['start'])} "
                  f"{dl(r['deps'], r['start'])} {dl(r['staged'], r['start'])} {dl(r['end'], r['start'])}  {nm}")
        print()

    # ready time per op: the last producer's end (every block's last task of each dependency)
    ready = {}
    for op, info in ops.items():
        ready[op] = max((op_end[d] for d in info["deps"] if d in op_end), default=0)

    # per block accounting
    wait_by_op = collections.defaultdict(float)
    comp_by_op = collections.defaultdict(float)
    lat_by_op = collections.defaultdict(list)
    lat_by_kind = collections.defaultdict(list)
    stage_by_kind = collections.defaultdict(float)  # compute split: x staging, then the reader and epilogue
    read_by_kind = collections.defaultdict(float)
    block_wait = []
    block_comp = []
    block_gap = []
    block_active = []
    bytes_windows = collections.defaultdict(float)
    bin_ns = a.bin_us * 1e3
    for b in blocks:
        runs = per_block[b]
        w = c = g = 0.0
        prev_end = None
        for r in runs:
            wait = max(0, r["deps"] - r["start"])
            comp = max(0, r["end"] - r["deps"])
            w += wait
            c += comp
            if prev_end is not None:
                g += max(0, r["start"] - prev_end)
            prev_end = r["end"]
            wait_by_op[r["op"]] += wait
            comp_by_op[r["op"]] += comp
            if ops.get(r["op"], {}).get("deps"):
                # observed latency: from the last producer's end (or this block's arrival, if later) to the
                # dependencies being met; a block that arrives after its producers finished sees none
                lat = r["deps"] - max(r["start"], ready[r["op"]])
                lat_by_op[r["op"]].append(lat)
                lat_by_kind[ops[r["op"]]["kind"]].append(lat)
            k = ops.get(r["op"], {}).get("kind", "?")
            stage_by_kind[k] += max(0, r["staged"] - r["t0"]) if r["staged"] > r["t0"] else 0
            read_by_kind[k] += max(0, r["end"] - max(r["staged"], r["deps"]))
        block_wait.append(w)
        block_comp.append(c)
        block_gap.append(g)
        block_active.append(runs[-1]["end"] - runs[0]["start"])

    # bytes per run: the op's bytes split by tasks, spread over the run's window (start to end)
    for b in blocks:
        for r in per_block[b]:
            info = ops.get(r["op"])
            if not info or not info["ntasks"]:
                continue
            nbytes = info["bytes"] * r["tasks"] / info["ntasks"]
            r["bytes"] = nbytes
            lo, hi = r["start"], r["end"]
            if hi <= lo:
                bytes_windows[lo // bin_ns] += nbytes
            else:
                first = int(lo // bin_ns)
                last = int((hi - 1) // bin_ns)
                for k in range(first, last + 1):
                    a0 = max(lo, k * bin_ns)
                    a1 = min(hi, (k + 1) * bin_ns)
                    bytes_windows[k] += nbytes * (a1 - a0) / (hi - lo)

    total_bytes = sum(info["bytes"] + info["wbytes"] for info in ops.values())
    read_bytes = sum(info["bytes"] for info in ops.values())
    dram_us = total_bytes / (a.gbps * 1e9) * 1e6
    nb = len(blocks)
    print(f"blocks {nb}  ops {len(ops)}  tasks {sum(op_tasks.values())}  runs {sum(len(v) for v in per_block.values())}")
    print(f"step span (first task start to last end): {span_ns / 1e3:.1f} us")
    print(f"bytes per step: reads {read_bytes / 1e6:.1f} MB + state writes {(total_bytes - read_bytes) / 1e6:.1f} MB"
          f" = {total_bytes / 1e6:.1f} MB")
    print(f"DRAM floor at {a.gbps:.0f} GB/s: {dram_us:.1f} us")
    print(f"span / floor: {span_ns / 1e3 / dram_us:.2f}x   floor / span (DRAM fraction used): "
          f"{dram_us / (span_ns / 1e3):.1%}")
    if a.step_ms:
        print(f"bench step {a.step_ms:.3f} ms: floor / step = {dram_us / (a.step_ms * 1e3):.1%}")
    print()

    tot_w = sum(block_wait)
    tot_c = sum(block_comp)
    tot_g = sum(block_gap)
    tot_active = sum(block_active)
    idle = span_ns * nb - tot_active  # blocks idle before their first run or after their last
    print("block time (sum over blocks, share of span x blocks):")
    denom = span_ns * nb
    print(f"  waiting for dependencies  {tot_w / 1e3:10.1f} us  {tot_w / denom:6.1%}")
    print(f"  computing (run to end)    {tot_c / 1e3:10.1f} us  {tot_c / denom:6.1%}")
    print(f"  gaps between runs         {tot_g / 1e3:10.1f} us  {tot_g / denom:6.1%}")
    print(f"  before first / after last {idle / 1e3:10.1f} us  {idle / denom:6.1%}")
    print(f"  mean per block: wait {tot_w / nb / 1e3:.1f} us, compute {tot_c / nb / 1e3:.1f} us, "
          f"gaps {tot_g / nb / 1e3:.1f} us")
    print()

    print("by op kind (sum over blocks; compute split into x staging and reader + epilogue):")
    kinds = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, []])
    for op, info in ops.items():
        k = kinds[info["kind"]]
        k[0] += 1
        k[1] += info["bytes"] + info["wbytes"]
        k[2] += wait_by_op.get(op, 0.0)
        k[3] += comp_by_op.get(op, 0.0)
        k[4].extend(lat_by_op.get(op, []))
    # run start breakdown (runs with the trace's start points): prefetch issue by warp 1, descriptor ready, deps met
    pf_d, desc_d, deps_d = [], [], []
    for b in blocks:
        for r in per_block[b]:
            if r.get("pf_done"):
                pf_d.append(r["pf_done"] - r["start"])
            if r.get("desc"):
                desc_d.append(r["desc"] - r["start"])
                deps_d.append(r["deps"] - r["desc"])
    if pf_d:
        def med(v):
            v = sorted(v)
            return v[len(v) // 2], sum(v) / len(v)
        pm, pa = med(pf_d)
        dm, da = med(desc_d)
        qm, qa = med(deps_d)
        print("run start (ns, median / mean): warp 1 prefetch issue "
              f"{pm:.0f} / {pa:.0f}; descriptor ready {dm:.0f} / {da:.0f} after start; "
              f"deps met {qm:.0f} / {qa:.0f} after the descriptor")
        print()
    print(f"  {'kind':12s} {'ops':>5s} {'MB':>9s} {'wait us':>10s} {'compute us':>11s} {'staging':>9s} "
          f"{'read+epi':>9s} {'share':>7s} {'obs med ns':>11s}")
    for name, (n_ops, by, w, c, lats) in sorted(kinds.items(), key=lambda kv: -(kv[1][2] + kv[1][3])):
        med = statistics.median(lats) if lats else 0
        print(f"  {name:12s} {n_ops:5d} {by / 1e6:9.1f} {w / 1e3:10.1f} {c / 1e3:11.1f} "
              f"{stage_by_kind[name] / 1e3:9.1f} {read_by_kind[name] / 1e3:9.1f} "
              f"{(w + c) / denom:7.1%} {med:11.0f}")
    print()

    all_lat = [x for v in lat_by_op.values() for x in v]
    if all_lat:
        all_lat.sort()
        q = lambda p: all_lat[min(len(all_lat) - 1, int(p * len(all_lat)))]
        print(f"observed sync latency over {len(all_lat)} runs with deps (ns): median {q(0.5):.0f}, "
              f"p90 {q(0.9):.0f}, p99 {q(0.99):.0f}; sum {sum(all_lat) / 1e3:.1f} us")
        print()

    print(f"top {a.top} ops by block time (wait + compute, sum over blocks):")
    print(f"  {'op':>4s} {'kind':10s} {'N x K':>12s} {'tasks':>6s} {'blk':>4s} {'MB':>8s} {'wait us':>9s} "
          f"{'comp us':>9s} {'share':>7s} {'sync med':>9s}  name")
    order = sorted(ops, key=lambda o: -(wait_by_op.get(o, 0) + comp_by_op.get(o, 0)))
    for op in order[: a.top]:
        info = ops[op]
        w, c = wait_by_op.get(op, 0.0), comp_by_op.get(op, 0.0)
        lats = lat_by_op.get(op, [])
        med = statistics.median(lats) if lats else 0
        shape = f"{info['N']}x{info['K']}" if info["kind"] == "gemv" else ""
        nm = "; ".join(names.get(op, [])) or ""
        print(f"  {op:4d} {info['kind']:10s} {shape:>12s} {info['ntasks']:6d} {info['nblocks']:4d} "
              f"{(info['bytes'] + info['wbytes']) / 1e6:8.2f} {w / 1e3:9.1f} {c / 1e3:9.1f} "
              f"{(w + c) / denom:7.2%} {med:9.0f}  {nm}")
    print()

    # DRAM busy proxy: bytes per window against the bandwidth
    if bytes_windows:
        bins = sorted(bytes_windows)
        busy = 0.0
        full = 0
        for k in range(bins[0], bins[-1] + 1):
            rate = bytes_windows.get(k, 0.0) / (bin_ns * 1e-9) / 1e9
            busy += min(1.0, rate / a.gbps) * bin_ns
            if rate >= 0.9 * a.gbps:
                full += 1
        nwin = bins[-1] - bins[0] + 1
        print(f"DRAM proxy over {nwin} windows of {a.bin_us:.0f} us: {busy / (nwin * bin_ns):.1%} of the span "
              f"busy at the bandwidth (bytes over the window / bandwidth), {full / nwin:.1%} of windows at 90%")


if __name__ == "__main__":
    main()
