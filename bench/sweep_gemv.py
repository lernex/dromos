"""The lx_gemv_w autotuner, run offline on the GPU server. The engine never runs it.

Every variant is timed for each format, Praxis decode shape and M, with the same graph timing as
bench_gemv.py, and the heuristic is timed beside them. The choice then has two steps:

  1. The order class of each (format, shape): the split whose worst relative loss over M = 1, 2, 4, 8 is
     smallest (CLASS=minimax, the default). The single-stream guard keeps the M = 1 class for a shape whose
     minimax class costs M = 1 more than CLASS_GUARD. The M = 1 class (CLASS=m1) is the split fastest at M = 1,
     with splits within CLASS_TIE of the fastest broken by their worst loss over the other buckets.
  2. The variant for each M bucket: the fastest variant of that class at M. The checked loop (G = 1) is left
     out below M = 8, where it runs the same code as its round-loop twin.

A row's bits depend on the class alone, so every M bucket of a shape shares its class. lx_tune_set admits a
variant only inside the class the kernel's pick table gives the shape (lx::TuneAdmit in gemv.cu), and the
shipped table must load in full: a table entry outside its class is refused. The class is written into
gemv.cu's kPicks; the tuner checks the chosen variants against what the built library admits and says so when
the two disagree (paste KPICKS and rebuild, then run again).

Outputs:
  SWEEP_CSV   every measurement, with the class and the chosen variant per M marked
  KPICKS      the Pick rows for kPicks in engine/cuda/gemv.cu
  TUNE_MERGE  a tuning table: its gemv_w_* entries are replaced, the rest kept

Environment: LX_KERNELS, FORMATS (default bf16,int8,fp8,nvfp4), M (default 1,2,4,8), SHAPES (N:K pairs, default
the Praxis set), CLASS (minimax or m1), SWEEP_CSV, KPICKS, TUNE_MERGE.
Run: scripts/remote.sh formats x '$PY bench/sweep_gemv.py'
"""

import ctypes
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gemv as B  # noqa: E402

lib = B.lib
lib.lx_tune_set.restype = ctypes.c_int
lib.lx_tune_set.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.lx_tune_force.restype = ctypes.c_int
lib.lx_tune_force.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.lx_tune_clear.restype = None
lib.lx_tune_clear.argtypes = []
lib.lx_tune_variants.restype = ctypes.c_int
lib.lx_tune_variants.argtypes = [ctypes.c_char_p]
lib.lx_tune_variant_name.restype = ctypes.c_char_p
lib.lx_tune_variant_name.argtypes = [ctypes.c_char_p, ctypes.c_int]

F32, BF16 = 0, 1
FORMAT_CODE = {"bf16": "LX_W_BF16", "int8": "LX_W_INT8", "fp8": "LX_W_FP8", "nvfp4": "LX_W_NVFP4"}
BUCKETS = (1, 2, 4, 8)


def time_us(copies, x, y, N, K, M, x_dtype):
    launches = []
    for lw, _ in copies:
        pointer = ctypes.pointer(lw)
        launches.append(
            lambda p=pointer: B.ok(
                lib.lx_gemv_w(
                    x.data_ptr(), x_dtype, K, p, None, F32, y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, B.stream(),
                )
            )
        )
    return B.graph_us_per_launch(launches)


def config(name):
    """(S, RG, D, G) of a variant name: "s2r4d3g" is (2, 4, 3, 1)."""
    s, rest = name[1:].split("r", 1)
    rg, rest = rest.split("d", 1)
    return int(s), int(rg), int(rest.rstrip("g")), int(name.endswith("g"))


def split(name):
    return config(name)[0]


def checked(name):
    return config(name)[3] == 1


# Splits within this of the fastest at M = 1 tie there: timing noise is about this size, and a class must not
# flip on it. The tie is broken by the worst loss over the other buckets.
CLASS_TIE = 0.01
# Single-stream guard: the minimax class is used only if it costs M = 1 no more than this against the M = 1
# class. Otherwise the shape keeps the M = 1 class.
CLASS_GUARD = 0.01


def choose_class(names, us, ms, criterion):
    """The split of the order class, and why: "" for the criterion's own choice, "guard" when the single-stream
    guard kept the M = 1 class. us[(M, v)] is the time of variant v at M."""
    splits = sorted({split(n) for n in names})

    def best(M, s):
        return min(us[(M, v)] for v, n in enumerate(names) if split(n) == s)

    best_all = {M: min(us[(M, v)] for v in range(len(names))) for M in ms}

    def worst_loss(s, buckets):
        return max((best(M, s) / best_all[M] - 1 for M in buckets), default=0.0)

    fastest = min(best(1, s) for s in splits)
    tied = [s for s in splits if best(1, s) <= fastest * (1 + CLASS_TIE)]
    m1 = min(tied, key=lambda s: worst_loss(s, [M for M in ms if M != 1]))
    if criterion == "m1":
        return m1, ""
    mm = min(splits, key=lambda s: worst_loss(s, ms))
    if best(1, mm) > best(1, m1) * (1 + CLASS_GUARD):
        return m1, "guard"
    return mm, ""


def main():
    formats = os.environ.get("FORMATS", "bf16,int8,fp8,nvfp4").split(",")
    ms = [int(m) for m in os.environ.get("M", "1,2,4,8").split(",")]
    criterion = os.environ.get("CLASS", "minimax")
    if criterion not in ("m1", "minimax"):
        raise SystemExit(f"CLASS must be m1 or minimax, not {criterion}")
    if criterion == "m1" and 1 not in ms:
        raise SystemExit("CLASS=m1 needs M=1 in the sweep")
    if "SHAPES" in os.environ:
        shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ["SHAPES"].split(",")]
    else:
        shapes = B.SHAPES
    sweep_path = os.environ.get("SWEEP_CSV", "sweep.csv")
    kpicks_path = os.environ.get("KPICKS", "kpicks.txt")
    merge_path = os.environ.get("TUNE_MERGE")
    print(f"device: {torch.cuda.get_device_name(0)}; DRAM {B.DRAM_GBPS:.0f} GB/s; class by {criterion}", flush=True)

    kpicks, entries = [], []
    with open(sweep_path, "w") as csv:
        csv.write("format,N,K,M,variant,us,gbs,class,chosen\n")
        for fmt in formats:
            kernel = f"gemv_w_{fmt}"
            nvar = lib.lx_tune_variants(kernel.encode())
            names = [lib.lx_tune_variant_name(kernel.encode(), v).decode() for v in range(nvar)]
            for N, K in shapes:
                nbytes = B.weight_bytes(N, K, fmt)
                copies = [B.make_copy(N, K, fmt) for _ in range(B.copies_for(nbytes))]
                us, heur = {}, {}
                for M in ms:
                    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                    y = torch.empty(M, N, dtype=torch.float32, device="cuda")
                    lib.lx_tune_clear()
                    heur[M] = time_us(copies, x, y, N, K, M, BF16)
                    for v in range(nvar):
                        lib.lx_tune_clear()
                        if lib.lx_tune_force(kernel.encode(), M, N, K, v) != 0:
                            raise SystemExit(f"{kernel} has no variant {v}")
                        us[(M, v)] = time_us(copies, x, y, N, K, M, BF16)
                    lib.lx_tune_clear()
                cls, why = choose_class(names, us, ms, criterion)
                chosen = {}
                for M in ms:
                    pool = [v for v in range(nvar) if split(names[v]) == cls and (M == 8 or not checked(names[v]))]
                    chosen[M] = min(pool, key=lambda v: (us[(M, v)], v))
                    # The built library must admit the chosen variant and refuse the rest of the shape's classes.
                    lib.lx_tune_clear()
                    status = lib.lx_tune_set(kernel.encode(), M, N, K, chosen[M])
                    if status != 0:
                        print(f"  class mismatch: {kernel} {N}x{K} class {cls} is refused (status {status}); "
                              "paste KPICKS into gemv.cu and rebuild before the next run", flush=True)
                    lib.lx_tune_clear()
                for M in ms:
                    gbs_h = nbytes / (heur[M] * 1e-6) / 1e9
                    csv.write(f"{fmt},{N},{K},{M},heuristic,{heur[M]:.3f},{gbs_h:.1f},,\n")
                    for v in range(nvar):
                        gbs = nbytes / (us[(M, v)] * 1e-6) / 1e9
                        csv.write(
                            f"{fmt},{N},{K},{M},{names[v]},{us[(M, v)]:.3f},{gbs:.1f},"
                            f"{int(split(names[v]) == cls)},{int(chosen[M] == v)}\n"
                        )
                    gbs_c = nbytes / (us[(M, chosen[M])] * 1e-6) / 1e9
                    entries.append({"key": kernel, "M": M, "N": N, "K": K, "variant": chosen[M],
                                    "name": names[chosen[M]], "us": us[(M, chosen[M])]})
                    kpicks.append((FORMAT_CODE[fmt], M, N, K, config(names[chosen[M]])))
                    print(f"| {fmt} | {N} x {K} | M={M} | class S={cls}{' (guard: M=1 class)' if why else ''} | heuristic {gbs_h:.0f} GB/s "
                          f"({100 * gbs_h / B.DRAM_GBPS:.1f}%) | chosen {names[chosen[M]]} {gbs_c:.0f} GB/s "
                          f"({100 * gbs_c / B.DRAM_GBPS:.1f}%) |", flush=True)
                del copies
                torch.cuda.empty_cache()

    if set(BUCKETS) <= set(ms):
        lines = [f"    {{{f}, {M}, {N}, {K}, {{{s}, {rg}, {d}, {g}}}}}," for f, M, N, K, (s, rg, d, g) in kpicks]
        Path(kpicks_path).write_text("\n".join(lines) + "\n")
        print(f"wrote {sweep_path} and {kpicks_path} ({len(lines)} kPicks rows)", flush=True)
    else:
        print(f"wrote {sweep_path}; no kPicks file, the sweep needs M = 1, 2, 4 and 8", flush=True)

    if merge_path:
        doc = json.loads(Path(merge_path).read_text())
        doc["entries"] = [e for e in doc["entries"] if not e["key"].startswith("gemv_w_")] + entries
        Path(merge_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"merged {len(entries)} gemv_w entries into {merge_path}", flush=True)


if __name__ == "__main__":
    main()
