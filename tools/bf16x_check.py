"""Encode every decode weight of a Metis checkpoint as LX_W_BF16X, check that each decodes back to its
exact bf16 bits, and report the compressed sizes.

    python tools/bf16x_check.py MODEL_DIR

Decode weights are every large projection the engine reads per token (Mamba in/out, attention qkv/out,
the MoE latent, shared and routed experts, the depth-memory projections, the LM head). Needs
LX_KERNELS (scripts/remote.sh sets it); run through `remote.sh SLOT x` since it loads the whole model.
"""

import collections
import ctypes
import os
import pathlib
import sys

import torch
from safetensors import safe_open

GROUPS = [  # (name fragment, label) in report order
    ("mixer.impl.mixer.in_proj", "mamba in_proj"),
    ("mixer.impl.mixer.out_proj", "mamba out_proj"),
    ("mixer.qkv", "attention qkv"),
    ("mixer.out.", "attention out"),
    ("moe.latent_down", "moe latent_down"),
    ("moe.latent_up", "moe latent_up"),
    ("shared_expert.gate_up", "shared gate_up"),
    ("shared_expert.down", "shared down"),
    ("local_experts.gate_up", "experts gate_up"),
    ("local_experts.down", "experts down"),
    ("depth_memory.state_write", "memory state_write"),
    ("depth_memory.query", "memory query"),
    ("depth_memory.output", "memory output"),
    ("depth_memory.route_projection", "memory route_projection"),
    ("embedding.weight", "lm head (embedding)"),
]


def main():
    model = pathlib.Path(sys.argv[1])
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    lib.lx_bf16x_bytes.restype = ctypes.c_int64
    lib.lx_bf16x_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    lib.lx_bf16x_row_bytes.restype = ctypes.c_int64
    lib.lx_bf16x_row_bytes.argtypes = [ctypes.c_int]
    lib.lx_bf16x_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.c_void_p]
    lib.lx_bf16x_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    stream = torch.cuda.current_stream().cuda_stream

    f = safe_open(str(model / "model.safetensors"), "pt", device="cuda")
    stats = collections.OrderedDict((label, [0, 0, 0, 0, 0, None]) for _, label in GROUPS)  # tensors, values, bytes, rows, overflow rows, shape
    exact = True
    for name in f.keys():
        label = next((lab for frag, lab in GROUPS if frag in name), None)
        if label is None or not name.endswith("weight") and "local_experts" not in name:
            continue
        w = f.get_tensor(name).contiguous()
        if w.dtype != torch.bfloat16 or w.dim() < 2:
            continue
        K = w.shape[-1]
        rows = w.numel() // K
        n = lib.lx_bf16x_bytes(w.data_ptr(), rows, K, stream)
        assert n > 0, f"{name}: lx_bf16x_bytes returned {n}"
        enc = torch.empty(n, dtype=torch.uint8, device="cuda")
        assert lib.lx_bf16x_encode(w.data_ptr(), rows, K, enc.data_ptr(), n, stream) == 0
        back = torch.empty_like(w)
        assert lib.lx_bf16x_decode(enc.data_ptr(), rows, K, back.data_ptr(), stream) == 0
        torch.cuda.synchronize()
        same = torch.equal(back.view(torch.int16), w.view(torch.int16))
        exact &= same
        if not same:
            print(f"MISMATCH {name}")
        s = stats[label]
        s[0] += 1
        s[1] += w.numel()
        s[2] += n
        s[3] += rows
        s[4] += 1 if n > rows * lib.lx_bf16x_row_bytes(K) else 0
        s[5] = tuple(w.shape)
        del w, enc, back

    print("| weights | shape | tensors | values | BF16X bytes | bits / value | vs bf16 |")
    print("|---|---|---|---|---|---|---|")
    tv = tb = 0
    for label, (count, values, nbytes, rows, overflow, shape) in stats.items():
        if count == 0:
            continue
        tv += values
        tb += nbytes
        print(f"| {label} | {' x '.join(map(str, shape))} | {count} | {values:,} | {nbytes:,} | "
              f"{8 * nbytes / values:.3f} | {2 * values / nbytes:.4f}x |")
    print(f"| total | | | {tv:,} | {tb:,} | {8 * tb / tv:.3f} | {2 * tv / tb:.4f}x |")
    print(f"bf16 bytes {2 * tv:,}; saved {2 * tv - tb:,} ({100 * (1 - tb / (2 * tv)):.2f}%)")
    print(f"tensors with an overflow record: {sum(s[4] for s in stats.values())}")
    print("every value decodes to its exact bf16 bits" if exact else "DECODE MISMATCH")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
