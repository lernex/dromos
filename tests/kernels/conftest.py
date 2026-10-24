"""The kernel tests call each kernel through the C ABI (engine/cuda/kernels.h) with ctypes, and compare
against a PyTorch reference written beside the test.

LX_KERNELS points at the backend library under test: build/liblernex_kernels.so (CUDA; scripts/remote.sh
sets it) or build-metal/liblernex_metal.dylib (Metal). LX_BACKEND picks the backend, cuda or metal; by
default it follows the library's name. Tests make their tensors on DEVICE: CUDA tensors on the CUDA
backend; on Metal, host tensors, which the backend reads where they are (unified memory), with every
call followed by a stream sync.

Argument types come from the prototypes in kernels.h and tunable.h, so every call passes each argument
at its declared width (an int64_t parameter gets 64 bits, as the ABI requires on arm64).
"""

import ctypes
import os
import pathlib
import re

import pytest
import torch

F32, BF16 = 0, 1
DTYPE = {torch.float32: F32, torch.bfloat16: BF16}

_LIB = os.environ.get("LX_KERNELS", "")
BACKEND = os.environ.get("LX_BACKEND") or ("metal" if "metal" in pathlib.Path(_LIB).name else "cuda")
DEVICE = "cuda" if BACKEND == "cuda" else "cpu"

_HEADERS = [pathlib.Path(__file__).resolve().parents[2] / "engine" / "cuda" / name for name in ("kernels.h", "tunable.h")]
_ENUMS = {"lx_dtype", "lx_wtype", "lx_math", "lx_copy_kind", "lx_backend"}


def _ctype(decl):
    """The ctypes type of one C parameter or return type."""
    decl = decl.strip()
    if "*" in decl:
        return ctypes.c_char_p if decl.replace(" ", "") == "constchar*" else ctypes.c_void_p
    base = decl.replace("const", "").split()[0]
    if base == "int64_t":
        return ctypes.c_int64
    if base == "float":
        return ctypes.c_float
    if base in ("int", "int32_t", "uint32_t") or base in _ENUMS:
        return ctypes.c_int
    if base == "void":
        return None
    raise ValueError(f"no ctypes mapping for {decl!r}")


def _prototypes():
    """{name: (restype, [argtypes])} for every LX_API function of the headers."""
    out = {}
    for header in _HEADERS:
        text = re.sub(r"//[^\n]*", "", header.read_text())
        for m in re.finditer(r"LX_API\s+([\w\s\*]+?)\s*\b(lx_\w+)\s*\(([^)]*)\)\s*;", text):
            ret, name, params = m.group(1), m.group(2), m.group(3).strip()
            args = []
            if params and params != "void":
                for p in params.split(","):
                    p = p.strip()
                    # Drop the parameter name: the last identifier, unless the declaration is a bare type.
                    p = re.sub(r"\b\w+\s*$", "", p) if re.search(r"[\w\*]\s+\w+\s*$", p) else p
                    args.append(_ctype(p))
            out[name] = (_ctype(ret), args)
    return out


class Kernels:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        self.backend = BACKEND
        for name, (restype, argtypes) in _prototypes().items():
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.restype = restype
                fn.argtypes = argtypes

    @property
    def stream(self):
        """The stream every call runs on: the current CUDA stream, or the backend's default stream."""
        if self.backend == "cuda":
            return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        return ctypes.c_void_p(None)

    def sync(self):
        if self.backend == "cuda":
            torch.cuda.synchronize()
        else:
            status = self.lib.lx_rt_stream_sync(None)
            assert status == 0, f"lx_rt_stream_sync returned error {status}"

    def __call__(self, name, *args):
        """Call lx_<name>(*args, stream); tensors pass as device pointers."""
        fn = getattr(self.lib, "lx_" + name)
        conv = []
        for a in args:
            if isinstance(a, torch.Tensor):
                assert a.device.type == DEVICE and a.is_contiguous(), f"kernels take contiguous {DEVICE} tensors"
                conv.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, bool):
                conv.append(int(a))
            else:
                conv.append(a)
        conv.append(self.stream)
        status = fn(*conv)
        assert status == 0, f"lx_{name} returned error {status}"
        self.sync()


def pytest_addoption(parser):
    parser.addoption(
        "--lx-arch",
        type=int,
        default=0,
        help="run the suite as this architecture (80, 86, 89, 90, ... on CUDA; 1007, 1009, ... for Apple GPU "
        "families on Metal): the kernels take their fallback paths, as they would on that GPU. Only lowers; see "
        "lx_set_arch_override.",
    )


@pytest.fixture(scope="session")
def lx(request):
    if not _LIB or not os.path.exists(_LIB):
        pytest.skip("LX_KERNELS is not set to a built kernel library")
    kernels = Kernels(_LIB)
    arch = request.config.getoption("--lx-arch")
    if arch:
        lib = kernels.lib
        real = lib.lx_set_arch_override(0)
        effective = lib.lx_set_arch_override(arch)
        assert effective == min(arch, real), f"--lx-arch={arch} did not take effect (the device is {real})"
    return kernels


def close(actual, expected, atol, rtol=0.0):
    """Assert two tensors agree; reports the worst element."""
    a, e = actual.float(), expected.float()
    err = (a - e).abs()
    bound = atol + rtol * e.abs()
    worst = (err - bound).argmax()
    assert bool((err <= bound).all()), (
        f"max abs err {err.max().item():.3e} at {worst.item()}: got {a.flatten()[worst].item():.6f}, "
        f"want {e.flatten()[worst].item():.6f}"
    )
