"""The kernel tests call each kernel through the C ABI (engine/cuda/kernels.h) with ctypes,
on PyTorch CUDA tensors, and compare against a PyTorch reference written beside the test.

LX_KERNELS points at build/liblernex_kernels.so (scripts/remote.sh sets it).
"""

import ctypes
import os

import pytest
import torch

F32, BF16 = 0, 1
DTYPE = {torch.float32: F32, torch.bfloat16: BF16}


class Kernels:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)

    def __call__(self, name, *args):
        """Call lx_<name>(*args, stream=current); tensors pass as device pointers."""
        fn = getattr(self.lib, "lx_" + name)
        fn.restype = ctypes.c_int
        conv = []
        for a in args:
            if isinstance(a, torch.Tensor):
                assert a.is_cuda and a.is_contiguous(), "kernels take contiguous CUDA tensors"
                conv.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, bool):
                conv.append(ctypes.c_int(int(a)))
            elif isinstance(a, int):
                conv.append(ctypes.c_int64(a) if abs(a) >= 2**31 else ctypes.c_int(a))
            elif isinstance(a, float):
                conv.append(ctypes.c_float(a))
            elif a is None:
                conv.append(ctypes.c_void_p(None))
            else:
                conv.append(a)
        conv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        status = fn(*conv)
        assert status == 0, f"lx_{name} returned CUDA error {status}"
        torch.cuda.synchronize()


def pytest_addoption(parser):
    parser.addoption(
        "--lx-arch",
        type=int,
        default=0,
        help="run the suite as this architecture (80, 86, 89, 90, ...): the kernels take their fallback "
        "paths, as they would on that card. Only lowers; see lx_set_arch_override.",
    )


@pytest.fixture(scope="session")
def lx(request):
    path = os.environ.get("LX_KERNELS")
    if not path or not os.path.exists(path):
        pytest.skip("LX_KERNELS is not set to a built liblernex_kernels.so")
    kernels = Kernels(path)
    arch = request.config.getoption("--lx-arch")
    if arch:
        lib = kernels.lib
        lib.lx_set_arch_override.restype = ctypes.c_int
        lib.lx_set_arch_override.argtypes = [ctypes.c_int]
        real = lib.lx_set_arch_override(0)
        effective = lib.lx_set_arch_override(arch)
        assert effective == min(arch, real), f"--lx-arch={arch} did not take effect (the card is sm_{real})"
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
