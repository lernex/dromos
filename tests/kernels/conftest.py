"""The kernel tests call each kernel through the C ABI (engine/cuda/kernels.h) with ctypes,
on PyTorch CUDA tensors, and compare against a PyTorch reference written beside the test.

LX_KERNELS points at build/liblernex_kernels.so (scripts/remote.sh sets it).

The same tests run against any backend library. LX_BACKEND=cpu (with LX_KERNELS pointing at
liblernex_cpu.so) runs them on CPU tensors: every tensor the tests create on "cuda" is created on the
CPU instead, torch.cuda's stream and synchronize calls become the backend's (the CPU's streams are
synchronous), and the kernels get host pointers. The tests themselves are unchanged. LX_BACKEND=metal
(liblernex_metal.dylib) runs the same way: Metal reads host memory in place (unified memory), and
every call is followed by a stream sync (torch.cuda.synchronize waits for the device). Without LX_BACKEND,
the library's name picks the backend.
"""

import ctypes
import os

import pytest
import torch

_LIB_NAME = os.path.basename(os.environ.get("LX_KERNELS", ""))
BACKEND = os.environ.get("LX_BACKEND") or ("metal" if "metal" in _LIB_NAME else "cpu" if "cpu" in _LIB_NAME else "cuda")
DEVICE = "cpu" if BACKEND in ("cpu", "metal") else "cuda"

if DEVICE == "cpu":
    from torch.overrides import TorchFunctionMode

    def _host(device):
        return "cpu" if device is not None and str(device).startswith("cuda") else device

    class _CudaOnHost(TorchFunctionMode):
        """Creates on the CPU every tensor a test asks for on "cuda"."""

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if "device" in kwargs:
                kwargs["device"] = _host(kwargs["device"])
            if func in (torch.Tensor.to, torch.Tensor.new_empty, torch.Tensor.new_zeros, torch.Tensor.new_full,
                        torch.Tensor.new_ones, torch.Tensor.new_tensor):
                args = tuple(_host(a) if isinstance(a, (str, torch.device)) else a for a in args)
            return func(*args, **kwargs)

    _sync_lib = []

    def _device_sync(*a, **k):
        """The CPU's streams are synchronous; Metal's are not, so a sync there waits for the device."""
        if BACKEND == "metal":
            if not _sync_lib:
                _sync_lib.append(ctypes.CDLL(os.environ["LX_KERNELS"]))
            assert _sync_lib[0].lx_rt_device_sync() == 0

    class _HostStream:
        cuda_stream = 0

        def synchronize(self):
            _device_sync()

    class _HostGenerator(torch.Generator):
        """torch.Generator(device="cuda") is a constructor, which the function mode does not see."""

        def __new__(cls, device=None):
            device = _host(device)
            return super().__new__(cls) if device is None else super().__new__(cls, device)

    _CudaOnHost().__enter__()
    torch.Generator = _HostGenerator
    torch.cuda.current_stream = lambda *a, **k: _HostStream()
    torch.cuda.synchronize = _device_sync
    torch.Tensor.cuda = lambda self, *a, **k: self

F32, BF16 = 0, 1
DTYPE = {torch.float32: F32, torch.bfloat16: BF16}


class Kernels:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)

    @property
    def stream(self):
        """The stream the calls run on, for tests that call the library directly."""
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def __call__(self, name, *args):
        """Call lx_<name>(*args, stream=current); tensors pass as device pointers."""
        fn = getattr(self.lib, "lx_" + name)
        fn.restype = ctypes.c_int
        conv = []
        for a in args:
            if isinstance(a, torch.Tensor):
                assert a.device.type == DEVICE and a.is_contiguous(), f"kernels take contiguous {DEVICE} tensors"
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
        if DEVICE == "cpu":
            assert self.lib.lx_rt_stream_sync(None) == 0
        else:
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
        pytest.skip("LX_KERNELS is not set to a built kernel library")
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
