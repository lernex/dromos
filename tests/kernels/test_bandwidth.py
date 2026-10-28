"""The bandwidth probe (kernels.h, lx_bandwidth_probe): a read-only kernel's rate is positive and not above the spec."""

import ctypes

import pytest

from conftest import BACKEND


def probe(lx, nbytes, reps=3):
    read, spec = ctypes.c_double(), ctypes.c_double()
    lx("bandwidth_probe", ctypes.c_int64(nbytes), reps, ctypes.byref(read), ctypes.byref(spec))
    return read.value, spec.value


@pytest.mark.skipif(BACKEND != "cuda", reason="the probe is CUDA's and HIP's; CPU and Metal answer not supported")
def test_probe_is_within_spec(lx):
    read, spec = probe(lx, 256 << 20)
    assert spec > 0
    assert 0 < read <= spec * 1.05, f"measured {read:.1f} GB/s against a spec of {spec:.1f} GB/s"


@pytest.mark.skipif(BACKEND != "cuda", reason="the probe is CUDA's and HIP's")
def test_probe_rejects_a_size_that_is_not_whole_words(lx):
    lib = lx.lib
    read, spec = ctypes.c_double(), ctypes.c_double()
    lib.lx_bandwidth_probe.restype = ctypes.c_int
    status = lib.lx_bandwidth_probe(ctypes.c_int64(1000), 1, ctypes.byref(read), ctypes.byref(spec), None)
    assert status != 0
