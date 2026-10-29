"""The `dromos` command: runs the engine binary this package carries, with the same arguments.

On Linux and macOS the binary is exec'd, not run as a child, so this process turns into the engine: its arguments,
exit status, signals (Ctrl-C included) and terminal are the engine's from the first instruction.

On Windows os.execve does not replace the process, so the console would not wait for the engine. There the engine runs
as a child that this process waits for, and this command exits with the engine's status. Ctrl-C reaches both, since
they share the console: this process ignores it while the engine runs, and the engine decides what it means (in
`dromos run` it stops a reply).

The CUDA backend links the CUDA runtime and cuBLASLt by soname (libdromos_kernels.so on Linux) or by DLL name
(dromos_kernels.dll on Windows). Those come from the NVIDIA wheels this package depends on, and their directories are
found from the installed files, not from a fixed path. They go first on LD_LIBRARY_PATH (Linux) or PATH (Windows) for
the engine only. Without a GPU or driver the backend does not load or reports no device, and the engine falls back to
the CPU backend, which needs none of them.
"""

import ctypes
import importlib.metadata
import os
import signal
import subprocess
import sys

_PACKAGE = os.path.dirname(os.path.abspath(__file__))
# The engine is bin/dromos, or bin/dromos.exe on Windows, where the backends sit beside it in bin/ too.
_ENGINE = os.path.join(_PACKAGE, "_bin", "bin", "dromos.exe" if sys.platform == "win32" else "dromos")

# The NVIDIA distributions the CUDA backend was built against (its toolkit's cudart and cuBLASLt), and the shared
# libraries of each that the backend loads: sonames on Linux, DLLs on Windows.
_NVIDIA_DISTRIBUTIONS = ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")
_NVIDIA_SONAMES = ("libcudart.so.12", "libcublasLt.so.12")
_NVIDIA_DLLS = ("cudart64_12.dll", "cublasLt64_12.dll")


def _nvidia_library_dirs(names):
    """The directories holding the NVIDIA libraries named in `names`, from each installed distribution's file list."""
    dirs = []
    for name in _NVIDIA_DISTRIBUTIONS:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in dist.files or ():
            if item.name.startswith(names):
                directory = os.path.dirname(str(dist.locate_file(item)))
                if directory not in dirs:
                    dirs.append(directory)
    return dirs


def _run_engine_windows(env):
    """Runs the engine as a child, with the NVIDIA DLL folders first on PATH, and returns its exit status."""
    found = _nvidia_library_dirs(_NVIDIA_DLLS)
    if found:
        existing = env.get("PATH")
        env["PATH"] = os.pathsep.join(found + ([existing] if existing else []))
    # Ctrl-C goes to every process on the console, this one included. It is ignored here, so the engine alone decides
    # what it does with it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        completed = subprocess.run([_ENGINE] + sys.argv[1:], env=env)
    except OSError as error:
        sys.exit("dromos: could not start the engine (%s): %s" % (_ENGINE, error))
    # Windows reports the status as an unsigned 32-bit value (a crash is an NTSTATUS such as 0xC0000005), and sys.exit
    # takes a C int: the signed view of the same bits is the status the engine exited with.
    return ctypes.c_int32(completed.returncode).value


def main():
    if not os.path.isfile(_ENGINE):
        sys.exit("dromos: no engine binary in this installation (expected %s); reinstall the dromos package" % _ENGINE)
    env = dict(os.environ)
    if sys.platform == "win32":
        sys.exit(_run_engine_windows(env))
    if sys.platform.startswith("linux"):
        found = _nvidia_library_dirs(_NVIDIA_SONAMES)
        if found:
            existing = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(found + ([existing] if existing else []))
    if hasattr(signal, "SIGPIPE"):
        # Python ignores SIGPIPE, and an exec keeps ignored signals ignored. The engine gets the default.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        os.execve(_ENGINE, ["dromos"] + sys.argv[1:], env)
    except OSError as error:
        sys.exit("dromos: could not start the engine (%s): %s" % (_ENGINE, error))
