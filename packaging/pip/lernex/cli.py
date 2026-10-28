"""The `lernex` command: runs the engine binary this package carries, with the same arguments.

The binary is exec'd, not run as a child, so this process turns into the engine: its arguments, exit status, signals
(Ctrl-C included) and terminal are the engine's from the first instruction.

The CUDA backend (liblernex_kernels.so) links the CUDA runtime and cuBLASLt by soname. Those come from the NVIDIA
wheels this package depends on, and their directories are found from the installed files, not from a fixed path.
They go first on LD_LIBRARY_PATH for the engine only. Without a GPU or driver the backend does not load or reports no
device, and the engine falls back to the CPU backend, which needs none of them.
"""

import importlib.metadata
import os
import signal
import sys

_PACKAGE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_PACKAGE, "_bin", "bin", "lernex")

# The NVIDIA distributions the CUDA backend was built against (its toolkit's cudart and cuBLASLt), and the
# shared libraries of each that the backend loads.
_NVIDIA_DISTRIBUTIONS = ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")
_NVIDIA_SONAMES = ("libcudart.so.12", "libcublasLt.so.12")


def _nvidia_library_dirs():
    """The directories holding the NVIDIA libraries above, from the files each installed distribution records."""
    dirs = []
    for name in _NVIDIA_DISTRIBUTIONS:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in dist.files or ():
            if item.name.startswith(_NVIDIA_SONAMES):
                directory = os.path.dirname(str(dist.locate_file(item)))
                if directory not in dirs:
                    dirs.append(directory)
    return dirs


def main():
    if not os.path.isfile(_ENGINE):
        sys.exit("lernex: no engine binary in this installation (expected %s); reinstall the lernex package" % _ENGINE)
    env = dict(os.environ)
    if sys.platform.startswith("linux"):
        found = _nvidia_library_dirs()
        if found:
            existing = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(found + ([existing] if existing else []))
    if hasattr(signal, "SIGPIPE"):
        # Python ignores SIGPIPE, and an exec keeps ignored signals ignored. The engine gets the default.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        os.execve(_ENGINE, ["lernex"] + sys.argv[1:], env)
    except OSError as error:
        sys.exit("lernex: could not start the engine (%s): %s" % (_ENGINE, error))
