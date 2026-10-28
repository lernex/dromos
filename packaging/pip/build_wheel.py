#!/usr/bin/env python3
"""Builds one lernex wheel from a prebuilt tree (docs/RELEASE.md): the release tarball's contents, or the output of
`cmake --install`. Nothing here compiles; the binaries are the caller's, byte for byte.

    build_wheel.py --tree DIR --out DIR --version X.Y.Z --plat TAG [--cuda 13]

--plat is the wheel's platform tag. The caller picks it, because the build decides it: a manylinux tag for the glibc
the Linux binaries need (auditwheel show), macosx_14_0_arm64 for the Metal build. --cuda names the CUDA toolkit major
the CUDA backend was built against; the NVIDIA wheels it needs come from CUDA_WHEELS below.

The tree is copied into a staged copy of this directory as lernex/_bin, with share/lernex/channel set to "pip" (so
`lernex update` names pip). The NVIDIA runtime libraries are never bundled: a tree's libcudart and libcublasLt are
left out, and the wheel depends on the NVIDIA wheels that carry them instead.
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The NVIDIA wheels per CUDA major: the distribution names and the exact versions the toolkit build used. The
# requirement is that version up to the next major, so a user gets the same ABI and any later 13.x.
CUDA_WHEELS = {
    "13": [("nvidia-cuda-runtime", "13.4.92"), ("nvidia-cublas", "13.8.0.4")],
}

# Files a tree may carry that belong to the NVIDIA wheels, not to the wheel.
NVIDIA_PREFIXES = ("libcudart", "libcublas", "libnvrtc", "libnvJitLink", "libcurand", "libcufft")


def fail(message):
    sys.exit("build_wheel: " + message)


def requirements(cuda):
    if cuda is None:
        return []
    if cuda not in CUDA_WHEELS:
        fail("no NVIDIA wheel versions are recorded for CUDA %s (CUDA_WHEELS in %s)" % (cuda, __file__))
    marker = 'sys_platform == "linux" and platform_machine == "x86_64"'
    out = []
    for name, version in CUDA_WHEELS[cuda]:
        major = version.split(".")[0]
        out.append('%s>=%s,<%d; %s' % (name, version, int(major) + 1, marker))
    return out


def check_tree(tree, cuda):
    engine = os.path.join(tree, "bin", "lernex")
    if not os.path.isfile(engine):
        fail("no bin/lernex under %s" % tree)
    if not os.access(engine, os.X_OK):
        fail("bin/lernex is not executable")
    lib = os.path.join(tree, "lib")
    has_kernels = os.path.isfile(os.path.join(lib, "liblernex_kernels.so"))
    if has_kernels and cuda is None:
        fail("the tree has the CUDA backend (liblernex_kernels.so): pass --cuda MAJOR")
    if cuda is not None and not has_kernels:
        fail("--cuda given, but the tree has no liblernex_kernels.so")
    if not os.path.isfile(os.path.join(lib, "liblernex_cpu.so")) and not os.path.isfile(
            os.path.join(lib, "liblernex_metal.dylib")):
        fail("the tree has no CPU or Metal backend library")


def stage_tree(tree, dest):
    lib_src = os.path.join(tree, "lib")
    shutil.copytree(tree, dest, symlinks=True, ignore=lambda d, names: [
        n for n in names if d == lib_src and n.startswith(NVIDIA_PREFIXES)])
    left_out = sorted(n for n in os.listdir(lib_src) if n.startswith(NVIDIA_PREFIXES)) if os.path.isdir(lib_src) else []
    if left_out:
        print("build_wheel: left out of the wheel (from the NVIDIA wheels instead): " + ", ".join(left_out))
    channel = os.path.join(dest, "share", "lernex")
    os.makedirs(channel, exist_ok=True)
    with open(os.path.join(channel, "channel"), "w", encoding="utf-8") as f:
        f.write("pip\n")
    # The launcher's own mode bits come from the tree; make sure the engine is executable after staging.
    engine = os.path.join(dest, "bin", "lernex")
    os.chmod(engine, os.stat(engine).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build(args):
    tree = os.path.abspath(args.tree)
    out = os.path.abspath(args.out)
    check_tree(tree, args.cuda)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lernex-wheel-") as work:
        stage = os.path.join(work, "pip")
        shutil.copytree(HERE, stage, ignore=shutil.ignore_patterns("build_wheel.py", "__pycache__"))
        stage_tree(tree, os.path.join(stage, "lernex", "_bin"))
        with open(os.path.join(stage, "wheel.json"), "w", encoding="utf-8") as f:
            json.dump({"version": args.version, "requires": requirements(args.cuda)}, f, indent=1)
        # The platform tag goes in setup.cfg, so pip builds the wheel with it (no setup.py command line).
        with open(os.path.join(stage, "setup.cfg"), "w", encoding="utf-8") as f:
            f.write("[bdist_wheel]\npython_tag = py3\nplat_name = %s\n" % args.plat)
        dist = os.path.join(work, "dist")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", dist, "."], cwd=stage, check=True)
        wheels = [n for n in os.listdir(dist) if n.endswith(".whl")]
        if len(wheels) != 1:
            fail("expected one wheel, got %s" % wheels)
        shutil.copy2(os.path.join(dist, wheels[0]), out)
        print(os.path.join(out, wheels[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, help="the prebuilt tree (bin/, lib/, share/)")
    parser.add_argument("--out", required=True, help="the directory the wheel is written to")
    parser.add_argument("--version", required=True, help="the PEP 440 version, e.g. 0.1.0")
    parser.add_argument("--plat", required=True, help="the wheel's platform tag, e.g. manylinux_2_28_x86_64")
    parser.add_argument("--cuda", default=None, help="the CUDA toolkit major of the CUDA backend: 13")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
