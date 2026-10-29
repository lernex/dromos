#!/usr/bin/env python3
"""Builds one dromos wheel from a prebuilt tree (docs/RELEASE.md): the release tarball's or zip's contents, or the
output of `cmake --install`. Nothing here compiles; the binaries are the caller's, byte for byte.

    build_wheel.py --tree DIR --out DIR --version X.Y.Z --plat TAG [--cuda 12]

--plat is the wheel's platform tag. The caller picks it, because the build decides it: a manylinux tag for the glibc
the Linux binaries need (auditwheel show), macosx_14_0_arm64 for the Metal build, win_amd64 for the Windows tree (the
release zip, unpacked). --cuda names the CUDA toolkit major the CUDA backend was built against; the NVIDIA wheels it
needs come from CUDA_WHEELS below.

The tree is copied into a staged copy of this directory as dromos/_bin, with share/dromos/channel set to "pip" (so
`dromos update` names pip). The NVIDIA runtime libraries are never bundled: a tree's libcudart and libcublasLt (the
cudart64 and cublasLt64 DLLs on Windows) are left out, and the wheel depends on the NVIDIA wheels that carry them.
"""

import argparse
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The NVIDIA wheels per CUDA major: the distribution names and the exact versions the toolkit build used. The
# requirement is that version up to the next major, so a user gets the same ABI and any later 13.x. Both also publish
# win_amd64 wheels at these versions, so the Windows wheel takes the same pins.
CUDA_WHEELS = {
    "12": [("nvidia-cuda-runtime-cu12", "12.8.90"), ("nvidia-cublas-cu12", "12.8.5.5")],
}

# Files a tree may carry that belong to the NVIDIA wheels, not to the wheel: the shared libraries in lib/ on Linux and
# the DLLs beside the engine in bin/ on Windows (matched lowercased, since Windows file names are case-insensitive).
NVIDIA_PREFIXES = ("libcudart", "libcublas", "libnvrtc", "libnvJitLink", "libcurand", "libcufft")
NVIDIA_DLL_PREFIXES = ("cudart64", "cublas64", "cublaslt64", "nvrtc", "nvjitlink", "curand64", "cufft64")

# A 64-bit Windows image: the COFF machine of x64, and the optional-header magic that marks PE32+.
PE_MACHINE_AMD64 = 0x8664
PE_OPTIONAL_PE32_PLUS = 0x20B


def fail(message):
    sys.exit("build_wheel: " + message)


def is_windows(plat):
    return plat.startswith("win_")


def backend_dir(plat):
    """Where a tree keeps its backend libraries: beside the engine in bin/ on Windows, in lib/ elsewhere."""
    return "bin" if is_windows(plat) else "lib"


def is_nvidia_library(name, plat):
    if is_windows(plat):
        return name.lower().startswith(NVIDIA_DLL_PREFIXES)
    return name.startswith(NVIDIA_PREFIXES)


def pe_header(path):
    """The machine and optional-header magic of a PE image, read from its headers, or None if it is not a PE image."""
    with open(path, "rb") as f:
        dos = f.read(64)
        if len(dos) < 64 or dos[:2] != b"MZ":
            return None
        (offset,) = struct.unpack_from("<I", dos, 0x3C)
        f.seek(offset)
        head = f.read(26)
    if len(head) < 26 or head[:4] != b"PE\0\0":
        return None
    (machine,) = struct.unpack_from("<H", head, 4)
    (magic,) = struct.unpack_from("<H", head, 24)
    return machine, magic


def engine_binaries(tree, plat):
    """The files this package ships that the platform floor applies to: the engine and its backends."""
    found = []
    for root, _, names in os.walk(tree):
        for name in names:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            if is_windows(plat):
                ours = name == "dromos.exe" or (name.startswith("dromos_") and name.endswith(".dll"))
            else:
                ours = name == "dromos" or name.startswith("libdromos_")
            if ours:
                found.append(path)
    return sorted(found)


def macos_targets(otool_output):
    """The deployment targets (major, minor) an otool -l listing declares: LC_BUILD_VERSION's minos and the older
    LC_VERSION_MIN_MACOSX's version. A dylib's own versions are not targets and are not read."""
    targets = []
    mode = None
    for line in otool_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            command = stripped[4:]
            mode = command if command in ("LC_BUILD_VERSION", "LC_VERSION_MIN_MACOSX") else None
        elif mode == "LC_BUILD_VERSION" and stripped.startswith("minos "):
            targets.append(stripped.split()[1])
        elif mode == "LC_VERSION_MIN_MACOSX" and stripped.startswith("version "):
            targets.append(stripped.split()[1])
    parsed = []
    for text in targets:
        match = re.match(r"(\d+)\.(\d+)", text)
        if match:
            parsed.append((int(match.group(1)), int(match.group(2))))
    return parsed


def floor_check(tree, plat):
    """The binaries must need no newer glibc than the manylinux tag claims, or no newer macOS than the macosx tag.
    A Windows tree has no such floor: its images must be 64-bit AMD64 PE files."""
    manylinux = re.match(r"manylinux_(\d+)_(\d+)_", plat)
    macos = re.match(r"macosx_(\d+)_(\d+)_", plat)
    if manylinux:
        claimed = (int(manylinux.group(1)), int(manylinux.group(2)))
        worst = (0, 0)
        for path in engine_binaries(tree, plat):
            out = subprocess.run(["objdump", "-T", path], capture_output=True, text=True, check=True).stdout
            for major, minor in re.findall(r"GLIBC_(\d+)\.(\d+)", out):
                worst = max(worst, (int(major), int(minor)))
        if worst > claimed:
            fail("the binaries need glibc %d.%d, newer than the %s tag claims" % (worst[0], worst[1], plat))
        print("build_wheel: glibc floor %d.%d (tag %s)" % (worst[0], worst[1], plat))
    elif macos:
        claimed = (int(macos.group(1)), int(macos.group(2)))
        worst = (0, 0)
        for path in engine_binaries(tree, plat):
            out = subprocess.run(["otool", "-l", path], capture_output=True, text=True, check=True).stdout
            for target in macos_targets(out):
                worst = max(worst, target)
        if worst > claimed:
            fail("the binaries target macOS %d.%d, newer than the %s tag claims" % (worst[0], worst[1], plat))
        print("build_wheel: macOS target %d.%d (tag %s)" % (worst[0], worst[1], plat))
    elif is_windows(plat):
        if plat != "win_amd64":
            fail("the Windows tree is built for win_amd64, not %s" % plat)
        images = engine_binaries(tree, plat)
        for path in images:
            name = os.path.relpath(path, tree)
            header = pe_header(path)
            if header is None:
                fail("%s is not a PE image" % name)
            if header != (PE_MACHINE_AMD64, PE_OPTIONAL_PE32_PLUS):
                fail("%s is machine 0x%04x, optional header 0x%04x: not a PE32+ AMD64 image (tag %s)"
                     % (name, header[0], header[1], plat))
        print("build_wheel: %d Windows images are PE32+ AMD64 (tag %s)" % (len(images), plat))


def requirements(cuda, plat):
    if cuda is None:
        return []
    if cuda not in CUDA_WHEELS:
        fail("no NVIDIA wheel versions are recorded for CUDA %s (CUDA_WHEELS in %s)" % (cuda, __file__))
    if is_windows(plat):
        marker = 'sys_platform == "win32" and platform_machine == "AMD64"'
    else:
        marker = 'sys_platform == "linux" and platform_machine == "x86_64"'
    out = []
    for name, version in CUDA_WHEELS[cuda]:
        major = version.split(".")[0]
        out.append('%s>=%s,<%d; %s' % (name, version, int(major) + 1, marker))
    return out


def check_tree(tree, cuda, plat):
    windows = is_windows(plat)
    engine_name = "dromos.exe" if windows else "dromos"
    engine = os.path.join(tree, "bin", engine_name)
    if not os.path.isfile(engine):
        fail("no bin/%s under %s" % (engine_name, tree))
    # A Windows executable has no mode bits to check: the .exe name is what makes it runnable.
    if not windows and not os.access(engine, os.X_OK):
        fail("bin/dromos is not executable")
    backends = os.path.join(tree, backend_dir(plat))
    kernels = "dromos_kernels.dll" if windows else "libdromos_kernels.so"
    has_kernels = os.path.isfile(os.path.join(backends, kernels))
    if has_kernels and cuda is None:
        fail("the tree has the CUDA backend (%s): pass --cuda MAJOR" % kernels)
    if cuda is not None and not has_kernels:
        fail("--cuda given, but the tree has no %s" % kernels)
    for name in ("LICENSE", "NOTICE"):
        if not os.path.isfile(os.path.join(tree, name)):
            fail("the tree has no %s (build it with scripts/release/build.sh)" % name)
    if not os.path.isdir(os.path.join(tree, "share", "dromos", "licenses")):
        fail("the tree has no share/dromos/licenses (build it with scripts/release/build.sh)")
    if windows:
        if not os.path.isfile(os.path.join(backends, "dromos_cpu.dll")):
            fail("the tree has no CPU backend library (bin/dromos_cpu.dll)")
    elif not os.path.isfile(os.path.join(backends, "libdromos_cpu.so")) and not os.path.isfile(
            os.path.join(backends, "libdromos_metal.dylib")):
        fail("the tree has no CPU or Metal backend library")


def stage_tree(tree, dest, plat):
    backends = os.path.join(tree, backend_dir(plat))
    shutil.copytree(tree, dest, symlinks=True, ignore=lambda d, names: [
        n for n in names if d == backends and is_nvidia_library(n, plat)])
    left_out = []
    if os.path.isdir(backends):
        left_out = sorted(n for n in os.listdir(backends) if is_nvidia_library(n, plat))
    if left_out:
        print("build_wheel: left out of the wheel (from the NVIDIA wheels instead): " + ", ".join(left_out))
    channel = os.path.join(dest, "share", "dromos")
    os.makedirs(channel, exist_ok=True)
    with open(os.path.join(channel, "channel"), "w", encoding="utf-8") as f:
        f.write("pip\n")
    # The wheel bundles no NVIDIA library, so the NVIDIA license that came with the toolkit's libraries is not its
    # to carry: the NVIDIA wheels ship their own.
    licenses = os.path.join(dest, "share", "dromos", "licenses")
    for name in sorted(os.listdir(licenses)) if os.path.isdir(licenses) else []:
        if name.startswith("NVIDIA"):
            os.remove(os.path.join(licenses, name))
            print("build_wheel: left out of the wheel (the NVIDIA wheels carry their own license): " + name)
    # The launcher's own mode bits come from the tree; make sure the engine is executable after staging.
    engine = os.path.join(dest, "bin", "dromos.exe" if is_windows(plat) else "dromos")
    os.chmod(engine, os.stat(engine).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def stage_licenses(dest, stage):
    """The license files the wheel's metadata names (pyproject.toml license-files): LICENSE and NOTICE at the
    top of the package directory, and the third-party licenses beside them, copied from the tree's own copies."""
    shutil.copy2(os.path.join(dest, "LICENSE"), os.path.join(stage, "LICENSE"))
    shutil.copy2(os.path.join(dest, "NOTICE"), os.path.join(stage, "NOTICE"))
    os.makedirs(os.path.join(stage, "licenses"), exist_ok=True)
    source = os.path.join(dest, "share", "dromos", "licenses")
    for name in sorted(os.listdir(source)):
        shutil.copy2(os.path.join(source, name), os.path.join(stage, "licenses", name))


def build(args):
    tree = os.path.abspath(args.tree)
    out = os.path.abspath(args.out)
    check_tree(tree, args.cuda, args.plat)
    floor_check(tree, args.plat)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dromos-wheel-") as work:
        stage = os.path.join(work, "pip")
        shutil.copytree(HERE, stage, ignore=shutil.ignore_patterns("build_wheel.py", "__pycache__"))
        stage_tree(tree, os.path.join(stage, "dromos", "_bin"), args.plat)
        stage_licenses(os.path.join(stage, "dromos", "_bin"), stage)
        with open(os.path.join(stage, "wheel.json"), "w", encoding="utf-8") as f:
            json.dump({"version": args.version, "requires": requirements(args.cuda, args.plat)}, f, indent=1)
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
    parser.add_argument("--plat", required=True, help="the wheel's platform tag, e.g. manylinux_2_28_x86_64, win_amd64")
    parser.add_argument("--cuda", default=None, help="the CUDA toolkit major of the CUDA backend: 12")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
