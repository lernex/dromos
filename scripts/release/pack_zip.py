#!/usr/bin/env python3
"""Packs a release tree into a zip with no top folder (scripts/release/build.ps1), so the same tree gives the same
bytes: entries in sorted order with forward slashes, every timestamp the given epoch (the commit's time), deflated at
level 9, no directory entries and no extra fields.

    pack_zip.py TREE OUT.zip EPOCH
"""

import os
import sys
import time
import zipfile


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: pack_zip.py TREE OUT.zip EPOCH")
    tree, out, epoch = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    # Zip stores local DOS time from 1980 on; the commit's UTC time is written as is, as `tar --mtime` does.
    stamp = time.gmtime(max(epoch, 315532800))[:6]
    files = []
    for root, dirs, names in os.walk(tree):
        dirs.sort()
        for name in names:
            path = os.path.join(root, name)
            files.append((os.path.relpath(path, tree).replace(os.sep, "/"), path))
    files.sort()
    if not files:
        sys.exit("pack_zip.py: %s is empty" % tree)
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, path in files:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 0  # MS-DOS: no Unix modes for an archive Windows unpacks
            info.compress_level = 9  # Python 3.13 and later
            info._compresslevel = 9  # earlier
            info.file_size = os.path.getsize(path)  # decides Zip64 per entry, as zip -9 would
            with open(path, "rb") as f, z.open(info, "w") as dst:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
    print("pack_zip.py: %s, %d files" % (out, len(files)))


if __name__ == "__main__":
    main()
