#!/usr/bin/env python3
# A release host for `dromos update` tests, answering the way github.com/lernex/dromos/releases does.
#
#   tests/release/serve.py DIR PORT
#
# DIR holds one directory per release, DIR/vX.Y.Z/, with the assets and SHA256SUMS in it, and DIR/LATEST, the version
# /latest leads to. Requests:
#   GET /latest                     302 to /tag/vLATEST (the redirect `dromos update` follows)
#   GET /tag/vX.Y.Z                 200, a release page (its URL names the tag)
#   GET /download/vX.Y.Z/NAME       the asset, or 404
# Binds 127.0.0.1 only. Stop it with Ctrl-C.
import http.server
import os
import re
import sys
from urllib.parse import unquote, urlparse

RELEASE_PATH = re.compile(r"^/download/(v[0-9][^/]*)/([A-Za-z0-9._-]+)$")
TAG_PATH = re.compile(r"^/tag/(v[0-9][^/]*)$")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "dromos-release-host/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: %s %s\n" % (self.command, self.path))

    def send_body(self, status, body, content_type="text/plain", headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        root = self.server.root
        if path == "/latest":
            with open(os.path.join(root, "LATEST")) as f:
                latest = f.read().strip()
            self.send_body(302, b"", headers=[("Location", "/tag/v" + latest)])
            return
        match = TAG_PATH.match(path)
        if match:
            tag = match.group(1)
            if not os.path.isdir(os.path.join(root, tag)):
                self.send_body(404, b"no such release\n")
                return
            self.send_body(200, ("<html><body>release %s</body></html>" % tag).encode(), "text/html")
            return
        match = RELEASE_PATH.match(path)
        if match:
            tag, name = match.groups()
            file = os.path.join(root, tag, name)
            if not os.path.isfile(file):
                self.send_body(404, b"no such asset\n")
                return
            with open(file, "rb") as f:
                self.send_body(200, f.read(), "application/octet-stream")
            return
        self.send_body(404, b"not found\n")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    server = http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler)
    server.root = os.path.abspath(sys.argv[1])
    sys.stderr.write("serve: releases from %s on http://127.0.0.1:%d\n" % (server.root, server.server_port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
