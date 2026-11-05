# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve the exported Dashboard with the same route prefix as Flask."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / "out"
PREFIX = "/nvflare-dashboard"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", PREFIX + "/")
            self.end_headers()
            return
        if path != PREFIX and not path.startswith(PREFIX + "/"):
            self.send_error(404)
            return
        relative = path[len(PREFIX) :] or "/"
        candidate = ROOT / relative.lstrip("/")
        if not candidate.suffix and candidate.with_suffix(".html").is_file():
            relative += ".html"
        self.path = relative
        super().do_GET()


if __name__ == "__main__":
    if not (ROOT / "index.html").is_file():
        raise SystemExit("Build the static Dashboard before running browser tests.")
    ThreadingHTTPServer(("127.0.0.1", 3100), Handler).serve_forever()
