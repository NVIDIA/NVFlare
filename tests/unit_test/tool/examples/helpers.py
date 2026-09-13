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

import hashlib
import json
from urllib.parse import unquote, urlsplit

from nvflare.tool.examples import source

COMMIT = "a" * 40


VERSION = {"version": "2.10.0", "full-revisionid": COMMIT, "dirty": False, "error": None}


CATALOG = {
    "schema_version": 1,
    "examples": {
        "hello-pt": {
            "path": "examples/hello-world/hello-pt",
            "destination": "hello-pt",
            "nvflare": ">=2.10.0.dev0,<2.11.0.dev0",
            "extra": "PT",
            "next_command": ["python", "job.py"],
        }
    },
}


EXAMPLE_PATH = CATALOG["examples"]["hello-pt"]["path"]


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        if isinstance(self.data, BaseException):
            raise self.data
        for offset in range(0, len(self.data), chunk_size):
            yield self.data[offset : offset + chunk_size]


class Remote:
    """HTTP fixture with real blob IDs and GitHub's separate shallow/recursive tree responses."""

    def __init__(self, files=None, refs=None):
        self.files = files or {
            "examples/catalog.json": json.dumps(CATALOG).encode(),
            f"{EXAMPLE_PATH}/job.py": b"print('run only when the user asks')\n",
            f"{EXAMPLE_PATH}/README.md": b"# Hello PyTorch\n",
            f"{EXAMPLE_PATH}/data/small.csv": b"x,y\n1,2\n",
            "unrelated/secret.txt": b"must never be downloaded",
        }
        self.refs = refs or {"refs/tags/2.10.0": COMMIT, "main": COMMIT, COMMIT: COMMIT, "feature/foo": COMMIT}
        self.trees = {}
        self.calls = []
        self.overrides = {}
        self.source = source.GitHubSource()
        self.source.session.get = self.get
        self.build_tree("")

    def build_tree(self, directory):
        prefix = directory + "/" if directory else ""
        names = sorted({path[len(prefix) :].split("/")[0] for path in self.files if path.startswith(prefix)})
        items = []
        for name in names:
            path = prefix + name
            if path in self.files:
                data = self.files[path]
                items.append(
                    {"path": name, "type": "blob", "mode": "100644", "sha": source.blob_sha(data), "size": len(data)}
                )
            else:
                revision = self.build_tree(path)
                items.append({"path": name, "type": "tree", "mode": "040000", "sha": revision})
        revision = COMMIT if not directory else hashlib.sha256(directory.encode()).hexdigest()[:40]
        self.trees[revision] = items
        return revision

    def recursive(self, revision):
        items = []
        for item in self.trees[revision]:
            items.append(dict(item))
            if item["type"] == "tree":
                for child in self.recursive(item["sha"]):
                    items.append({**child, "path": item["path"] + "/" + child["path"]})
        return items

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == (10, 30)
        parts = urlsplit(url)
        path = unquote(parts.path)
        route = path.removeprefix("/repos/" + source.REPOSITORY + "/")
        if parts.hostname == "raw.githubusercontent.com":
            route = "raw/" + path.removeprefix("/" + source.REPOSITORY + "/")
        self.calls.append(route)
        if route in self.overrides:
            value = self.overrides[route]
            return value() if callable(value) else value
        if route.startswith("commits/"):
            commit = self.refs.get(route[len("commits/") :])
            return Response(commit.encode()) if commit else Response(b"not found", 404)
        if route.startswith("raw/"):
            path = route[len("raw/") + 41 :]
            data = self.files.get(path)
            return Response(data) if data is not None else Response(b"missing", 404)
        if route.startswith("git/trees/"):
            revision = route[len("git/trees/") :]
            tree = self.recursive(revision) if parts.query else self.trees[revision]
            return Response(json.dumps({"tree": tree, "truncated": False}).encode())
        raise AssertionError(f"Unexpected URL: {url}")


def get_example(cache, tmp_path, **kwargs):
    return cache.get(VERSION, name="hello-pt", destination=tmp_path / "delivered", **kwargs)
