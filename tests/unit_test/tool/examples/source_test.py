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

import copy
import hashlib
import json
from pathlib import Path

import pytest
import requests

from nvflare.tool.examples import source
from tests.unit_test.tool.examples.helpers import CATALOG, COMMIT, EXAMPLE_PATH, VERSION, Response


@pytest.mark.parametrize(
    "version,revision,expected",
    [
        ("2.10.0", COMMIT, "refs/tags/2.10.0"),
        ("2.10.1", COMMIT, "refs/tags/2.10.1"),
        ("2.10.0rc3", COMMIT, "refs/tags/2.10.0rc3"),
        ("2.10.0.dev0+21.gabcdef", COMMIT, COMMIT),
        ("2.10.0.dev260912", COMMIT, COMMIT),
        ("2.10.0+custom", COMMIT, COMMIT),
    ],
)
def test_version_selects_exact_source(version, revision, expected):
    assert source.selected_ref({**VERSION, "version": version, "full-revisionid": revision}) == expected


@pytest.mark.parametrize("revision", [None, "", "abcdef", "not-a-commit"])
def test_development_without_provenance_requires_explicit_ref(revision):
    info = {**VERSION, "version": "2.10.0.dev0", "full-revisionid": revision}
    with pytest.raises(source.ExampleError, match="no usable source revision"):
        source.selected_ref(info)
    assert source.selected_ref(info, "main") == "main"


def test_dirty_checkout_requires_explicit_remote_selection():
    with pytest.raises(source.ExampleError, match="uncommitted"):
        source.selected_ref({**VERSION, "dirty": True})
    assert source.selected_ref({**VERSION, "dirty": True}, COMMIT) == COMMIT


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../../escape", "a\\b", "C:/data", "a\n.py", "a/.git/config", "CON", "a."]
)
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(source.ExampleError):
        source.safe_path(path)


@pytest.mark.parametrize("mutation", ["case", "parent_case", "reserved", "files", "bytes", "total", "ancestor"])
def test_inventory_bounds_and_portable_collisions(remote, mutation, monkeypatch):
    items = remote.recursive(hashlib.sha256(EXAMPLE_PATH.encode()).hexdigest()[:40])
    if mutation == "case":
        items.append({**items[0], "path": items[0]["path"].swapcase()})
    elif mutation == "parent_case":
        items.append({**items[-1], "path": "DATA/other.csv"})
    elif mutation == "reserved":
        items[0]["path"] = source.PROVENANCE_FILE
    elif mutation == "files":
        monkeypatch.setattr(source, "MAX_FILES", 1)
    elif mutation == "bytes":
        items[0]["size"] = source.MAX_FILE_BYTES + 1
    elif mutation == "total":
        monkeypatch.setattr(source, "MAX_TOTAL_BYTES", 1)
    elif mutation == "ancestor":
        items.append({**items[-1], "path": "job.py/child"})
    with pytest.raises(source.ExampleError):
        source.validate_inventory(items)


@pytest.mark.parametrize(
    "field,value",
    [
        ("destination", "../outside"),
        ("path", "../outside"),
        ("nvflare", "invalid"),
        ("extra", "PT;do-something"),
        ("next_command", ["sh", "install.sh"]),
    ],
)
def test_invalid_catalog_contract(field, value):
    catalog = copy.deepcopy(CATALOG)
    catalog["examples"]["hello-pt"][field] = value
    with pytest.raises(source.ExampleError):
        source.validate_catalog(catalog)


def test_checked_in_catalog_matches_runnable_example():
    root = Path(__file__).resolve().parents[4]
    catalog = json.loads((root / source.CATALOG_PATH).read_text())
    entries = source.validate_catalog(catalog)
    for entry in entries.values():
        assert (root / entry["path"] / "job.py").is_file()
        assert (root / entry["path"] / "README.md").is_file()


def test_requests_does_not_load_netrc_credentials(monkeypatch):
    client = source.GitHubSource()
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", lambda url: pytest.fail("implicit netrc authentication"))
    prepared = client.session.prepare_request(requests.Request("GET", "https://api.github.com/repos/NVIDIA/NVFlare"))
    assert "Authorization" not in prepared.headers
    client.close()


def test_file_download_stops_at_declared_size(remote):
    consumed = []

    class OversizedResponse(Response):
        def iter_content(self, chunk_size):
            for chunk in super().iter_content(chunk_size):
                consumed.append(len(chunk))
                yield chunk

    path = f"{EXAMPLE_PATH}/README.md"
    original = remote.files[path]
    item = {"type": "blob", "mode": "100644", "sha": source.blob_sha(original), "size": len(original)}
    remote.overrides[f"raw/{COMMIT}/{path}"] = OversizedResponse(b"x" * 1024 * 1024)
    with pytest.raises(source.ExampleError) as error:
        remote.source.file(COMMIT, path, item)
    assert error.value.code == "EXAMPLE_LIMIT_EXCEEDED"
    assert sum(consumed) <= len(original) + 1
