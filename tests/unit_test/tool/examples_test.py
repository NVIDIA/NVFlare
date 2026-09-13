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
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import requests
from filelock import FileLock

from nvflare.tool.examples import source, store

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


@pytest.fixture
def remote():
    result = Remote()
    yield result
    result.source.close()


@pytest.fixture(autouse=True)
def reset_output_mode():
    from nvflare.tool.cli_output import set_output_format

    set_output_format("txt")
    yield
    set_output_format("txt")


@pytest.fixture
def cache(tmp_path, remote):
    return store.ExampleStore(tmp_path / "cache", remote.source)


def get_example(cache, tmp_path, **kwargs):
    return cache.get(VERSION, name="hello-pt", destination=tmp_path / "delivered", **kwargs)


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


def test_release_does_not_fall_forward(cache, remote, tmp_path):
    with pytest.raises(source.ExampleError) as error:
        cache.get({**VERSION, "version": "2.10.1"}, name="hello-pt", destination=tmp_path / "out")
    assert error.value.code == "EXAMPLE_REF_NOT_FOUND"
    assert remote.calls == ["commits/refs/tags/2.10.1"]
    assert not (tmp_path / "out").exists()


def test_download_delivers_only_selected_source_and_records_provenance(cache, remote, tmp_path):
    result = get_example(cache, tmp_path)
    destination = tmp_path / "delivered"
    assert result["cache_status"] == "downloaded"
    assert result["commit"] == COMMIT
    assert result["next_command"] == ["python", "job.py"]
    assert (destination / "data/small.csv").read_bytes() == remote.files[f"{EXAMPLE_PATH}/data/small.csv"]
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == {
        "job.py",
        "README.md",
        "data/small.csv",
        source.PROVENANCE_FILE,
    }
    provenance = json.loads((destination / source.PROVENANCE_FILE).read_text())
    assert provenance["source_url"] == f"https://github.com/NVIDIA/NVFlare/tree/{COMMIT}/{EXAMPLE_PATH}"
    assert provenance["requested_ref"] == "refs/tags/2.10.0"
    assert not any("unrelated" in route for route in remote.calls)
    assert not list(tmp_path.glob(".nvflare-example-*"))
    assert not list(cache.root.glob(".download-*"))


def test_cache_hit_downloads_no_content_and_delivered_edits_do_not_change_cache(cache, remote, tmp_path):
    get_example(cache, tmp_path)
    (tmp_path / "delivered/job.py").write_text("user edit")
    remote.calls.clear()
    result = cache.get(VERSION, name="hello-pt", destination=tmp_path / "second")
    assert result["cache_status"] == "cache hit"
    assert remote.calls == ["commits/refs/tags/2.10.0"]
    assert (tmp_path / "second/job.py").read_bytes() == remote.files[f"{EXAMPLE_PATH}/job.py"]


def test_refresh_downloads_and_new_revision_has_a_separate_cache(cache, remote, tmp_path):
    get_example(cache, tmp_path)
    remote.calls.clear()
    cache.get(VERSION, name="hello-pt", destination=tmp_path / "refresh", refresh=True)
    assert any(route.startswith("raw/") for route in remote.calls)
    second_commit = "b" * 40
    remote.refs["main"] = second_commit
    remote.trees[second_commit] = remote.trees[COMMIT]
    result = cache.get(VERSION, name="hello-pt", ref="main", destination=tmp_path / "new-revision")
    assert result["commit"] == second_commit
    assert len(cache._entries()) == 2


@pytest.mark.parametrize(
    "corruption", ["bytes", "missing", "extra", "extra_directory", "symlink", "manifest", "oversize"]
)
def test_corrupt_cache_is_repaired(cache, remote, tmp_path, corruption):
    get_example(cache, tmp_path)
    entry = cache._entries()[0]
    target = entry / "payload/job.py"
    if corruption == "bytes":
        target.write_text("corrupt")
    elif corruption == "missing":
        target.unlink()
    elif corruption == "extra":
        (entry / "payload/extra.py").write_text("unexpected")
    elif corruption == "extra_directory":
        (entry / "payload/unexpected").mkdir()
    elif corruption == "symlink":
        target.unlink()
        target.symlink_to(tmp_path / "delivered/job.py")
    elif corruption == "manifest":
        (entry / "manifest.json").write_text("{}")
    elif corruption == "oversize":
        target.write_bytes(b"x" * (source.MAX_FILE_BYTES + 1))
    result = cache.get(VERSION, name="hello-pt", destination=tmp_path / "repaired")
    assert result["cache_status"] == "downloaded (cache repaired)"
    assert (tmp_path / "repaired/job.py").read_bytes() == remote.files[f"{EXAMPLE_PATH}/job.py"]


def test_corrupt_cache_failed_repair_is_distinct(cache, remote, tmp_path):
    get_example(cache, tmp_path)
    (cache._entries()[0] / "payload/job.py").write_text("broken")
    remote.overrides[f"raw/{COMMIT}/{EXAMPLE_PATH}/job.py"] = Response(b"down", 503)
    with pytest.raises(source.ExampleError) as error:
        cache.get(VERSION, name="hello-pt", destination=tmp_path / "repair")
    assert error.value.code == "EXAMPLE_CACHE_CORRUPT"
    assert not (tmp_path / "repair").exists()
    assert cache._entries() == []


def test_clear_and_retention_preserve_delivered_files(cache, remote, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_CACHE_ENTRIES", 2)
    for index, letter in enumerate("abc"):
        revision = letter * 40
        remote.refs["main"] = revision
        remote.trees[revision] = remote.trees[COMMIT]
        cache.get(VERSION, name="hello-pt", ref="main", destination=tmp_path / f"out-{index}")
    assert len(cache._entries()) == 2
    assert cache.clear()["entries_removed"] == 2
    assert cache.clear()["entries_removed"] == 0
    for index in range(3):
        assert (tmp_path / f"out-{index}/job.py").is_file()


@pytest.mark.parametrize("kind", ["file", "directory", "nonempty", "dangling-symlink"])
def test_existing_destination_is_untouched_without_network(cache, remote, tmp_path, kind):
    destination = tmp_path / "delivered"
    if kind == "file":
        destination.write_text("original")
    elif kind == "dangling-symlink":
        destination.symlink_to(tmp_path / "absent")
    else:
        destination.mkdir()
        if kind == "nonempty":
            (destination / "original").write_text("original")
    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)
    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"
    assert remote.calls == []
    assert destination.exists() or destination.is_symlink()


def test_atomic_publish_refuses_destination_created_after_validation(cache, tmp_path, monkeypatch):
    original = store._publish

    def raced_publish(staging, destination):
        if destination == tmp_path / "delivered":
            destination.mkdir()
        original(staging, destination)

    monkeypatch.setattr(store, "_publish", raced_publish)
    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)
    assert error.value.code == "EXAMPLE_DESTINATION_EXISTS"
    assert list((tmp_path / "delivered").iterdir()) == []
    assert not list(tmp_path.glob(".nvflare-example-*"))


def test_two_concurrent_gets_download_once(cache, remote, tmp_path):
    barrier = threading.Barrier(2)

    def retrieve(index):
        barrier.wait(timeout=5)
        return cache.get(VERSION, name="hello-pt", destination=tmp_path / f"parallel-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(retrieve, range(2)))
    assert sorted(result["cache_status"] for result in results) == ["cache hit", "downloaded"]
    assert remote.calls.count(f"raw/{COMMIT}/{EXAMPLE_PATH}/job.py") == 1


def test_cache_lock_timeout_is_actionable(cache, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOCK_TIMEOUT", 0.01)
    cache.root.mkdir()
    with FileLock(str(cache.root / ".lock")):
        with pytest.raises(source.ExampleError) as error:
            get_example(cache, tmp_path)
    assert error.value.code == "EXAMPLE_CACHE_BUSY"


@pytest.mark.parametrize(
    "failure,code", [(KeyboardInterrupt(), None), (requests.ConnectionError(), "EXAMPLE_NETWORK_ERROR")]
)
def test_mid_download_failure_cleans_staging(cache, remote, tmp_path, failure, code):
    remote.overrides[f"raw/{COMMIT}/{EXAMPLE_PATH}/job.py"] = Response(failure)
    with pytest.raises(KeyboardInterrupt if code is None else source.ExampleError) as error:
        get_example(cache, tmp_path)
    if code:
        assert error.value.code == code
    assert not (tmp_path / "delivered").exists()
    assert cache._entries() == []
    assert not list(cache.root.glob(".download-*"))


@pytest.mark.parametrize(
    "status,code",
    [
        (404, "EXAMPLE_REF_NOT_FOUND"),
        (403, "EXAMPLE_ACCESS_LIMITED"),
        (429, "EXAMPLE_ACCESS_LIMITED"),
        (302, "EXAMPLE_HTTP_ERROR"),
        (500, "EXAMPLE_HTTP_ERROR"),
    ],
)
def test_http_failures(cache, remote, tmp_path, status, code):
    remote.overrides["commits/refs/tags/2.10.0"] = Response(b"failure", status)
    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)
    assert error.value.code == code
    assert error.value.hint
    assert not (tmp_path / "delivered").exists()


def test_bounded_stream_rejects_excess_response(cache, remote, tmp_path):
    remote.overrides["commits/refs/tags/2.10.0"] = Response(b"x" * 129)
    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)
    assert error.value.code == "EXAMPLE_LIMIT_EXCEEDED"


def test_bad_blob_hash_fails_without_delivery(cache, remote, tmp_path):
    remote.overrides[f"raw/{COMMIT}/{EXAMPLE_PATH}/job.py"] = Response(b"tampered")
    with pytest.raises(source.ExampleError, match="does not match"):
        get_example(cache, tmp_path)
    assert not (tmp_path / "delivered").exists()


def test_unknown_name_lists_catalog_choices(cache, tmp_path):
    with pytest.raises(source.ExampleError) as error:
        cache.get(VERSION, name="unknown", destination=tmp_path / "unknown")
    assert error.value.code == "EXAMPLE_UNKNOWN"
    assert "hello-pt" in error.value.hint


def test_absent_catalog_does_not_fallback_to_another_revision(tmp_path):
    remote = Remote({"README.md": b"older release"})
    cache = store.ExampleStore(tmp_path / "cache", remote.source)
    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)
    assert error.value.code == "EXAMPLE_CATALOG_UNAVAILABLE"
    assert remote.calls.count("commits/refs/tags/2.10.0") == 1


@pytest.mark.parametrize("version", ["2.9.0", "2.11.0.dev0", "2.11.0", "unknown"])
def test_incompatible_or_unknown_package_cannot_bypass_check_with_ref(cache, tmp_path, version):
    with pytest.raises(source.ExampleError) as error:
        cache.get({**VERSION, "version": version}, name="hello-pt", ref=COMMIT, destination=tmp_path / "bad")
    assert error.value.code in {"EXAMPLE_VERSION_INCOMPATIBLE", "EXAMPLE_VERSION_UNKNOWN"}
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("ref", ["main", COMMIT, "feature/foo"])
def test_explicit_ref_is_pinned_and_catalog_checked(cache, tmp_path, ref):
    result = cache.get(VERSION, name="hello-pt", ref=ref, destination=tmp_path / "source")
    assert result["example"] == "hello-pt"
    assert result["commit"] == COMMIT


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../../escape", "a\\b", "C:/data", "a\n.py", "a/.git/config", "CON", "a."]
)
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(source.ExampleError):
        source.safe_path(path)


@pytest.mark.parametrize("kind,mode", [("blob", "120000"), ("commit", "160000"), ("blob", "100664")])
def test_unsupported_git_objects_are_rejected_before_download(cache, remote, tmp_path, kind, mode):
    tree = remote.trees[hashlib.sha256(EXAMPLE_PATH.encode()).hexdigest()[:40]]
    tree[0].update(type=kind, mode=mode)
    with pytest.raises(source.ExampleError, match="Only ordinary"):
        get_example(cache, tmp_path)
    assert not any(route.startswith(f"raw/{COMMIT}/{EXAMPLE_PATH}/") for route in remote.calls)


def test_truncated_tree_is_rejected(cache, remote, tmp_path):
    remote.overrides[f"git/trees/{COMMIT}"] = Response(b'{"tree": [], "truncated": true}')
    with pytest.raises(source.ExampleError, match="incomplete tree"):
        get_example(cache, tmp_path)


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
    "paths",
    [
        ("caf\u00e9.txt", "cafe\u0301.txt"),
        ("caf\u00e9/one.txt", "cafe\u0301/two.txt"),
        ("caf\u00e9", "cafe\u0301/nested.txt"),
    ],
    ids=["filenames", "directory-prefixes", "file-directory"],
)
def test_unicode_equivalent_paths_are_rejected_before_download(cache, remote, tmp_path, paths):
    for index, path in enumerate(paths):
        remote.files[f"{EXAMPLE_PATH}/{path}"] = f"distinct file {index}".encode()
    remote.build_tree("")

    with pytest.raises(source.ExampleError) as error:
        get_example(cache, tmp_path)

    assert error.value.code == "EXAMPLE_CONTENT_REJECTED"
    assert not any(route.startswith(f"raw/{COMMIT}/{EXAMPLE_PATH}/") for route in remote.calls)
    assert not (tmp_path / "delivered").exists()
    assert cache._entries() == []


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
    root = Path(__file__).resolve().parents[3]
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


@pytest.mark.parametrize("command", [[], ["get"], ["cache"], ["cache", "clear"]])
def test_cli_schema_has_no_network_or_mutation(monkeypatch, capsys, command):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema"])
    monkeypatch.setattr(store, "ExampleStore", lambda: pytest.fail("schema must not create a cache"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == " ".join(["nvflare", "examples", *command])
    assert schema["mutating"] is True


@pytest.mark.parametrize("command", [["bogus"], ["cache", "bogus"]])
@pytest.mark.parametrize("output_format", ["txt", "json"])
def test_cli_unknown_schema_command_is_rejected(monkeypatch, capsys, command, output_format):
    from nvflare import cli

    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--schema", "--format", output_format])
    monkeypatch.setattr(store, "ExampleStore", lambda: pytest.fail("invalid schema must not create a cache"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 4
    captured = capsys.readouterr()
    if output_format == "json":
        result = json.loads(captured.out)
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_ARGS"
        assert "command" not in result
    else:
        assert captured.out == ""
        assert "INVALID_ARGS" in captured.err


def test_cli_download_json_and_cache_clear(monkeypatch, capsys, cache, tmp_path):
    from nvflare import cli

    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("nvflare._version.get_versions", lambda: VERSION)
    monkeypatch.setattr(
        "sys.argv", ["nvflare", "examples", "get", "hello-pt", "--dest", str(tmp_path / "out"), "--format", "json"]
    )
    cli.run("nvflare")
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["commit"] == COMMIT
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "cache", "clear", "--format", "json"])
    cli.run("nvflare")
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["entries_removed"] == 1
    assert (tmp_path / "out/job.py").is_file()


@pytest.mark.parametrize(
    "failure,code,exit_code",
    [(KeyboardInterrupt(), "EXAMPLE_INTERRUPTED", 130), (OSError("disk full"), "EXAMPLE_IO_ERROR", 1)],
)
def test_cli_failure_is_structured_and_restores_signal_handler(monkeypatch, capsys, cache, failure, code, exit_code):
    from nvflare import cli

    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr(cache, "get", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", "get", "hello-pt", "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["error_code"] == code
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize(
    "command",
    [[], ["cache"], ["get"], ["get", "hello-pt", "--source", "https://github.com/NVIDIA/NVFlare/tree/main/examples/a"]],
)
def test_cli_missing_or_unsupported_arguments_fail(monkeypatch, capsys, cache, command):
    from nvflare import cli

    monkeypatch.setattr(store, "ExampleStore", lambda: cache)
    monkeypatch.setattr("sys.argv", ["nvflare", "examples", *command, "--format", "json"])
    with pytest.raises(SystemExit) as error:
        cli.run("nvflare")
    assert error.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"
