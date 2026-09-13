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

import ctypes
import errno
import hashlib
import json
import os
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from filelock import FileLock

from nvflare.tool.examples import source, store
from tests.unit_test.tool.examples.helpers import COMMIT, EXAMPLE_PATH, VERSION, Remote, Response, get_example


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
    "corruption",
    [
        "bytes",
        "same_size_and_mtime",
        "missing",
        "extra",
        "extra_directory",
        "symlink",
        "manifest",
        "manifest_directory",
        "manifest_fifo",
        "oversize",
    ],
)
def test_corrupt_cache_is_repaired(cache, remote, tmp_path, monkeypatch, corruption):
    get_example(cache, tmp_path)
    entry = cache._entries()[0]
    target = entry / "payload/job.py"
    if corruption == "bytes":
        target.write_text("corrupt")
    elif corruption == "same_size_and_mtime":
        original = target.stat()
        target.write_bytes(b"x" * original.st_size)
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert target.stat().st_size == original.st_size
        assert target.stat().st_mtime_ns == original.st_mtime_ns
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
    elif corruption == "manifest_directory":
        manifest = entry / "manifest.json"
        manifest.unlink()
        manifest.mkdir()
    elif corruption == "manifest_fifo":
        manifest = entry / "manifest.json"
        manifest.unlink()
        os.mkfifo(manifest)
        read_bytes = Path.read_bytes

        def reject_fifo_read(path):
            if path == manifest:
                pytest.fail("A FIFO manifest must be rejected before reading")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", reject_fifo_read)
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


@pytest.mark.parametrize("relative_path", ["manifest.json", "payload/job.py", "payload/data"])
def test_cache_permission_error_preserves_entry(cache, remote, tmp_path, monkeypatch, relative_path):
    get_example(cache, tmp_path)
    entry = cache._entries()[0]
    protected = entry / relative_path
    original_read = Path.read_bytes
    original_scandir = os.scandir

    def read(path):
        if path == protected:
            raise PermissionError("cache permission denied")
        return original_read(path)

    def scandir(path):
        if isinstance(path, (str, bytes, os.PathLike)) and Path(path) == protected:
            raise PermissionError("cache permission denied")
        return original_scandir(path)

    remote.calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read)
        patch.setattr(os, "scandir", scandir)
        with pytest.raises(PermissionError):
            cache.get(VERSION, name="hello-pt", destination=tmp_path / "second")
    assert remote.calls == ["commits/refs/tags/2.10.0"]
    assert (entry / "payload/job.py").read_bytes() == remote.files[f"{EXAMPLE_PATH}/job.py"]
    assert not (tmp_path / "second").exists()


@pytest.mark.parametrize(
    "version,code", [("2.11.0", "EXAMPLE_VERSION_INCOMPATIBLE"), ("unknown", "EXAMPLE_VERSION_UNKNOWN")]
)
def test_cache_repair_preserves_version_error(cache, remote, tmp_path, version, code):
    get_example(cache, tmp_path)
    (cache._entries()[0] / "manifest.json").write_text("{}")
    remote.calls.clear()
    with pytest.raises(source.ExampleError) as error:
        cache.get({**VERSION, "version": version}, name="hello-pt", ref=COMMIT, destination=tmp_path / "second")
    assert error.value.code == code
    assert not any(route.startswith(f"raw/{COMMIT}/{EXAMPLE_PATH}/") for route in remote.calls)
    assert not (tmp_path / "second").exists()


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


@pytest.mark.parametrize("kind", ["direct", "symlink", "relative"])
def test_destination_inside_cache_is_rejected_before_network(cache, remote, tmp_path, monkeypatch, kind):
    cache.root.mkdir()
    destination = cache.root / "out"
    if kind == "symlink":
        alias = tmp_path / "cache-alias"
        alias.symlink_to(cache.root, target_is_directory=True)
        destination = alias / "out"
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        destination = Path("cache/out")
    with pytest.raises(source.ExampleError) as error:
        cache.get(VERSION, name="hello-pt", destination=destination)
    assert error.value.code == "INVALID_ARGS"
    assert remote.calls == []
    assert list(cache.root.iterdir()) == []


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


@pytest.mark.parametrize(
    "paths",
    [
        ("caf\u00e9.txt", "cafe\u0301.txt"),
        ("caf\u00e9/one.txt", "cafe\u0301/two.txt"),
        ("caf\u00e9", "cafe\u0301/nested.txt"),
        ("\u0390.txt", "\u03aa\u0301.txt"),
    ],
    ids=["filenames", "directory-prefixes", "file-directory", "fold-induced-decomposition"],
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


@pytest.mark.parametrize("variable", ["XDG_CACHE_HOME", "NVFLARE_EXAMPLES_CACHE_DIR"])
def test_cache_override_does_not_require_home(monkeypatch, tmp_path, variable):
    def missing_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(store.sys, "platform", "linux")
    monkeypatch.delenv("NVFLARE_EXAMPLES_CACHE_DIR", raising=False)
    monkeypatch.setenv(variable, str(tmp_path))
    monkeypatch.setattr(Path, "home", missing_home)
    expected = tmp_path / "nvflare/examples" if variable == "XDG_CACHE_HOME" else tmp_path
    assert store.default_cache_dir() == expected


def test_missing_macos_rename_symbol_is_an_os_error(monkeypatch, tmp_path):
    monkeypatch.setattr(store.sys, "platform", "darwin")
    monkeypatch.setattr(store.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(OSError) as error:
        store._publish(tmp_path / "staging", tmp_path / "destination")
    assert error.value.errno == errno.ENOTSUP


@pytest.mark.parametrize("system", ["win32", "freebsd"])
def test_unsupported_platform_is_rejected_before_loading_libc(monkeypatch, tmp_path, system):
    monkeypatch.setattr(store.sys, "platform", system)
    monkeypatch.setattr(
        store.ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("unsupported libc must not be loaded")
    )
    with pytest.raises(OSError) as error:
        store._publish(tmp_path / "staging", tmp_path / "destination")
    assert error.value.errno == errno.ENOTSUP


@pytest.mark.parametrize("architecture,number", [("x86_64", 316), ("aarch64", 276)])
@pytest.mark.parametrize("failure", [None, errno.EEXIST, errno.ENOSYS])
def test_linux_without_libc_wrapper_uses_noreplace_syscall(monkeypatch, tmp_path, architecture, number, failure):
    syscall = Mock(return_value=-1 if failure else 0)
    monkeypatch.setattr(store.sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: architecture)
    monkeypatch.setattr(store.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(syscall=syscall))
    previous_errno = ctypes.get_errno()
    try:
        ctypes.set_errno(failure or 0)
        if failure:
            with pytest.raises(OSError) as error:
                store._publish(tmp_path / "staging", tmp_path / "destination")
            assert error.value.errno == failure
        else:
            store._publish(tmp_path / "staging", tmp_path / "destination")
        syscall.assert_called_once_with(
            number, -100, os.fsencode(tmp_path / "staging"), -100, os.fsencode(tmp_path / "destination"), 1
        )
    finally:
        ctypes.set_errno(previous_errno)


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or platform.machine() not in {"x86_64", "aarch64"}
    or ctypes.sizeof(ctypes.c_void_p) != 8,
    reason="requires 64-bit x86_64 or aarch64 Linux",
)
def test_native_linux_fallback_delivers_and_refuses_existing_destination(monkeypatch, tmp_path):
    libc = ctypes.CDLL(None, use_errno=True)
    monkeypatch.setattr(store.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(syscall=libc.syscall))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "content").write_text("complete")
    destination = tmp_path / "destination"
    store._publish(staging, destination)
    assert (destination / "content").read_text() == "complete"
    assert not staging.exists()
    staging.mkdir()
    (staging / "replacement").write_text("new")
    with pytest.raises(FileExistsError):
        store._publish(staging, destination)
    assert (destination / "content").read_text() == "complete"
    assert (staging / "replacement").is_file()
    (destination / "content").unlink()
    with pytest.raises(FileExistsError):
        store._publish(staging, destination)
    assert list(destination.iterdir()) == []
