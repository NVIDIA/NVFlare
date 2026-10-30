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

"""Bounded statistics sidecars do not participate in training result commits."""

import json
import os
from pathlib import Path

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.private.fed.task_worker import artifacts
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from tests.task_worker_test_utils import _identity


def _record(identity):
    return {
        "schema_version": "1.0",
        "identity": identity.to_dict(),
        "scope": "worker",
        "duration_seconds": "10",
        "resource_capacity": {
            "cpu": {"groups": [{"units": "2"}]},
            "memory": {"bytes": "1024"},
            "gpu": {"groups": []},
        },
        "cpu_consumed": {"cpu_seconds": "4"},
        "message_traffic": {"sent_bytes": "7", "received_bytes": "11"},
    }


def _store(tmp_path):
    identity = _identity("resources")
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    store.create_attempt(identity)
    return store, identity


def test_sidecar_is_separate_immutable_and_can_be_retired(tmp_path):
    store, identity = _store(tmp_path)
    assert store.read_resource_stats(identity) is None
    store.commit_result(identity, Shareable({"value": 42}))
    store.write_resource_stats(identity, _record(identity))
    with pytest.raises(FileExistsError):
        store.write_resource_stats(identity, _record(identity))
    assert store.read_resource_stats(identity) == _record(identity)
    store.remove_resource_stats(identity)
    store.remove_resource_stats(identity)
    assert store.read_resource_stats(identity) is None
    assert store.read_result(identity)[0]["value"] == 42


@pytest.mark.parametrize("change", ["identity", "version", "scope", "unknown", "cpu"])
def test_malformed_sidecar_does_not_invalidate_result(tmp_path, change):
    store, identity = _store(tmp_path)
    store.commit_result(identity, Shareable({"value": 42}))
    record = _record(identity)
    if change == "identity":
        record["identity"]["site_name"] = "another-site"
    elif change == "version":
        record["schema_version"] = "2.0"
    elif change == "scope":
        record["scope"] = "unknown"
    elif change == "unknown":
        record["private"] = "unexpected"
    else:
        record["cpu_consumed"]["cpu_seconds"] = "-1"
    path = Path(store.attempt_dir(identity), "resource_stats.json")
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, TypeError)):
        store.read_resource_stats(identity)
    assert store.read_result(identity)[0]["value"] == 42


def test_sidecar_size_bound_is_checked_before_decoding(tmp_path):
    store, identity = _store(tmp_path)
    Path(store.attempt_dir(identity), "resource_stats.json").write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded regular"):
        store.read_resource_stats(identity)


def test_sidecar_symlink_is_rejected_without_reading_target(tmp_path):
    store, identity = _store(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(_record(identity)))
    Path(store.attempt_dir(identity), "resource_stats.json").symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        store.read_resource_stats(identity)
    assert json.loads(target.read_text()) == _record(identity)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires Unix FIFOs")
def test_sidecar_fifo_is_rejected_without_blocking(tmp_path):
    store, identity = _store(tmp_path)
    os.mkfifo(Path(store.attempt_dir(identity), "resource_stats.json"))
    with pytest.raises(ValueError, match="bounded regular"):
        store.read_resource_stats(identity)


def test_read_detects_replacement_of_owned_directory(tmp_path, monkeypatch):
    store, identity = _store(tmp_path)
    store.write_resource_stats(identity, _record(identity))
    directory = Path(store.attempt_dir(identity))
    read = artifacts._read_json_at

    def replace_after_read(fd, name, limit):
        value = read(fd, name, limit)
        if name == "resource_stats.json":
            directory.rename(tmp_path / "old-attempt")
            directory.mkdir()
        return value

    monkeypatch.setattr(artifacts, "_read_json_at", replace_after_read)
    with pytest.raises(ValueError, match="replaced"):
        store.read_resource_stats(identity)


def test_interrupted_sidecar_temporary_is_exclusive_and_can_be_retired(tmp_path, monkeypatch):
    store, identity = _store(tmp_path)
    directory = Path(store.attempt_dir(identity))
    temporary = directory / ".resource_stats.tmp"
    temporary.write_bytes(b'{"schema_version":')
    with pytest.raises(FileExistsError):
        store.write_resource_stats(identity, _record(identity))
    assert temporary.read_bytes() == b'{"schema_version":'
    assert store.read_resource_stats(identity) is None

    def no_scan(*_args, **_kwargs):
        raise AssertionError("retirement must not scan the attempt directory")

    with monkeypatch.context() as context:
        context.setattr(artifacts.os, "listdir", no_scan)
        store.remove_resource_stats(identity)
    assert not temporary.exists()
    assert (directory / "identity.json").is_file()


def test_malformed_temporary_symlink_is_removed_without_following_it(tmp_path):
    store, identity = _store(tmp_path)
    outside = tmp_path / "training-artifact"
    outside.write_bytes(b"keep this result")
    temporary = Path(store.attempt_dir(identity), ".resource_stats.tmp")
    temporary.symlink_to(outside)
    store.remove_resource_stats(identity)
    assert not temporary.is_symlink()
    assert outside.read_bytes() == b"keep this result"


def test_temporary_cleanup_failure_reaches_the_budget_owner(tmp_path, monkeypatch):
    store, identity = _store(tmp_path)
    store.write_resource_stats(identity, _record(identity))
    temporary = Path(store.attempt_dir(identity), ".resource_stats.tmp")
    temporary.write_bytes(b"interrupted")
    unlink = artifacts.os.unlink

    def denied(name, *args, **kwargs):
        if name == ".resource_stats.tmp":
            raise PermissionError("cannot retire statistics temporary")
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", denied)
    with pytest.raises(PermissionError, match="cannot retire"):
        store.remove_resource_stats(identity)
    assert temporary.read_bytes() == b"interrupted"
