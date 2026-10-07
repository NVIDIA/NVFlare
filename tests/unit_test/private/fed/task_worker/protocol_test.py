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

import os
from dataclasses import replace

import pytest

from nvflare.private.fed.task_worker import ContextProperty, TaskAttemptIdentity, WorkerBootstrap, protocol


@pytest.fixture
def bootstrap(tmp_path):
    return WorkerBootstrap(
        identity=TaskAttemptIdentity("job", "site", "task", "train", "attempt"),
        artifact_root=str(tmp_path / "artifacts"),
        workspace_root=str(tmp_path / "workspace"),
        executor={"path": "compute.Executor", "args": {}},
        components=({"id": "helper", "path": "compute.Helper"},),
        context_properties={"value": ContextProperty(3, private=False, sticky=True)},
    )


def test_bootstrap_round_trip_is_private_and_immutable(tmp_path, bootstrap):
    path = tmp_path / "attempt" / "bootstrap.json"
    protocol.write_bootstrap(str(path), bootstrap)
    assert protocol.read_bootstrap(str(path)) == bootstrap
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        protocol.write_bootstrap(str(path), replace(bootstrap, executor={}))
    assert protocol.read_bootstrap(str(path)) == bootstrap
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("value", [None, [], "identity", {"job_id": "job"}])
def test_identity_requires_exact_mapping(value):
    with pytest.raises((TypeError, ValueError)):
        TaskAttemptIdentity.from_dict(value)


@pytest.mark.parametrize("attempt", [".", "..", "../attempt", "nested/attempt"])
def test_identity_rejects_path_traversal(attempt):
    with pytest.raises(ValueError, match="safe path component"):
        TaskAttemptIdentity("job", "site", "task", "train", attempt)


@pytest.mark.parametrize("value", [None, "", "nul\x00value", "a" * 4097])
def test_identity_rejects_invalid_text(value):
    with pytest.raises(ValueError, match="job_id"):
        TaskAttemptIdentity(value, "site", "task", "train", "attempt")


@pytest.mark.parametrize("value", [None, [], {"value": 3}, {"value": 3, "private": 1, "sticky": False}])
def test_context_property_rejects_malformed_records(value):
    with pytest.raises((TypeError, ValueError)):
        ContextProperty.from_dict(value)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"identity": {}}, TypeError),
        ({"schema_version": 2}, ValueError),
        ({"artifact_root": "relative"}, ValueError),
        ({"workspace_root": "relative"}, ValueError),
        ({"executor": []}, TypeError),
        ({"components": "invalid"}, TypeError),
        ({"components": [None]}, TypeError),
        ({"components": [{"id": ""}]}, ValueError),
        ({"components": [{"id": "same"}, {"id": "same"}]}, ValueError),
        ({"context_properties": []}, TypeError),
        ({"context_properties": {"value": 3}}, TypeError),
        ({"state_revision": -1}, ValueError),
        ({"state_revision": True}, ValueError),
        ({"state_revision": 1.5}, ValueError),
        ({"state_records": []}, TypeError),
        ({"state_records": [("counter", {"encoding": "json", "value": 1})]}, TypeError),
        ({"state_names": ["counter"], "state_records": {"other": {"encoding": "json", "value": 1}}}, KeyError),
    ],
)
def test_bootstrap_rejects_invalid_fields(bootstrap, changes, error):
    with pytest.raises(error):
        replace(bootstrap, **changes)


@pytest.mark.parametrize("value", [None, [], {}, {"unknown": True}])
def test_bootstrap_requires_exact_top_level_mapping(value):
    with pytest.raises((TypeError, ValueError)):
        WorkerBootstrap.from_dict(value)


def test_bootstrap_rejects_invalid_context_mapping(bootstrap):
    record = bootstrap.to_dict()
    record["context_properties"] = []
    with pytest.raises(TypeError, match="context_properties"):
        WorkerBootstrap.from_dict(record)


def test_bootstrap_round_trips_declared_state_and_reads_legacy_defaults(tmp_path, bootstrap):
    stateful = replace(
        bootstrap,
        state_names=("counter",),
        state_revision=3,
        state_records={"counter": {"encoding": "json", "value": 4}},
    )
    path = str(tmp_path / "state-bootstrap.json")
    protocol.write_bootstrap(path, stateful)
    assert protocol.read_bootstrap(path) == stateful
    legacy = bootstrap.to_dict()
    for field in ("state_names", "state_revision", "state_records"):
        legacy.pop(field)
    assert WorkerBootstrap.from_dict(legacy) == bootstrap


def test_bootstrap_rejects_non_json_values_and_oversized_records(tmp_path, bootstrap, monkeypatch):
    path = str(tmp_path / "bootstrap.json")
    with pytest.raises(TypeError, match="WorkerBootstrap"):
        protocol.write_bootstrap(path, {})
    with pytest.raises(ValueError, match="JSON-compatible"):
        protocol.write_bootstrap(path, replace(bootstrap, executor={"value": object()}))
    monkeypatch.setattr(protocol, "_MAX_BOOTSTRAP_BYTES", 32)
    with pytest.raises(ValueError, match="too large"):
        protocol.write_bootstrap(path, bootstrap)
    assert not os.path.exists(path)
    (tmp_path / "bootstrap.json").write_bytes(b" " * 33)
    with pytest.raises(ValueError, match="too large"):
        protocol.read_bootstrap(path)


def test_bootstrap_rejects_non_regular_files(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        protocol.read_bootstrap(str(tmp_path))


def test_bootstrap_publish_tolerates_temporary_file_disappearing(tmp_path, bootstrap, monkeypatch):
    unlink = protocol.os.unlink

    def already_unlinked(path):
        unlink(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(protocol.os, "unlink", already_unlinked)
    path = str(tmp_path / "bootstrap.json")
    protocol.write_bootstrap(path, bootstrap)
    assert protocol.read_bootstrap(path) == bootstrap
