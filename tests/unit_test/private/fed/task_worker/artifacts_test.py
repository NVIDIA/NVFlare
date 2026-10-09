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
import os
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.fed.task_worker import (
    FileTaskArtifactStore,
    IncompleteTaskArtifactError,
    TaskAttemptIdentity,
    artifacts,
)
from nvflare.private.fed.task_worker.artifacts import ArtifactReference, TaskCompletion


def _identity(attempt_id="attempt-1"):
    return TaskAttemptIdentity(
        job_id="job-1",
        site_name="site-1",
        task_id="task-1",
        task_name="train",
        attempt_id=attempt_id,
    )


def test_result_is_not_readable_until_completion_is_installed(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    attempt_dir = store.create_attempt(identity)
    store.write_input(identity, Shareable({"input": 1}))

    # A payload left by an interrupted worker is not a committed result.
    (tmp_path / "artifacts" / identity.attempt_id / "result.fobs").write_bytes(b"partial")
    with pytest.raises(IncompleteTaskArtifactError, match="completion.json"):
        store.read_result(identity)

    assert store.read_input(identity)["input"] == 1
    assert os.path.isdir(attempt_dir)


def test_round_trip_binds_complete_attempt_identity_and_survives_writer(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    store.create_attempt(identity)
    store.write_input(identity, Shareable({"input": 1}))
    completion = store.commit_result(
        identity,
        Shareable({"output": 2}),
        worker_pid=123,
        worker_ppid=45,
        started_at=100.0,
        completed_at=101.0,
        diagnostics={"phase": "finalized"},
    )

    result, restored_completion = FileTaskArtifactStore(store.root_dir).read_result(identity)

    assert result["output"] == 2
    assert restored_completion == completion
    assert completion.identity == identity
    assert completion.worker_pid == 123
    assert completion.worker_ppid == 45
    assert completion.started_at == 100.0
    assert completion.completed_at == 101.0
    assert completion.diagnostics == {"phase": "finalized"}

    stale_identity = TaskAttemptIdentity(
        job_id=identity.job_id,
        site_name="other-site",
        task_id=identity.task_id,
        task_name=identity.task_name,
        attempt_id=identity.attempt_id,
    )
    with pytest.raises(ValueError, match="stale"):
        store.read_result(stale_identity)


@pytest.mark.parametrize("fault", ["tampered_payload", "partial_completion", "wrong_attempt"])
def test_result_rejects_tampered_partial_and_stale_commits(tmp_path, fault):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    store.create_attempt(identity)
    store.commit_result(identity, Shareable({"output": 2}))
    completion_path = tmp_path / "artifacts" / identity.attempt_id / "completion.json"

    expected_identity = identity
    if fault == "tampered_payload":
        with open(tmp_path / "artifacts" / identity.attempt_id / "result.fobs", "ab") as stream:
            stream.write(b"tampered")
        expected_error = (ValueError, "checksum")
    elif fault == "partial_completion":
        completion_path.write_text("{")
        expected_error = (json.JSONDecodeError, None)
    else:
        expected_identity = _identity("attempt-2")
        os.rename(store.attempt_dir(identity), store.attempt_dir(expected_identity))
        expected_error = (ValueError, "stale")

    with pytest.raises(expected_error[0], match=expected_error[1]):
        store.read_result(expected_identity)


def test_committed_artifacts_are_immutable_and_cleanup_retains_diagnostics(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    attempt_dir = store.create_attempt(identity)
    store.write_input(identity, Shareable({"input": 1}))
    store.commit_result(identity, Shareable({"output": 2}))

    with pytest.raises(FileExistsError):
        store.commit_result(identity, Shareable({"output": 3}))
    assert os.path.exists(attempt_dir)

    store.release_payloads(identity)
    assert store.read_completion(identity).identity == identity
    assert not os.path.exists(os.path.join(attempt_dir, "result.fobs"))

    store.remove_attempt(identity)
    assert not os.path.exists(attempt_dir)


def test_lazy_or_pass_through_input_is_not_a_durable_handoff(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    store.create_attempt(identity)
    data = Shareable()
    data.set_header(ReservedHeaderKey.PASS_THROUGH, True)

    with pytest.raises(ValueError, match="eager"):
        store.write_input(identity, data)


@pytest.mark.parametrize("data", [{}, Shareable({"nested": [LazyDownloadRef("server", "batch", "item")]})])
def test_artifact_requires_eager_shareable(data):
    with pytest.raises((TypeError, ValueError)):
        artifacts.require_eager_shareable(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "unknown"},
        {"file_name": "../result.fobs"},
        {"size": -1},
        {"size": "1"},
        {"sha256": "short"},
        {"sha256": "z" * 64},
    ],
)
def test_artifact_reference_rejects_invalid_fields(changes):
    values = {"kind": "result", "file_name": "result.fobs", "size": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        ArtifactReference(**(values | changes))


@pytest.mark.parametrize("record", [None, [], {}, {"unknown": True}])
def test_artifact_records_require_exact_fields(record):
    with pytest.raises(ValueError):
        ArtifactReference.from_dict(record)
    with pytest.raises(ValueError):
        TaskCompletion.from_dict(record)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"schema_version": 2}, ValueError),
        ({"identity": {}}, TypeError),
        ({"result": {}}, TypeError),
        ({"worker_pid": 0}, ValueError),
        ({"worker_ppid": -1}, ValueError),
        ({"started_at": "now"}, TypeError),
        ({"completed_at": "now"}, TypeError),
        ({"diagnostics": []}, TypeError),
    ],
)
def test_completion_rejects_invalid_fields(changes, error):
    completion = TaskCompletion(_identity(), ArtifactReference("result", "result.fobs", 0, "0" * 64), 1, 0, 1, 2)
    with pytest.raises(error):
        replace(completion, **changes)


@pytest.mark.parametrize("root", [None, "relative"])
def test_artifact_store_requires_absolute_root(root):
    with pytest.raises(ValueError, match="absolute path"):
        FileTaskArtifactStore(root)


def test_artifact_store_rejects_invalid_identity_and_escape(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = FileTaskArtifactStore(str(root))
    with pytest.raises(TypeError, match="TaskAttemptIdentity"):
        store.attempt_dir({})
    (root / "attempt-1").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.attempt_dir(_identity())


def test_artifact_record_limits_and_regular_file_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_MAX_RECORD_BYTES", 16)
    with pytest.raises(ValueError, match="too large"):
        artifacts._write_json_exclusive(str(tmp_path), "record.json", {"value": "x" * 16})
    path = tmp_path / "record.json"
    path.write_bytes(b" " * 17)
    with pytest.raises(ValueError, match="too large"):
        artifacts._read_json(str(path))
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        artifacts._read_json(str(path))
    with pytest.raises(ValueError, match="regular file"):
        artifacts._read_json(str(tmp_path))


@pytest.mark.parametrize("fault", ["unknown_field", "schema", "kind", "stale"])
def test_input_manifest_is_strictly_validated(tmp_path, fault):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    store.write_input(identity, Shareable())
    path = tmp_path / identity.attempt_id / "input.json"
    record = json.loads(path.read_text())
    if fault == "unknown_field":
        record["extra"] = 1
    elif fault == "schema":
        record["schema_version"] = 2
    elif fault == "kind":
        record["artifact"].update(kind="result", file_name="result.fobs")
    else:
        record["identity"]["site_name"] = "another-site"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        store.read_input(identity)


def test_missing_attempt_and_invalid_staged_results_are_rejected(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    with pytest.raises(ValueError, match="missing or invalid"):
        store.write_input(identity, Shareable())
    with pytest.raises(ValueError, match="missing or invalid"):
        store.release_payloads(identity)
    store.create_attempt(identity)
    reference = store.write_input(identity, Shareable())
    for invalid in ({}, reference):
        with pytest.raises(ValueError, match="result artifact"):
            store.commit_staged_result(identity, invalid)
    staged = store.stage_result(identity, Shareable())
    with pytest.raises(ValueError, match="JSON-compatible"):
        store.commit_staged_result(identity, staged, diagnostics={"bad": object()})
    assert not (tmp_path / identity.attempt_id / "completion.json").exists()


@pytest.mark.parametrize("records", [[{"key": "loss", "value": 0.5}], "not-a-list", [None]])
def test_analytics_records_are_validated_on_read(tmp_path, records):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    reference = store.write_analytics(identity, records)
    completion = store.commit_result(identity, Shareable(), diagnostics={"analytics": reference.to_dict()})
    if isinstance(records, list) and all(isinstance(record, dict) for record in records):
        assert store.read_analytics(identity, completion) == records
    else:
        with pytest.raises(ValueError, match="list of records"):
            store.read_analytics(identity, completion)
    with pytest.raises(ValueError, match="wrong artifact kind"):
        store.read_analytics(identity, replace(completion, diagnostics={"analytics": completion.result.to_dict()}))


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_cleanup_refuses_non_regular_payloads(tmp_path, kind):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    directory = store.create_attempt(identity)
    path = tmp_path / identity.attempt_id / "input.fobs"
    if kind == "directory":
        path.mkdir()
    else:
        target = tmp_path / "precious"
        target.write_text("retain")
        path.symlink_to(target)
    with pytest.raises(ValueError, match="invalid attempt artifact"):
        store.release_payloads(identity)
    assert os.path.isdir(directory)
    if kind == "symlink":
        assert target.read_text() == "retain"


def test_cleanup_refuses_attempt_directory_redirected_to_another_owned_root_entry(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    target = _identity("target-attempt")
    store.create_attempt(target)
    store.write_input(target, Shareable({"precious": 1}))
    alias = _identity("alias-attempt")
    (tmp_path / alias.attempt_id).symlink_to(store.attempt_dir(target), target_is_directory=True)
    with pytest.raises(ValueError, match="missing or invalid"):
        store.release_payloads(alias)
    assert store.read_input(target)["precious"] == 1


def test_artifact_publication_tolerates_disappearing_temporary_file(tmp_path, monkeypatch):
    unlink = artifacts.os.unlink

    def already_unlinked(path, **kwargs):
        unlink(path, **kwargs)
        raise FileNotFoundError(path)

    monkeypatch.setattr(artifacts.os, "unlink", already_unlinked)
    artifacts._write_json_exclusive(str(tmp_path), "record.json", {"value": 1})
    assert artifacts._read_json(str(tmp_path / "record.json")) == {"value": 1}


def test_attempt_removal_refuses_directory_replaced_by_symlink_after_resolution(tmp_path, monkeypatch):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    directory = store.create_attempt(identity)
    marker = tmp_path / identity.attempt_id / "retain"
    marker.write_text("evidence")
    resolve = store.attempt_dir
    moved = tmp_path / "moved"

    def replace_after_resolution(attempt):
        path = resolve(attempt)
        os.rename(path, moved)
        os.symlink(moved, path)
        return path

    monkeypatch.setattr(store, "attempt_dir", replace_after_resolution)
    with pytest.raises(ValueError, match="symlinked attempt"):
        store.remove_attempt(identity)
    assert os.path.islink(directory)
    assert (moved / "retain").read_text() == "evidence"


@pytest.mark.parametrize("operation", ["write_input", "stage_result", "claim_worker", "remove_attempt"])
def test_stale_identity_cannot_write_claim_or_delete_attempt(tmp_path, operation):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    stale = replace(identity, task_id="other-task")
    args = (stale, Shareable()) if operation in ("write_input", "stage_result") else (stale,)
    with pytest.raises(ValueError, match="stale"):
        getattr(store, operation)(*args)
    assert os.path.isdir(store.attempt_dir(identity))


def test_payload_bound_is_enforced_before_publication(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path), max_payload_bytes=64)
    identity = _identity()
    store.create_attempt(identity)
    with pytest.raises(ValueError, match="too large"):
        store.stage_result(identity, Shareable({"large": b"x" * 1000}))
    assert not (tmp_path / identity.attempt_id / "result.fobs").exists()
    assert not list((tmp_path / identity.attempt_id).glob(".result*"))


def test_tampered_staged_result_cannot_install_completion(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    reference = store.stage_result(identity, Shareable({"result": 1}))
    path = tmp_path / identity.attempt_id / "result.fobs"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.commit_staged_result(identity, reference)
    with pytest.raises(IncompleteTaskArtifactError):
        store.read_completion(identity)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_completion_rejects_nonfinite_or_boolean_timestamps(value):
    with pytest.raises(ValueError):
        TaskCompletion(_identity(), ArtifactReference("result", "result.fobs", 0, "0" * 64), 1, 0, value, 2)


def test_attempt_claim_is_exclusive_and_released_payloads_cannot_be_reexecuted(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    store.claim_worker(identity)
    with pytest.raises(FileExistsError):
        store.claim_worker(identity)
    store.write_input(identity, Shareable())
    store.commit_result(identity, Shareable())
    store.release_payloads(identity)
    with pytest.raises(FileExistsError):
        store.claim_worker(identity)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "64"])
def test_artifact_store_rejects_invalid_payload_policy(tmp_path, limit):
    with pytest.raises(ValueError, match="max_payload_bytes"):
        FileTaskArtifactStore(str(tmp_path), max_payload_bytes=limit)


def test_payload_policy_applies_to_input_and_result_reads(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    store.write_input(identity, Shareable({"value": "x" * 1024}))
    store.commit_result(identity, Shareable({"value": "x" * 1024}))
    limited = FileTaskArtifactStore(str(tmp_path), max_payload_bytes=64)
    for read in (limited.read_input, limited.read_result):
        with pytest.raises(ValueError, match="too large"):
            read(identity)
    assert store.read_result(identity)[0]["value"] == "x" * 1024


def test_payload_reference_can_represent_large_models():
    reference = ArtifactReference("result", "result.fobs", 14 * 1024**3, "0" * 64)
    assert ArtifactReference.from_dict(reference.to_dict()) == reference


def test_streaming_checksum_matches_serialized_file_and_staging_needs_no_reread(tmp_path, monkeypatch):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    store.create_attempt(identity)
    fingerprint = artifacts._fingerprint

    def unexpected_reread(*args):
        raise AssertionError("payload staging must hash during writing")

    monkeypatch.setattr(artifacts, "_fingerprint", unexpected_reread)
    reference = store.stage_result(identity, Shareable({"large": b"x" * (2 * 1024 * 1024)}))
    encoded = (tmp_path / identity.attempt_id / "result.fobs").read_bytes()
    assert reference.size == len(encoded)
    assert reference.sha256 == hashlib.sha256(encoded).hexdigest()
    monkeypatch.setattr(artifacts, "_fingerprint", fingerprint)
    store.commit_staged_result(identity, reference)
    assert store.read_result(identity)[0]["large"] == b"x" * (2 * 1024 * 1024)


def test_artifact_store_traverses_execute_only_parent(tmp_path):
    parent = tmp_path / "search-only"
    parent.mkdir()
    parent.chmod(0o311)
    try:
        store = FileTaskArtifactStore(str(parent / "artifacts"))
        identity = _identity()
        store.create_attempt(identity)
        store.write_input(identity, Shareable({"input": 1}))
        store.commit_result(identity, Shareable({"result": 2}))
        assert store.read_input(identity)["input"] == 1
        assert store.read_result(identity)[0]["result"] == 2
        store.release_payloads(identity)
    finally:
        parent.chmod(0o700)


def test_release_removes_interrupted_payload_temporaries_and_retains_diagnostics(tmp_path):
    store = FileTaskArtifactStore(str(tmp_path))
    identity = _identity()
    directory = Path(store.create_attempt(identity))
    store.commit_result(identity, Shareable({"result": 1}))
    temporary_names = [f".{name}.{'a' * 32}" for name in ("input.fobs", "result.fobs", "analytics.fobs")]
    for name in temporary_names:
        (directory / name).write_bytes(b"interrupted payload")
    retained_names = ["identity.json", "completion.json", "failure.json", f".completion.json.{'b' * 32}"]
    (directory / "failure.json").write_text("diagnostics")
    (directory / retained_names[-1]).write_text("commit diagnostics")
    store.release_payloads(identity)
    assert all(not (directory / name).exists() for name in temporary_names)
    assert all((directory / name).exists() for name in retained_names)
    assert store.read_completion(identity).identity == identity


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_release_refuses_non_regular_payload_temporary(tmp_path, kind):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    directory = Path(store.create_attempt(identity))
    temporary = directory / f".result.fobs.{'a' * 32}"
    protected = tmp_path / "protected"
    protected.write_text("retain")
    if kind == "directory":
        temporary.mkdir()
    else:
        temporary.symlink_to(protected)
    with pytest.raises(ValueError, match="invalid attempt artifact"):
        store.release_payloads(identity)
    assert protected.read_text() == "retain"
    assert temporary.exists()


@pytest.mark.parametrize("failure_point", ["before_identity", "during_identity", "after_identity"])
def test_remove_can_clean_attempt_after_identity_publication_failure(tmp_path, monkeypatch, failure_point):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    publish = artifacts._write_json_exclusive

    def interrupted(directory, name, value, **kwargs):
        assert name == "identity.json"
        if failure_point == "during_identity":
            Path(directory, f".identity.json.{'a' * 32}").write_text("partial")
        elif failure_point == "after_identity":
            publish(directory, name, value, **kwargs)
        raise OSError("interrupted identity publication")

    monkeypatch.setattr(artifacts, "_write_json_exclusive", interrupted)
    with pytest.raises(OSError, match="interrupted identity"):
        store.create_attempt(identity)
    directory = Path(store.attempt_dir(identity))
    assert directory.is_dir()
    protected = tmp_path / "protected"
    protected.write_text("retain")
    store.remove_attempt(identity)
    assert not directory.exists()
    assert protected.read_text() == "retain"


@pytest.mark.parametrize("fault", ["malformed", "symlink"])
def test_remove_still_refuses_invalid_existing_identity(tmp_path, fault):
    store = FileTaskArtifactStore(str(tmp_path / "artifacts"))
    identity = _identity()
    directory = Path(store.create_attempt(identity))
    identity_path = directory / "identity.json"
    if fault == "malformed":
        identity_path.write_text("{")
    else:
        original = tmp_path / "saved-identity.json"
        identity_path.rename(original)
        identity_path.symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        store.remove_attempt(identity)
    assert directory.is_dir()
    assert identity_path.exists()
