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

import json
import os
import stat
from dataclasses import replace

import pytest

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.task_state import MAX_TASK_STATE_BYTES, TaskState
from nvflare.private.fed.task_worker import state as state_module
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.private.fed.task_worker.state import FileTaskStateStore, StaleTaskStateError
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


@pytest.fixture(autouse=True)
def _initialize_fobs():
    nvflare_fobs_initialize()


def _identity(attempt="attempt-1", **changes):
    return TaskAttemptIdentity(
        **(
            {"job_id": "job-1", "site_name": "site-1", "task_id": "task-1", "task_name": "train", "attempt_id": attempt}
            | changes
        )
    )


def _stores(tmp_path, names=("optimizer",)):
    return FileTaskStateStore(str(tmp_path / "state"), names), FileTaskArtifactStore(str(tmp_path / "artifacts"))


def _completion(artifacts, identity, revision=0, value=None, names=("optimizer",)):
    artifacts.create_attempt(identity)
    records = TaskState(names, {} if value is None else {names[0]: value}).to_wire()
    candidate = artifacts.stage_state(identity, records)
    return artifacts.commit_result(
        identity, Shareable({"result": identity.attempt_id}), state=candidate, state_revision=revision
    )


def _checkpoint(store):
    with open(store.path, encoding="utf-8") as stream:
        return json.load(stream)


def _rewrite_checkpoint(store, checkpoint):
    with open(store.path, "w", encoding="utf-8") as stream:
        json.dump(checkpoint, stream)


@pytest.mark.parametrize("root", [None, "relative/path", 123])
def test_state_root_must_be_an_absolute_string(root):
    with pytest.raises(ValueError, match="absolute path"):
        FileTaskStateStore(root, [])


def test_state_revision_promotes_result_and_candidate_then_survives_payload_cleanup(tmp_path):
    state, artifacts = _stores(tmp_path)
    assert state.snapshot() == (0, {})
    identity = _identity()
    completion = _completion(artifacts, identity, value={"step": 1})
    assert state.snapshot() == (0, {})
    assert state.commit(identity, completion, artifacts) == 1
    revision, records = state.snapshot()
    assert revision == 1
    assert TaskState.from_wire(state.names, records)["optimizer"] == {"step": 1}
    records["optimizer"]["value"]["step"] = 999
    assert TaskState.from_wire(state.names, state.snapshot()[1])["optimizer"] == {"step": 1}
    assert stat.S_IMODE(os.stat(state.path).st_mode) == 0o600
    artifacts.release_payloads(identity)
    assert not os.path.exists(os.path.join(artifacts.attempt_dir(identity), "state.fobs"))
    assert not os.path.exists(os.path.join(artifacts.attempt_dir(identity), "result.fobs"))
    assert state.commit(identity, completion, artifacts) == 1
    assert FileTaskStateStore(state.root_dir, state.names).snapshot() == state.snapshot()


def test_stale_revision_cannot_overwrite_new_state_and_next_revision_can_promote(tmp_path):
    state, artifacts = _stores(tmp_path)
    first, stale, next_attempt = _identity("first"), _identity("stale"), _identity("next", task_id="task-2")
    first_completion = _completion(artifacts, first, value=1)
    stale_completion = _completion(artifacts, stale, value=2)
    assert state.commit(first, first_completion, artifacts) == 1
    with pytest.raises(StaleTaskStateError, match="no longer current"):
        state.commit(stale, stale_completion, artifacts)
    assert TaskState.from_wire(state.names, state.snapshot()[1])["optimizer"] == 1
    next_completion = _completion(artifacts, next_attempt, revision=1, value=3)
    assert state.commit(next_attempt, next_completion, artifacts) == 2
    assert TaskState.from_wire(state.names, state.snapshot()[1])["optimizer"] == 3
    with pytest.raises(StaleTaskStateError):
        state.commit(first, first_completion, artifacts)


@pytest.mark.parametrize("identity_changes", [{"job_id": "job-2"}, {"site_name": "site-2"}])
def test_committed_namespace_cannot_be_reused_by_another_job_or_site(tmp_path, identity_changes):
    state, artifacts = _stores(tmp_path)
    first = _identity("first")
    state.commit(first, _completion(artifacts, first, value=1), artifacts)
    other = _identity("other", **identity_changes)
    with pytest.raises(StaleTaskStateError, match="different job/site"):
        state.commit(other, _completion(artifacts, other, revision=1, value=2), artifacts)
    assert state.snapshot()[0] == 1


def test_promotion_requires_matching_identity_state_and_exact_durable_completion(tmp_path):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    completion = _completion(artifacts, identity, value=1)
    with pytest.raises(ValueError, match="this attempt"):
        state.commit(_identity("other"), completion, artifacts)
    with pytest.raises(ValueError, match="this attempt"):
        state.commit(identity, replace(completion, state=None), artifacts)
    with pytest.raises(ValueError, match="durable record"):
        state.commit(identity, replace(completion, diagnostics={"forged": True}), artifacts)
    assert state.snapshot() == (0, {})


@pytest.mark.parametrize("payload", ["result.fobs", "state.fobs"])
def test_both_result_and_state_payload_digests_are_verified_before_promotion(tmp_path, payload):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    completion = _completion(artifacts, identity, value=1)
    path = tmp_path / "artifacts" / identity.attempt_id / payload
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        state.commit(identity, completion, artifacts)
    assert state.snapshot() == (0, {})


@pytest.mark.parametrize("reference", ["result", "state"])
def test_same_attempt_cannot_replace_an_already_acknowledged_digest(tmp_path, reference):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    completion = _completion(artifacts, identity, value=1)
    state.commit(identity, completion, artifacts)
    changed = replace(completion, **{reference: replace(getattr(completion, reference), sha256="f" * 64)})
    path = tmp_path / "artifacts" / identity.attempt_id / "completion.json"
    path.write_text(json.dumps(changed.to_dict()), encoding="utf-8")
    with pytest.raises(StaleTaskStateError, match="cannot replace"):
        state.commit(identity, changed, artifacts)
    assert state.snapshot()[0] == 1


def test_candidate_records_must_match_the_persisted_declarations(tmp_path):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    completion = _completion(artifacts, identity, value=1, names=("undeclared",))
    with pytest.raises(KeyError, match="not declared"):
        state.commit(identity, completion, artifacts)
    assert state.snapshot() == (0, {})


def test_reopening_with_different_declarations_is_rejected(tmp_path):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    state.commit(identity, _completion(artifacts, identity, value=1), artifacts)
    with pytest.raises(ValueError, match="declarations changed"):
        FileTaskStateStore(state.root_dir, ["metrics"]).snapshot()


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": True},
        {"revision": 0},
        {"revision": "1"},
        {"extra": True},
        {"identity": {}},
        {"records": []},
        {"result_sha256": "not-a-digest"},
        {"state_sha256": "g" * 64},
        {"state_sha256": None},
    ],
)
def test_malformed_persisted_checkpoints_are_rejected(tmp_path, changes):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    state.commit(identity, _completion(artifacts, identity, value=1), artifacts)
    _rewrite_checkpoint(state, _checkpoint(state) | changes)
    with pytest.raises((ValueError, TypeError)):
        state.snapshot()


def test_oversized_checkpoint_is_rejected_before_json_decode(tmp_path):
    state, _artifacts = _stores(tmp_path)
    os.makedirs(state.root_dir)
    with open(state.path, "wb") as stream:
        stream.write(b" " * (MAX_TASK_STATE_BYTES + 65537))
    with pytest.raises(ValueError, match="too large"):
        state.snapshot()


@pytest.mark.parametrize("file_name", ["current.json", "promotion.lock"])
@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_checkpoint_and_lock_reject_symlinks_and_nonregular_files(tmp_path, file_name, kind):
    state, _artifacts = _stores(tmp_path)
    os.makedirs(state.root_dir)
    path = tmp_path / "state" / file_name
    if kind == "symlink":
        target = tmp_path / "outside"
        target.write_text("must remain unchanged", encoding="utf-8")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises((ValueError, OSError)):
        state.snapshot()
    if kind == "symlink":
        assert target.read_text(encoding="utf-8") == "must remain unchanged"


@pytest.mark.parametrize("payload", ["result.fobs", "state.fobs"])
def test_promotion_rejects_symlinked_payloads(tmp_path, payload):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    completion = _completion(artifacts, identity, value=1)
    path = tmp_path / "artifacts" / identity.attempt_id / payload
    target = tmp_path / ("outside-" + payload)
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        state.commit(identity, completion, artifacts)
    assert state.snapshot() == (0, {})


def test_failed_result_cannot_promote_successfully_finalized_candidate_state(tmp_path):
    state, artifacts = _stores(tmp_path)
    identity = _identity()
    artifacts.create_attempt(identity)
    candidate = artifacts.stage_state(identity, TaskState(state.names, {"optimizer": 1}).to_wire())
    completion = artifacts.commit_result(identity, make_reply(ReturnCode.TASK_ABORTED), state=candidate)
    with pytest.raises(ValueError, match="successful task result"):
        state.commit(identity, completion, artifacts)
    assert state.snapshot() == (0, {})


def test_interrupted_checkpoint_replace_preserves_previous_state_and_cleans_temporary(tmp_path, monkeypatch):
    state, artifacts = _stores(tmp_path)
    first = _identity("first")
    state.commit(first, _completion(artifacts, first, value=1), artifacts)
    next_attempt = _identity("next")
    completion = _completion(artifacts, next_attempt, revision=1, value=2)
    original_replace = state_module.os.replace

    def fail_replace(source, destination):
        raise OSError("injected checkpoint replacement failure")

    monkeypatch.setattr(state_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        state.commit(next_attempt, completion, artifacts)
    assert state.snapshot()[0] == 1
    assert not list((tmp_path / "state").glob(".state-*"))
    monkeypatch.setattr(state_module.os, "replace", original_replace)
    assert state.commit(next_attempt, completion, artifacts) == 2
