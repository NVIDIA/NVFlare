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

import pytest

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.fed.task_worker import FileTaskArtifactStore, IncompleteTaskArtifactError, TaskAttemptIdentity
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


@pytest.fixture(autouse=True)
def _initialize_fobs():
    nvflare_fobs_initialize()


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
