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
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_execution import TaskArtifactCleanup
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLauncherSpec, TaskLaunchRequest
from nvflare.apis.task_state import TaskState
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap, read_bootstrap
from nvflare.private.fed.task_worker.state import FileTaskStateStore
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions


class RecordingLauncher(TaskLauncherSpec):
    launch_mode = "test"

    def __init__(self, on_launch):
        super().__init__()
        self.launch_task = Mock(side_effect=on_launch)

    def launch_task(self, request):
        raise NotImplementedError()


@pytest.fixture(autouse=True)
def _initialize_fobs():
    fobs.register(DictDecomposer(Shareable))


def _attempt(tmp_path, *, identity=None, directory="runtime"):
    identity = identity or TaskAttemptIdentity("job", "server-like-owner", "task", "compute", "attempt-1")
    runtime_root = tmp_path / directory
    store = FileTaskArtifactStore(str(runtime_root / "attempts"))
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(tmp_path),
        executor={"path": "unused.Compute", "args": {}},
    )
    status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
    handle = SimpleNamespace(
        execution_id=f"test:{identity.attempt_id}",
        poll=Mock(return_value=status),
        cancel=Mock(return_value=status),
        wait_for_settlement=Mock(return_value=status),
    )

    def on_launch(request):
        handle.request = request
        store.commit_result(identity, Shareable({"value": 2}))
        return handle

    launcher = RecordingLauncher(on_launch)

    def request_factory(actual_identity, bootstrap_path):
        assert actual_identity == identity
        assert read_bootstrap(bootstrap_path).identity == identity
        return TaskLaunchRequest(
            job_id=identity.job_id,
            site_name=identity.site_name,
            task_id=identity.task_id,
            attempt_id=identity.attempt_id,
            argv=("opaque-test-program", bootstrap_path),
            environment={},
        )

    return {
        "bootstrap": bootstrap,
        "store": store,
        "input_payload": Shareable({"value": 1}),
        "runtime_root": str(runtime_root),
        "launcher": launcher,
        "abort_signal": Signal(),
        "request_factory": request_factory,
    }, handle


def _records(attempt):
    with open(os.path.join(attempt["runtime_root"], "diagnostics.jsonl")) as stream:
        return [json.loads(line) for line in stream]


def test_plain_supervisor_accepts_role_neutral_identity_and_returns_uninterpreted_result(tmp_path):
    assert not issubclass(TaskSupervisor, Executor)
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)

    outcome = supervisor.run(**attempt)

    assert outcome.identity == attempt["bootstrap"].identity
    assert outcome.result["value"] == 2
    assert outcome.completion.identity == outcome.identity
    assert outcome.analytics == ()
    assert not outcome.aborted
    handle.wait_for_settlement.assert_called_once()
    assert supervisor._active_handle is None
    assert [record["event"] for record in _records(attempt)] == ["launched", "settled"]


def test_acknowledgement_is_fenced_by_full_physical_identity_not_logical_task_id(tmp_path):
    supervisor = TaskSupervisor(TaskArtifactCleanup.ACCEPTED)
    first, _ = _attempt(tmp_path)
    first_identity = first["bootstrap"].identity
    second_identity = replace(first_identity, attempt_id="attempt-2")
    second, _ = _attempt(tmp_path, identity=second_identity)
    supervisor.run(**first)
    supervisor.run(**second)
    assert set(supervisor._pending_publication) == {first_identity, second_identity}

    stale = replace(first_identity, job_id="another-job")
    supervisor.acknowledge(stale, sent=True, admitted=True)
    assert set(supervisor._pending_publication) == {first_identity, second_identity}
    supervisor.acknowledge(first_identity, sent=True, admitted=True)

    first_path = first["store"].attempt_dir(first_identity)
    second_path = second["store"].attempt_dir(second_identity)
    assert not os.path.exists(os.path.join(first_path, "input.fobs"))
    assert os.path.exists(os.path.join(second_path, "input.fobs"))
    assert set(supervisor._pending_publication) == {second_identity}
    assert set(supervisor._retained_attempts) == {second_identity}


@pytest.mark.parametrize("change", [{"job_id": "job-2"}, {"site_name": "other-owner"}, {"task_name": "other-compute"}])
def test_attempt_ownership_uses_every_identity_field(tmp_path, change):
    supervisor = TaskSupervisor(TaskArtifactCleanup.RETAIN)
    first, _ = _attempt(tmp_path)
    identity = first["bootstrap"].identity
    second, _ = _attempt(tmp_path, identity=replace(identity, **change), directory="other-runtime")
    supervisor.run(**first)
    supervisor.run(**second)
    assert len(supervisor._pending_publication) == 2
    assert len(supervisor._retained_attempts) == 2


def test_launch_invocation_identity_must_match_the_staged_attempt(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    request_factory = attempt["request_factory"]
    attempt["request_factory"] = lambda identity, path: replace(request_factory(identity, path), attempt_id="stale")
    with pytest.raises(ValueError, match="does not match staged"):
        supervisor.run(**attempt)
    attempt["launcher"].launch_task.assert_not_called()
    assert supervisor._retained_attempts
    supervisor.end_run()
    assert not supervisor._retained_attempts


def test_pretriggered_abort_does_not_stage_an_attempt(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt["abort_signal"].trigger(True)
    assert supervisor.run(**attempt).aborted
    assert not os.path.exists(attempt["runtime_root"])
    attempt["launcher"].launch_task.assert_not_called()


def test_cancel_returning_unsettled_marks_fatal_state_without_client_panic(tmp_path):
    supervisor = TaskSupervisor()
    _, handle = _attempt(tmp_path)
    handle.cancel.return_value = TaskExecutionStatus(TaskExecutionPhase.RUNNING, cancel_requested=True)
    supervisor._active_handle = handle
    supervisor.cancel_active()
    assert supervisor.settlement_unconfirmed
    assert supervisor._stopping
    assert supervisor._active_handle is handle


def test_retention_cleanup_reports_retryable_issues_without_client_logging(tmp_path, monkeypatch):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    supervisor.run(**attempt)
    release = Mock(side_effect=OSError("disk unavailable"))
    monkeypatch.setattr(attempt["store"], "release_payloads", release)

    supervisor.end_run()

    assert supervisor._retained_attempts
    assert supervisor.take_issues() == [
        ("warning", "failed to remove job-ended task worker artifacts: disk unavailable")
    ]
    assert supervisor.take_issues() == []
    release.side_effect = None
    supervisor.end_run()
    assert not supervisor._retained_attempts


@pytest.mark.parametrize("value", [True, False, "1", -1, float("inf"), float("nan")])
def test_options_reject_invalid_result_wait_timeout(value):
    with pytest.raises(ValueError, match="result_wait_timeout"):
        TaskSupervisorOptions(result_wait_timeout=value)


def _declared_state_attempt(tmp_path, *, identity=None, include_state=True, revision_offset=0, result_rc=None):
    attempt, handle = _attempt(tmp_path, identity=identity)
    attempt["bootstrap"] = replace(attempt["bootstrap"], state_names=("counter",))
    state_store = FileTaskStateStore(os.path.join(attempt["runtime_root"], "state"), ("counter",))
    attempt["state_store"] = state_store

    def on_launch(request):
        bootstrap = read_bootstrap(request.argv[1])
        identity = bootstrap.identity
        state = TaskState.from_wire(bootstrap.state_names, dict(bootstrap.state_records))
        state["counter"] = state.get("counter", 0) + 1
        reference = attempt["store"].stage_state(identity, state.to_wire()) if include_state else None
        result = Shareable({"value": 2})
        if result_rc is not None:
            result.set_return_code(result_rc)
        attempt["store"].commit_result(
            identity,
            result,
            state=reference,
            state_revision=bootstrap.state_revision + revision_offset,
        )
        return handle

    attempt["launcher"].launch_task.side_effect = on_launch
    return attempt, handle


def test_declared_state_is_staged_and_promoted_only_with_admitted_result(tmp_path):
    supervisor = TaskSupervisor(TaskArtifactCleanup.ACCEPTED)
    first, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**first)
    assert first["state_store"].snapshot() == (0, {})
    assert outcome.completion.state is not None
    assert outcome.completion.state_revision == 0

    supervisor.acknowledge(outcome.identity, sent=True, admitted=True)

    revision, records = first["state_store"].snapshot()
    assert revision == 1
    assert TaskState.from_wire(("counter",), records)["counter"] == 1
    candidate_path = os.path.join(first["store"].attempt_dir(outcome.identity), "state.fobs")
    assert not os.path.exists(candidate_path)
    assert os.path.exists(first["state_store"].path)

    second, _ = _declared_state_attempt(tmp_path, identity=replace(outcome.identity, attempt_id="attempt-2"))
    next_outcome = supervisor.run(**second)
    staged = read_bootstrap(second["store"].bootstrap_path(next_outcome.identity))
    assert staged.state_revision == 1
    assert TaskState.from_wire(staged.state_names, dict(staged.state_records))["counter"] == 1
    supervisor.acknowledge(next_outcome.identity, sent=True, admitted=True)
    revision, records = second["state_store"].snapshot()
    assert revision == 2
    assert TaskState.from_wire(("counter",), records)["counter"] == 2


def test_definitively_rejected_result_never_promotes_candidate_state(tmp_path):
    supervisor = TaskSupervisor(TaskArtifactCleanup.ACCEPTED)
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    supervisor.acknowledge(outcome.identity, sent=True, admitted=False)
    assert attempt["state_store"].snapshot() == (0, {})
    assert os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))


@pytest.mark.parametrize("admitted", [None, False, True])
def test_never_submitted_state_is_rejected_without_stopping_or_advancing_state(tmp_path, admitted):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)

    supervisor.acknowledge(outcome.identity, sent=False, admitted=admitted, submission_attempted=False)

    assert not supervisor._stopping
    assert not supervisor._pending_publication
    assert attempt["state_store"].snapshot() == (0, {})
    assert _records(attempt)[-1]["publication_outcome"] == "not_submitted"
    assert _records(attempt)[-1]["submission_attempted"] is False
    next_attempt, _ = _declared_state_attempt(tmp_path, identity=replace(outcome.identity, attempt_id="attempt-2"))
    next_outcome = supervisor.run(**next_attempt)
    supervisor.acknowledge(next_outcome.identity, sent=True, admitted=True, submission_attempted=True)
    assert next_attempt["state_store"].snapshot()[0] == 1
    supervisor.end_run()
    assert not os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))
    assert _records(attempt)[-1]["event"] == "publication"


@pytest.mark.parametrize("admitted", [None, False])
def test_failed_send_after_submission_still_retains_state_and_stops(tmp_path, admitted):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    with pytest.raises(RuntimeError, match="state admission is unconfirmed"):
        supervisor.acknowledge(outcome.identity, sent=False, admitted=admitted, submission_attempted=True)
    assert supervisor._stopping
    assert outcome.identity in supervisor._pending_publication
    supervisor.end_run()
    assert os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))


def test_contradictory_no_submission_and_send_success_preserves_candidate_and_stops(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    with pytest.raises(RuntimeError, match="state admission is unconfirmed"):
        supervisor.acknowledge(outcome.identity, sent=True, admitted=True, submission_attempted=False)
    assert supervisor._stopping
    assert outcome.identity in supervisor._pending_publication
    assert attempt["state_store"].snapshot() == (0, {})


@pytest.mark.parametrize("sent,admitted", [(False, False), (False, True), (True, None), (True, 1), (True, "true")])
def test_unconfirmed_state_admission_preserves_candidate_and_halts_instead_of_using_stale_state(
    tmp_path, sent, admitted
):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    with pytest.raises(RuntimeError, match="state admission is unconfirmed"):
        supervisor.acknowledge(outcome.identity, sent=sent, admitted=admitted)
    assert supervisor._stopping
    assert outcome.identity in supervisor._pending_publication
    supervisor.end_run()
    assert attempt["state_store"].snapshot() == (0, {})
    assert os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))


@pytest.mark.parametrize(
    "rc", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_ABORTED, ReturnCode.TASK_RESULT_FILTER_ERROR]
)
def test_admitted_failure_reply_never_promotes_candidate_or_panics(tmp_path, monkeypatch, rc):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path, result_rc=rc)
    outcome = supervisor.run(**attempt)
    commit = Mock()
    monkeypatch.setattr(attempt["state_store"], "commit", commit)
    supervisor.acknowledge(outcome.identity, sent=True, admitted=True)
    commit.assert_not_called()
    assert not supervisor._stopping
    assert attempt["state_store"].snapshot() == (0, {})
    assert not supervisor._pending_publication
    supervisor.end_run()
    assert not os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))


def test_caller_failure_result_fact_cannot_promote_successful_worker_candidate(tmp_path, monkeypatch):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    commit = Mock()
    monkeypatch.setattr(attempt["state_store"], "commit", commit)
    supervisor.acknowledge(outcome.identity, sent=True, admitted=True, result_succeeded=False)
    commit.assert_not_called()
    assert attempt["state_store"].snapshot() == (0, {})
    assert not supervisor._stopping


def test_job_end_preserves_pending_state_candidate_until_admission(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)

    supervisor.end_run()

    candidate_path = os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs")
    assert os.path.exists(candidate_path)
    assert outcome.identity in supervisor._retained_attempts
    supervisor.acknowledge(outcome.identity, sent=True, admitted=True)
    assert not os.path.exists(candidate_path)
    assert attempt["state_store"].snapshot()[0] == 1
    assert not supervisor._retained_attempts


def test_job_end_cannot_race_state_promotion_reads_and_retries_cleanup(tmp_path, monkeypatch):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    reading = threading.Event()
    finish = threading.Event()
    original = attempt["state_store"].commit
    errors = []

    def blocked_commit(*args):
        reading.set()
        assert finish.wait(5)
        return original(*args)

    def acknowledge():
        try:
            supervisor.acknowledge(outcome.identity, sent=True, admitted=True)
        except BaseException as e:
            errors.append(e)

    monkeypatch.setattr(attempt["state_store"], "commit", blocked_commit)
    thread = threading.Thread(target=acknowledge)
    thread.start()
    try:
        assert reading.wait(5)
        supervisor.end_run()
        assert os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert attempt["state_store"].snapshot()[0] == 1
    assert not supervisor._retained_attempts


@pytest.mark.parametrize("failure", ["missing", "revision"])
def test_declared_state_completion_must_match_staged_contract(tmp_path, failure):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(
        tmp_path, include_state=failure != "missing", revision_offset=1 if failure == "revision" else 0
    )
    with pytest.raises(ValueError, match="state.*missing|state revision"):
        supervisor.run(**attempt)
    assert not supervisor._pending_publication
    assert attempt["state_store"].snapshot() == (0, {})


def test_failed_promotion_retains_exact_candidate_and_stops_new_attempts(tmp_path, monkeypatch):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    monkeypatch.setattr(attempt["state_store"], "commit", Mock(side_effect=OSError("promotion disk failure")))
    with pytest.raises(OSError, match="promotion disk failure"):
        supervisor.acknowledge(outcome.identity, sent=True, admitted=True)
    assert supervisor._stopping
    assert outcome.identity in supervisor._pending_publication
    supervisor.end_run()
    assert os.path.exists(os.path.join(attempt["store"].attempt_dir(outcome.identity), "state.fobs"))


def test_declared_state_requires_matching_supervisor_store_before_staging(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _declared_state_attempt(tmp_path)
    state_store = attempt.pop("state_store")
    with pytest.raises(ValueError, match="requires a supervising state store"):
        supervisor.run(**attempt)
    attempt["state_store"] = FileTaskStateStore(state_store.root_dir, ("other",))
    with pytest.raises(ValueError, match="declarations do not match"):
        supervisor.run(**attempt)
    assert not os.path.exists(attempt["store"].root_dir)


@pytest.mark.parametrize("failure", ["names", "tamper"])
def test_declared_state_candidate_names_and_integrity_are_verified_before_publication(tmp_path, failure):
    supervisor = TaskSupervisor()
    attempt, handle = _declared_state_attempt(tmp_path)

    def on_launch(request):
        bootstrap = read_bootstrap(request.argv[1])
        records = {"counter" if failure == "tamper" else "undeclared": {"encoding": "json", "value": 1}}
        reference = attempt["store"].stage_state(bootstrap.identity, records)
        attempt["store"].commit_result(
            bootstrap.identity, Shareable(), state=reference, state_revision=bootstrap.state_revision
        )
        if failure == "tamper":
            path = os.path.join(attempt["store"].attempt_dir(bootstrap.identity), "state.fobs")
            with open(path, "ab") as stream:
                stream.write(b"tampered")
        return handle

    attempt["launcher"].launch_task.side_effect = on_launch
    with pytest.raises((ValueError, KeyError), match="not declared|checksum mismatch"):
        supervisor.run(**attempt)
    assert not supervisor._pending_publication
    assert attempt["state_store"].snapshot() == (0, {})


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("input_payload", {}, "Shareable"),
        ("bootstrap", None, "WorkerBootstrap"),
        ("options", {}, "Options"),
        ("launcher", object(), "TaskLauncherSpec"),
    ],
)
def test_supervisor_rejects_invalid_service_inputs_without_staging(tmp_path, name, value, expected):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt[name] = value
    with pytest.raises(TypeError, match=expected):
        supervisor.run(**attempt)
    assert not os.path.exists(attempt["runtime_root"])


def test_acknowledgement_requires_a_full_attempt_identity():
    with pytest.raises(TypeError, match="TaskAttemptIdentity"):
        TaskSupervisor().acknowledge("logical-task-id", sent=True, admitted=True)
