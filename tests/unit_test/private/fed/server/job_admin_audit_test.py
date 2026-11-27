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

"""Administrative job facts are independent of replies and later bookkeeping."""

import json
from contextlib import contextmanager, nullcontext
from threading import Lock
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_def_manager_spec import RunStatus
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context, snapshot
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import MsgHeader, error_reply, ok_reply
from nvflare.private.fed.server import job_cmds
from nvflare.private.fed.server.job_cmds import JobCommandModule
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.message_send import ClientReply
from nvflare.private.fed.server.server_engine import ServerEngine

JOB_ID = "123e4567-e89b-42d3-a456-426614174000"
ALLOWED_FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


class Connection:
    def __init__(self, engine, **props):
        self.app_ctx = engine
        self.props = props
        self.errors = []
        self.strings = []
        self.successes = []
        self.dicts = []

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def set_prop(self, key, value):
        self.props[key] = value

    def append_error(self, message, meta=None):
        self.errors.append((message, meta))

    def append_string(self, message, meta=None):
        self.strings.append((message, meta))

    def append_success(self, message, meta=None):
        self.successes.append((message, meta))

    def append_dict(self, value, meta=None):
        self.dicts.append((value, meta))


@pytest.fixture
def audit_records(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)

    def read_records():
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            assert set(record) <= ALLOWED_FIELDS
            assert all(isinstance(value, str) for value in record.values())
            assert record["timestamp"].endswith("+00:00")
            assert record["user"] == "verified-admin@example.org"
            assert record["reference"] == "original-request"
            assert "session-secret" not in str(record)
            assert "exception-private" not in str(record)
            assert "/private/" not in str(record)
        return records

    with audit_context(AdminContext(user="verified-admin@example.org", request_id="original-request")):
        yield read_records, writer
    writer.close()


def make_engine(manager):
    return SimpleNamespace(job_def_manager=manager, new_context=lambda: nullcontext(object()))


def make_job(status=RunStatus.FINISHED_COMPLETED.value):
    return SimpleNamespace(meta={JobMetaKey.STATUS: status, JobMetaKey.SCOPE: "job-scope"})


def job_connection(engine, status=RunStatus.FINISHED_COMPLETED.value):
    return Connection(engine, job_id=JOB_ID, job=make_job(status))


@pytest.mark.parametrize("failure", [None, "delete", "mark_deleted", "mark_records", "remove_logs", "context_exit"])
def test_deletion_records_manager_effect_outside_lock_even_when_cleanup_fails(audit_records, monkeypatch, failure):
    records, writer = audit_records
    locked = False
    deleted = []

    def fail_at(stage):
        if failure == stage:
            raise RuntimeError("exception-private /private/path session-secret")

    @contextmanager
    def staging_lock():
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def delete(*args):
        fail_at("delete")
        deleted.append(JOB_ID)

    def mark_records(*args):
        fail_at("mark_records")
        return []

    staging = SimpleNamespace(
        locked=staging_lock,
        prepare_delete=lambda *args: None,
        cancel_delete=lambda *args: None,
        mark_deleted=lambda *args: fail_at("mark_deleted"),
        remove_closed_job_logs=lambda *args: fail_at("remove_logs"),
    )
    manager = SimpleNamespace(delete=delete, mark_submit_records_job_deleted=mark_records)
    engine = make_engine(manager)
    engine.get_workspace = lambda: object()

    @contextmanager
    def context():
        yield object()
        fail_at("context_exit")

    engine.new_context = context
    monkeypatch.setattr(job_cmds, "ClientLogStaging", lambda workspace: staging)
    real_add_event = writer.add_event

    def checked_add_event(*args, **kwargs):
        assert not locked, "audit I/O must happen after staging lock release"
        return real_add_event(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", checked_add_event)
    conn = job_connection(engine)
    JobCommandModule().delete_job(conn, ["delete_job", JOB_ID])
    events = records()
    if failure == "delete":
        assert not deleted
        assert [event["action"] for event in events] == ["job.delete.error"]
    else:
        assert deleted == [JOB_ID]
        expected = ["job.delete.applied"]
        if failure:
            expected.append("job.delete.cleanup_error")
        assert [event["action"] for event in events] == expected
    assert all(event["job"] == JOB_ID for event in events)
    assert bool(conn.errors) == bool(failure)
    assert bool(conn.successes) == (failure is None)


@pytest.mark.parametrize("status", [RunStatus.RUNNING.value, RunStatus.DISPATCHED.value])
def test_running_job_delete_is_rejected_without_effect(audit_records, status):
    records, _writer = audit_records
    manager = SimpleNamespace(delete=MagicMock())
    conn = job_connection(make_engine(manager), status)
    JobCommandModule().delete_job(conn, ["delete_job", JOB_ID])
    manager.delete.assert_not_called()
    assert [event["action"] for event in records()] == ["job.delete.rejected"]
    assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.JOB_RUNNING


@pytest.mark.parametrize(
    "state, signal_result, expected_action, reason",
    [
        ("starting", None, "job.abort.retry_required", "job_starting"),
        (RunStatus.SUBMITTED.value, None, "job.abort.before_execution", "aborted_before_execution"),
        (RunStatus.DISPATCHED.value, None, "job.abort.before_execution", "aborted_before_execution"),
        (RunStatus.FINISHED_COMPLETED.value, None, "job.abort.unchanged", "already_finished"),
        (RunStatus.RUNNING.value, None, "job.abort.requested", "abort_requested"),
        (None, "exception-private", "job.abort.error", "abort_request_failed"),
        ("exception", None, "job.abort.error", "processing_error"),
    ],
)
def test_abort_reports_observed_state_without_claiming_execution_completion(
    audit_records, state, signal_result, expected_action, reason
):
    records, _writer = audit_records
    starting = object()
    runner = SimpleNamespace(
        STARTING=starting,
        abort_before_start=MagicMock(return_value=starting if state == "starting" else state),
        stop_run=MagicMock(return_value=signal_result),
    )
    if state == "exception":
        runner.abort_before_start.side_effect = RuntimeError("exception-private")
    engine = make_engine(object())
    engine.job_runner = runner
    conn = job_connection(engine)
    JobCommandModule().abort_job(conn, ["abort_job", JOB_ID])
    [event] = records()
    assert event["action"] == expected_action
    assert event["message"] == "reason=" + reason
    assert event["job"] == JOB_ID
    if expected_action in {"job.abort.retry_required", "job.abort.before_execution", "job.abort.unchanged"}:
        runner.stop_run.assert_not_called()
    elif state != "exception":
        runner.stop_run.assert_called_once()


def test_job_log_configuration_reports_each_remote_result_without_recording_contents(audit_records, monkeypatch):
    records, _writer = audit_records
    manager = SimpleNamespace(get_job=lambda *args: make_job(RunStatus.RUNNING.value))
    engine = make_engine(manager)
    engine.configure_job_log = MagicMock(return_value=None)
    monkeypatch.setattr(job_cmds, "ServerEngine", SimpleNamespace)
    conn = job_connection(engine, RunStatus.RUNNING.value)
    conn.set_prop(
        JobCommandModule.TARGET_CLIENTS,
        {"session-secret-a": "site-a", "session-secret-b": "site-b", "session-secret-c": "site-c"},
    )
    replies = [
        ClientReply(client_token="session-secret-a", client_name="site-a", req=None, reply=ok_reply()),
        ClientReply(
            client_token="session-secret-b",
            client_name="site-b",
            req=None,
            reply=error_reply("exception-private /private/config"),
        ),
    ]
    module = JobCommandModule()
    monkeypatch.setattr(module, "send_request_to_clients", lambda *args: replies)
    monkeypatch.setattr(module, "process_replies_to_table", lambda *args: None)
    config = "/private/secret-logging-configuration"
    module.configure_job_log(conn, ["configure_job_log", JOB_ID, "all", config])
    engine.configure_job_log.assert_called_once_with(JOB_ID, config)
    events = records()
    assert len(events) == 4
    assert {event["message"] for event in events} == {
        "reason=unknown_outcome site=server",
        "reason=acknowledged site=site-a",
        "reason=remote_error site=site-b",
        "reason=no_reply site=site-c",
    }
    assert all(event["action"] == "logging.job.remote_reported" for event in events)
    assert all(event["job"] == JOB_ID and event["scope"] == "job-scope" for event in events)
    assert conn.errors


def test_missing_remote_return_code_is_not_reported_as_success(audit_records, monkeypatch):
    records, _writer = audit_records
    engine = make_engine(SimpleNamespace(get_job=lambda *args: make_job(RunStatus.RUNNING.value)))
    monkeypatch.setattr(job_cmds, "ServerEngine", SimpleNamespace)
    conn = job_connection(engine)
    reply = ok_reply()
    reply.headers.pop(MsgHeader.RETURN_CODE, None)
    module = JobCommandModule()
    monkeypatch.setattr(
        module,
        "send_request_to_clients",
        lambda *args: [ClientReply(client_token="session-secret", client_name="site-a", req=None, reply=reply)],
    )
    monkeypatch.setattr(module, "process_replies_to_table", lambda *args: None)
    module.configure_job_log(conn, ["configure_job_log", JOB_ID, "client", "site-a", "DEBUG"])
    [event] = records()
    assert "reason=remote_error" in event["message"]
    assert not conn.errors  # Preserve the existing ordinary reply handling default.


def test_scope_audit_preparation_failure_does_not_replay_or_block_configuration(audit_records, monkeypatch):
    records, _writer = audit_records

    class Meta(dict):
        def get(self, key, default=None):
            if key == JobMetaKey.SCOPE:
                raise RuntimeError("exception-private")
            return super().get(key, default)

    job = SimpleNamespace(meta=Meta({JobMetaKey.STATUS: RunStatus.RUNNING.value}))
    engine = make_engine(SimpleNamespace(get_job=lambda *args: job))
    engine.configure_job_log = MagicMock(return_value=None)
    monkeypatch.setattr(job_cmds, "ServerEngine", SimpleNamespace)
    conn = job_connection(engine)
    JobCommandModule().configure_job_log(conn, ["configure_job_log", JOB_ID, "server", "DEBUG"])
    engine.configure_job_log.assert_called_once()
    assert not conn.errors
    assert [event["action"] for event in records()] == ["logging.job.remote_reported"]


@pytest.mark.parametrize(
    "state, expected_reason", [(None, "job_unavailable"), (RunStatus.RUNNING.value, "job_not_finished")]
)
def test_download_rejection_has_one_record_and_only_established_job_context(
    audit_records, monkeypatch, tmp_path, state, expected_reason
):
    records, _writer = audit_records
    manager = SimpleNamespace(get_job=lambda *args: None if state is None else make_job(state))
    engine = make_engine(manager)
    monkeypatch.setattr(job_cmds, "JobDefManagerSpec", object)
    conn = Connection(engine)
    conn.set_prop(ConnProps.DOWNLOAD_DIR, str(tmp_path))
    module = JobCommandModule()
    module.download_folder = MagicMock()
    module.download_job(conn, ["download_job", JOB_ID])
    module.download_folder.assert_not_called()
    [event] = records()
    assert event["action"] == "download.setup.rejected"
    assert event["message"] == "reason=" + expected_reason
    if state is None:
        assert "job" not in event and "scope" not in event
    else:
        assert event["job"] == JOB_ID and event["scope"] == "job-scope"


def test_component_download_snapshots_established_job_without_using_file_names(audit_records, monkeypatch, tmp_path):
    _records, _writer = audit_records
    manager = SimpleNamespace(
        list_components=lambda **kwargs: ["ERRORLOG_site-1"],
        get_storage_for_download=lambda *args: None,
    )
    monkeypatch.setattr(job_cmds, "JobDefManagerSpec", object)
    conn = Connection(make_engine(manager))
    conn.set_prop(ConnProps.DOWNLOAD_DIR, str(tmp_path))
    observed = []
    module = JobCommandModule()
    monkeypatch.setattr(module, "download_folder", lambda *args, **kwargs: observed.append(snapshot()))
    module.download_job_components(conn, ["download_job_components", JOB_ID])
    assert len(observed) == 1
    assert observed[0].job_id == JOB_ID
    assert observed[0].user == "verified-admin@example.org"
    assert observed[0].request_id == "original-request"


def test_failed_download_cleanup_does_not_duplicate_setup_rejection(audit_records, monkeypatch, tmp_path):
    records, _writer = audit_records
    manager = SimpleNamespace(get_job=lambda *args: make_job())
    monkeypatch.setattr(job_cmds, "JobDefManagerSpec", object)
    conn = Connection(make_engine(manager))
    conn.set_prop(ConnProps.DOWNLOAD_DIR, str(tmp_path))
    module = JobCommandModule()
    monkeypatch.setattr(module, "_prepare_whole_job", MagicMock(side_effect=RuntimeError("exception-private")))
    monkeypatch.setattr(module, "_clean_up_download", MagicMock(side_effect=RuntimeError("exception-private")))
    with pytest.raises(RuntimeError):
        module.download_job(conn, ["download_job", JOB_ID])
    assert [event["action"] for event in records()] == ["download.setup.rejected"]


def test_binary_transfer_setup_error_is_not_recorded_again_by_job_download(audit_records, monkeypatch, tmp_path):
    records, _writer = audit_records
    manager = SimpleNamespace(
        get_job=lambda *args: make_job(),
        get_storage_for_download=lambda *args: None,
    )
    monkeypatch.setattr(job_cmds, "JobDefManagerSpec", object)
    conn = Connection(make_engine(manager))
    conn.set_prop(ConnProps.DOWNLOAD_DIR, str(tmp_path))
    module = JobCommandModule()
    transfer_boundary = MagicMock(side_effect=OSError("exception-private /private/download"))
    # Keep both real setup wrappers: BinaryTransfer.download_folder and the job command.
    monkeypatch.setattr(module, "_download_folder", transfer_boundary)

    result = module.download_job(conn, ["download_job", JOB_ID])

    transfer_boundary.assert_called_once()
    assert result is None
    assert not conn.successes
    assert conn.errors == [("internal error", {MetaKey.STATUS: MetaStatusValue.INTERNAL_ERROR, MetaKey.INFO: ""})]
    [event] = records()
    assert event["action"] == "download.setup.rejected"
    assert event["message"] == "reason=setup_error"
    assert event["reference"] == "original-request"
    assert event["job"] == JOB_ID


def test_direct_job_log_reply_records_preparation_only(audit_records, monkeypatch):
    records, _writer = audit_records
    engine = make_engine(object())
    monkeypatch.setattr(job_cmds, "ServerEngine", SimpleNamespace)
    conn = job_connection(engine)
    module = JobCommandModule()
    monkeypatch.setattr(module, "_read_server_job_log", lambda *args, **kwargs: "private file contents")
    module.get_job_log(conn, ["get_job_log", JOB_ID])
    assert conn.dicts[0][0] == {"logs": {"server": "private file contents"}}
    [event] = records()
    assert event["action"] == "download.job_log.prepared"
    assert event["message"] == "reason=reply_prepared component=job_log"
    assert "private file contents" not in str(event)


def test_direct_job_log_invalid_input_is_rejected_once_without_raw_arguments(audit_records):
    records, _writer = audit_records
    conn = Connection(object())
    JobCommandModule().get_job_log(conn, ["get_job_log", JOB_ID, "--tail-bytes", "-1"])
    [event] = records()
    assert event["action"] == "download.setup.rejected"
    assert event["message"] == "reason=invalid_arguments component=job_log"
    assert "job" not in event


@pytest.mark.parametrize("latest_status", [RunStatus.FINISHED_COMPLETED, RunStatus.RUNNING, None])
def test_abort_when_running_job_disappears_before_stop(audit_records, tmp_path, latest_status):
    records, _writer = audit_records
    # The initial read sees RUNNING. By the time stop_run looks, the runtime entry is gone.
    latest_job = make_job(latest_status) if latest_status else None
    manager = SimpleNamespace(get_job=MagicMock(side_effect=[make_job(RunStatus.RUNNING), latest_job]))
    engine = make_engine(manager)
    engine.job_runner = JobRunner(str(tmp_path))
    engine.run_processes = {}
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    engine.new_context = lambda: nullcontext(fl_ctx)
    conn = job_connection(engine, RunStatus.RUNNING)

    JobCommandModule().abort_job(conn, ["abort_job", JOB_ID])

    [event] = records()
    if latest_status == RunStatus.FINISHED_COMPLETED:
        assert not conn.errors
        assert conn.successes[0][1][MetaKey.STATUS] == MetaStatusValue.OK
        assert event["action"] == "job.abort.unchanged"
    else:
        assert not conn.successes
        assert conn.errors[0][0] == f"Job {JOB_ID} is not running."
        assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR
        assert event["action"] == "job.abort.error"


def test_abort_request_after_server_exit_does_not_claim_dispatch(audit_records, tmp_path):
    records, _writer = audit_records
    manager = SimpleNamespace(get_job=lambda *args: make_job(RunStatus.RUNNING.value))
    engine = make_engine(manager)
    engine.run_processes = {}
    engine.abort_app_on_server = MagicMock()
    runner = JobRunner(str(tmp_path))
    runner.abort_client_run = MagicMock()
    tracked_job = SimpleNamespace(job_id=JOB_ID, run_aborted=False)
    runner.running_jobs[JOB_ID] = tracked_job
    runner._pending_client_outcomes[JOB_ID] = {"site-a"}
    engine.job_runner = runner
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    engine.new_context = lambda: nullcontext(fl_ctx)
    conn = job_connection(engine, RunStatus.RUNNING.value)

    JobCommandModule().abort_job(conn, ["abort_job", JOB_ID])

    assert tracked_job.run_aborted
    assert runner._pending_client_outcomes[JOB_ID] == {"site-a"}
    engine.abort_app_on_server.assert_not_called()
    runner.abort_client_run.assert_not_called()
    [event] = records()
    assert event["action"] == "job.abort.requested"
    assert event["message"] == "reason=abort_requested"
    assert event["job"] == JOB_ID
    assert not conn.errors
    # Preserve the existing command response; only the new audit evidence changes.
    assert conn.strings == [("Abort signal has been sent to the server app.", None)]
    assert conn.successes[0][1][MetaKey.STATUS] == MetaStatusValue.OK


@pytest.mark.parametrize(
    "transport_code, payload, expected_reason",
    [
        (CellReturnCode.TIMEOUT, None, "unknown_outcome"),
        (CellReturnCode.SERVICE_UNAVAILABLE, None, "unknown_outcome"),
        (CellReturnCode.OK, None, "unknown_outcome"),
        (CellReturnCode.OK, "", "unknown_outcome"),
        (CellReturnCode.OK, "exception-private /private/config", "remote_error"),
    ],
)
def test_job_log_audits_actual_child_transport_without_inventing_acknowledgement(
    audit_records, monkeypatch, transport_code, payload, expected_reason
):
    records, _writer = audit_records
    engine = make_engine(SimpleNamespace(get_job=lambda *args: make_job(RunStatus.RUNNING.value)))
    engine.lock = Lock()
    cell = SimpleNamespace(
        send_request=MagicMock(return_value=CellMessage({MessageHeaderKey.RETURN_CODE: transport_code}, payload))
    )
    engine.server = SimpleNamespace(cell=cell)
    engine.logger = MagicMock()
    # Exercise both real engine methods; only the network boundary is controlled.
    engine.send_command_to_child_runner_process = MethodType(ServerEngine.send_command_to_child_runner_process, engine)
    engine.configure_job_log = MethodType(ServerEngine.configure_job_log, engine)
    monkeypatch.setattr(job_cmds, "ServerEngine", SimpleNamespace)
    conn = job_connection(engine, RunStatus.RUNNING.value)
    config = "/private/secret-logging-configuration"

    JobCommandModule().configure_job_log(conn, ["configure_job_log", JOB_ID, "server", config])

    cell.send_request.assert_called_once()
    assert cell.send_request.call_args.kwargs["request"].payload == config
    [event] = records()
    assert event["action"] == "logging.job.remote_reported"
    assert event["message"] == "reason=" + expected_reason + " site=server"
    assert event["job"] == JOB_ID
    assert event["scope"] == "job-scope"
    assert bool(conn.errors) == bool(payload)
    assert bool(conn.strings) == (not bool(payload))
    if not payload:
        assert conn.strings == [(f"successfully configured server job {JOB_ID} log", None)]
