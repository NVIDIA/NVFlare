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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import audit_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, MsgHeader, error_reply, ok_reply
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.client_manager import ClientManager
from nvflare.private.fed.server.message_send import ClientReply
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_status import ServerStatus
from nvflare.private.fed.server.training_cmds import TrainingCommandModule


@pytest.fixture
def audit_records(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)

    def read():
        records = [json.loads(line) for line in path.read_text().splitlines()]
        permitted = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
        for item in records:
            assert set(item) <= permitted
            assert all(isinstance(value, str) for value in item.values())
            assert item["user"] == "admin@example.com"
            assert item["reference"] == "request-event"
        assert "secret-token" not in path.read_text()
        assert "private-path" not in path.read_text()
        return records

    yield writer, read
    writer.close()


def _context(command):
    return audit_context(user="admin@example.com", request_id="request-event", command=command)


def _manager(tmp_path):
    manager = ClientManager()
    manager.set_disabled_clients_file(str(tmp_path / "disabled.json"))
    manager.logger = MagicMock()
    return manager


def _engine(manager):
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = SimpleNamespace(client_manager=manager, remove_client_data=MagicMock(), admin_server=MagicMock())
    engine.logger = MagicMock()
    return engine


def test_policy_applied_unchanged_and_persist_errors_are_outside_lock(tmp_path, monkeypatch, audit_records):
    writer, read = audit_records
    manager = _manager(tmp_path)
    client = Client("site-1", "secret-token")
    manager.set_clients({client.token: client})
    lock_observations = []
    original_write = writer.add_event

    def checked_write(*args, **kwargs):
        acquired = manager.lock.acquire(blocking=False)
        lock_observations.append(acquired)
        if acquired:
            manager.lock.release()
        return original_write(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", checked_write)
    with _context("training.disable_client"):
        assert manager.disable_client("site-1") == ["secret-token"]
        assert manager.disable_client("site-1") == []
    with _context("enable_client"):
        assert manager.enable_client("site-1") is True
        assert manager.enable_client("site-1") is False
    monkeypatch.setattr(manager, "_save_disabled_clients", MagicMock(side_effect=OSError("private-path")))
    with _context("disable_client"), pytest.raises(OSError):
        manager.disable_client("site-1")
    assert not manager.is_client_disabled("site-1")
    assert [item["action"] for item in read()] == [
        "client.disable.applied",
        "client.disable.unchanged",
        "client.enable.applied",
        "client.enable.unchanged",
        "client.disable.error",
    ]
    assert lock_observations == [True] * 5


def test_nonadministrative_policy_changes_are_silent(tmp_path, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    manager.disable_client("site-1")
    manager.enable_client("site-1")
    assert read() == []


def test_mixed_disable_results_keep_per_target_records_and_replies(tmp_path, monkeypatch, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    engine = _engine(manager)
    original_save = manager._save_disabled_clients

    def save(disabled):
        if "site-bad" in disabled:
            raise OSError("private-path")
        original_save(disabled)

    monkeypatch.setattr(manager, "_save_disabled_clients", save)
    with _context("disable_client"):
        result = engine.disable_clients(["site-1", "site-bad", "site-2", "site-1"])
    assert [item["state"] for item in result["clients"]] == ["disabled", "error", "disabled", "disabled"]
    assert manager.disabled_clients == {"site-1", "site-2"}
    records = read()
    assert [item["action"] for item in records] == [
        "client.disable.applied",
        "client.disable.error",
        "client.disable.applied",
        "client.disable.unchanged",
    ]
    for item, site in zip(records, ["site-1", "site-bad", "site-2", "site-1"]):
        assert f"site={site}" in item["message"]


def test_disable_persistence_precedes_cleanup_failure(tmp_path, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    client = Client("site-1", "secret-token")
    manager.set_clients({client.token: client})
    engine = _engine(manager)
    engine.server.remove_client_data.side_effect = RuntimeError("private-path")
    with _context("disable_client"), pytest.raises(RuntimeError):
        engine.disable_clients(["site-1"])
    assert manager.is_client_disabled("site-1")
    assert manager.clients == {}
    assert json.loads((tmp_path / "disabled.json").read_text()) == {"disabled_clients": ["site-1"]}
    assert [item["action"] for item in read()] == ["client.disable.applied", "client.disable.cleanup_error"]


def test_remove_applied_record_survives_cleanup_failure(tmp_path, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    client = Client("site-1", "secret-token")
    manager.set_clients({client.token: client})
    engine = _engine(manager)
    engine.server.admin_server.client_dead.side_effect = RuntimeError("private-path")
    with _context("remove_client"), pytest.raises(RuntimeError):
        engine.remove_clients([client.token])
    assert manager.clients == {}
    assert [item["action"] for item in read()] == ["client.remove.applied", "client.remove.cleanup_error"]


def test_remove_unknown_entry_is_unchanged_and_never_exposes_token(tmp_path, audit_records):
    _, read = audit_records
    engine = _engine(_manager(tmp_path))
    with _context("remove_client"):
        assert engine.remove_clients(["secret-token"]) == ""
    records = read()
    assert [item["action"] for item in records] == ["client.remove.unchanged"]
    assert "site=" not in records[0]["message"]


def test_remove_command_preserves_validated_names_after_disconnect(tmp_path, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    alive = Client("site-alive", "secret-token-alive")
    gone = Client("site-gone", "secret-token-gone")
    manager.set_clients({client.token: client for client in (alive, gone)})
    engine = _engine(manager)
    engine.client_manager = manager
    module = TrainingCommandModule()
    conn = Connection(app_ctx=engine, props={})
    args = ["remove_client", alive.name, gone.name]
    assert module.authorize_client_operation(conn, args) == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert manager.remove_client(gone.token) is gone

    def cleanup(token):
        if token == alive.token:
            conn.get_prop(module.TARGET_CLIENTS).pop(gone.token)

    engine.server.remove_client_data.side_effect = cleanup
    with _context("training.remove_client"):
        module.remove_client(conn, args)
        records = read()
        assert [item["action"] for item in records] == ["client.remove.applied", "client.remove.unchanged"]
        assert "site=site-alive" in records[0]["message"]
        assert "site=site-gone" in records[1]["message"]
        assert conn.buffer.data[-1][ProtoKey.TYPE] == ProtoKey.SUCCESS
        assert manager.clients == {}
        assert [call.args[0] for call in engine.server.remove_client_data.call_args_list] == [alive.token, gone.token]

        # A later direct engine call cannot reuse the command's target snapshot.
        assert engine.remove_clients([gone.token]) == ""
    assert read()[-1]["action"] == "client.remove.unchanged"
    assert "site=" not in read()[-1]["message"]


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
@pytest.mark.parametrize("state", ["starting", "scheduled", "submit_error"])
def test_server_control_records_submission_not_process_exit(operation, state, audit_records):
    _, read = audit_records
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = SimpleNamespace(status=ServerStatus.STARTING if state == "starting" else ServerStatus.STARTED)
    engine.logger = MagicMock()
    engine.args = SimpleNamespace(workspace="/private-path")
    engine.executor = MagicMock()
    if state == "submit_error":
        engine.executor.submit.side_effect = RuntimeError("private-path")
    with _context(operation):
        if state == "submit_error":
            with pytest.raises(RuntimeError):
                getattr(engine, operation + "_server")()
        else:
            response = getattr(engine, operation + "_server")()
            assert bool(response) is (state == "starting")
    records = read()
    expected = {"starting": "rejected", "scheduled": "scheduled", "submit_error": "error"}[state]
    assert [item["action"] for item in records] == [f"server.{operation}.{expected}"]
    assert engine.executor.submit.call_count == (0 if state == "starting" else 1)


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
def test_remote_control_records_only_observed_acknowledgement(operation, audit_records):
    _, read = audit_records
    replies = [
        ClientReply("secret-token-1", "site-1", None, ok_reply(body="private-path")),
        ClientReply("secret-token-2", "site-2", None, error_reply("private-path")),
        ClientReply("secret-token-3", "site-3", None, None),
    ]
    with _context(operation):
        TrainingCommandModule._record_control_replies(
            operation, replies, {"secret-token-1": "site-1", "secret-token-2": "site-2", "secret-token-3": "site-3"}
        )
    assert [item["action"] for item in read()] == [
        f"client.{operation}.acknowledged",
        f"client.{operation}.reported_error",
        f"client.{operation}.no_reply",
    ]


def test_mixed_enable_results_preserve_failed_target_policy(tmp_path, monkeypatch, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    for site in ("site-1", "site-bad", "site-2"):
        manager.disable_client(site)
    original_save = manager._save_disabled_clients

    def save(disabled):
        if "site-bad" not in disabled:
            raise OSError("private-path")
        original_save(disabled)

    monkeypatch.setattr(manager, "_save_disabled_clients", save)
    with _context("enable_client"):
        result = _engine(manager).enable_clients(["site-1", "site-bad", "site-2", "site-1"])
    assert [item["state"] for item in result["clients"]] == ["enabled", "error", "enabled", "enabled"]
    assert manager.disabled_clients == {"site-bad"}
    assert [item["action"] for item in read()] == [
        "client.enable.applied",
        "client.enable.error",
        "client.enable.applied",
        "client.enable.unchanged",
    ]


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
def test_running_jobs_reject_control_before_scheduling(operation, audit_records):
    _, read = audit_records
    engine = ServerEngine.__new__(ServerEngine)
    engine.job_runner = SimpleNamespace(running_jobs={"job-1": SimpleNamespace(run_aborted=False, job_id="job-1")})
    engine.shutdown_server = MagicMock()
    engine.restart_server = MagicMock()
    conn = MagicMock()
    conn.app_ctx = engine
    with _context(operation):
        getattr(TrainingCommandModule(), operation)(conn, [operation, "server"])
    assert [item["action"] for item in read()] == [f"admin.{operation}.rejected"]
    engine.shutdown_server.assert_not_called()
    engine.restart_server.assert_not_called()
    conn.append_error.assert_called_once()


def test_remote_ack_audit_preparation_failure_does_not_escape(audit_records):
    _, read = audit_records
    reply = MagicMock()
    reply.get_header.side_effect = RuntimeError("private-path")
    with _context("shutdown"):
        TrainingCommandModule._record_control_replies(
            "shutdown", [ClientReply("secret-token", "site-1", None, reply)], {"secret-token": "site-1"}
        )
    assert read() == []


def test_record_write_failure_preserves_applied_policy_and_control_submission(tmp_path, monkeypatch, audit_records):
    writer, read = audit_records
    monkeypatch.setattr(writer, "add_event", MagicMock(side_effect=OSError("private-path")))
    manager = _manager(tmp_path)
    with _context("disable_client"):
        assert manager.disable_client("site-1") == []
    assert manager.is_client_disabled("site-1")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = SimpleNamespace(status=ServerStatus.STARTED)
    engine.logger = MagicMock()
    engine.args = SimpleNamespace(workspace="/private-path")
    engine.executor = MagicMock()
    with _context("shutdown"):
        assert engine.shutdown_server() == ""
    engine.executor.submit.assert_called_once()
    assert read() == []


def test_automatic_client_removal_is_silent(tmp_path, audit_records):
    _, read = audit_records
    manager = _manager(tmp_path)
    manager.set_clients({"secret-token": Client("site-1", "secret-token")})
    engine = _engine(manager)
    engine._remove_dead_client("secret-token")
    assert manager.clients == {}
    assert read() == []


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
def test_selected_disconnected_control_targets_survive_real_send(operation, audit_records):
    _, read = audit_records
    module = TrainingCommandModule()
    selected = {
        "secret-token-alive": "site-alive",
        "secret-token-gone": "site-gone",
        "secret-token-empty": "site-empty",
    }
    clients = {token: SimpleNamespace(token=token, name=name, fqcn=name) for token, name in selected.items()}
    admin = FedAdminServer.__new__(FedAdminServer)
    admin.clients = clients
    admin.timeout = 2.0
    admin.sai = SimpleNamespace(new_context=FLContext, fire_event=MagicMock())
    broadcast = MagicMock(
        return_value={
            "site-alive": CellMessage(payload=ok_reply(body="response")),
            "site-empty": CellMessage(payload=None),
        }
    )
    admin.cell = SimpleNamespace(broadcast_multi_requests=broadcast)
    original_send = admin.send_requests
    conn = Connection(
        app_ctx=MagicMock(spec=ServerEngine),
        props={
            module.TARGET_CLIENT_TOKENS: list(selected),
            module.TARGET_CLIENTS: selected,
            ConnProps.ADMIN_SERVER: admin,
        },
    )

    def disconnect_before_send(requests, fl_ctx, **kwargs):
        clients.pop("secret-token-gone")
        selected.pop("secret-token-gone")
        return original_send(requests, fl_ctx, **kwargs)

    admin.send_requests = MagicMock(side_effect=disconnect_before_send)
    with _context(operation):
        result = module._shutdown_app_on_clients(conn) if operation == "shutdown" else module._restart_clients(conn)
    admin.send_requests.assert_called_once()
    broadcast.assert_called_once()
    assert set(broadcast.call_args.args[0]) == {"site-alive", "site-empty"}
    records = read()
    assert [event["action"] for event in records] == [
        f"client.{operation}.acknowledged",
        f"client.{operation}.no_reply",
        f"client.{operation}.no_reply",
    ]
    for event, site in zip(records, ["site-alive", "site-gone", "site-empty"]):
        assert "site=" + site in event["message"]
    if operation == "shutdown":
        assert result is True
        assert conn.buffer.data == [
            {
                ProtoKey.TYPE: ProtoKey.TABLE,
                ProtoKey.ROWS: [["CLIENT", "RESPONSE"], ["site-alive", "response"], ["site-empty", ""]],
            }
        ]
    else:
        assert result == "client:site-alive : response\nclient:site-empty : No replies\n"
        assert conn.buffer.meta[MetaKey.CLIENT_STATUS] == {
            "site-alive": "response",
            "site-empty": MetaStatusValue.NO_REPLY,
        }


@pytest.mark.parametrize("target_type", ["server", "all"])
def test_server_restart_uses_selected_live_clients_not_stale_target_names(target_type, audit_records):
    _, read = audit_records
    module = TrainingCommandModule()
    selected_clients = [Client("site-alive", "secret-token-alive"), Client("site-gone", "secret-token-gone")]
    engine = ServerEngine.__new__(ServerEngine)
    engine.job_runner = SimpleNamespace(running_jobs={})
    engine.get_clients = MagicMock(return_value=selected_clients)
    engine.restart_server = MagicMock(return_value="")
    conn = Connection(
        app_ctx=engine,
        props={module.TARGET_CLIENTS: {"secret-token-alive": "stale-name", "secret-token-unselected": "private-path"}},
    )
    returned = ClientReply("secret-token-alive", "site-alive", None, ok_reply(body="response"))

    def disconnect_during_send(connection, message):
        selected_clients.pop()
        return [returned]

    module.send_request_to_clients = MagicMock(side_effect=disconnect_during_send)
    with _context("restart"):
        module.restart(conn, ["restart", target_type])
    records = read()
    assert [event["action"] for event in records] == ["client.restart.acknowledged", "client.restart.no_reply"]
    assert "site=site-alive" in records[0]["message"]
    assert "site=site-gone" in records[1]["message"]
    assert "stale-name" not in json.dumps(records)
    engine.get_clients.assert_called_once()
    engine.restart_server.assert_called_once()
    module.send_request_to_clients.assert_called_once()
    assert conn.get_prop(module.TARGET_CLIENTS) == {
        "secret-token-alive": "stale-name",
        "secret-token-unselected": "private-path",
    }
    assert conn.buffer.data[-1][ProtoKey.TYPE] == ProtoKey.SUCCESS


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
def test_control_unknown_codes_are_not_acknowledgements_and_tokens_are_deduplicated(operation, audit_records):
    _, read = audit_records
    missing = Message("reply", "private-path")
    unknown = Message("reply", "private-path")
    unknown.set_header(MsgHeader.RETURN_CODE, "private-path")
    replies = [
        ClientReply("secret-token-1", "unapproved-name", None, missing),
        ClientReply("secret-token-1", "unapproved-duplicate", None, ok_reply()),
        ClientReply("secret-token-2", "site-2", None, unknown),
        ClientReply("secret-token-other", "private-path", None, ok_reply()),
    ]
    with _context(operation):
        TrainingCommandModule._record_control_replies(
            operation, replies, {"secret-token-1": "site-1", "secret-token-2": "site-2", "secret-token-3": "site-3"}
        )
    records = read()
    assert [event["action"] for event in records] == [
        f"client.{operation}.reply_received",
        f"client.{operation}.reply_received",
        f"client.{operation}.no_reply",
    ]
    for event, site in zip(records, ["site-1", "site-2", "site-3"]):
        assert "site=" + site in event["message"]
    assert "unapproved-" not in json.dumps(records)


@pytest.mark.parametrize("operation", ["shutdown", "restart"])
@pytest.mark.parametrize("failure", ["getter", "iteration"])
def test_control_target_snapshot_failure_preserves_send_and_reply(operation, failure, audit_records):
    _, read = audit_records
    module = TrainingCommandModule()

    class BrokenMapping:
        def __iter__(self):
            raise RuntimeError("private-path")

    class AuditConnection(Connection):
        def get_prop(self, key, default=None):
            if key == module.TARGET_CLIENTS and failure == "getter":
                raise RuntimeError("private-path")
            return super().get_prop(key, default)

    conn = AuditConnection(
        app_ctx=MagicMock(spec=ServerEngine),
        props={module.TARGET_CLIENT_TOKENS: ["secret-token"], module.TARGET_CLIENTS: BrokenMapping()},
    )
    module.send_request_to_clients = MagicMock(
        return_value=[ClientReply("secret-token", "site-1", None, ok_reply(body="response"))]
    )
    with _context(operation):
        result = module._shutdown_app_on_clients(conn) if operation == "shutdown" else module._restart_clients(conn)
    module.send_request_to_clients.assert_called_once()
    assert result is True if operation == "shutdown" else result == "client:site-1 : response\n"
    assert read() == []
