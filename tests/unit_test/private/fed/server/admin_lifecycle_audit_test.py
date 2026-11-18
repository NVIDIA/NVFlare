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
from nvflare.fuel.sec.admin_audit import audit_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import error_reply, ok_reply
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
        ClientReply("secret-token", "site-1", None, ok_reply(body="private-path")),
        ClientReply("secret-token", "site-2", None, error_reply("private-path")),
        ClientReply("secret-token", "site-3", None, None),
    ]
    with _context(operation):
        TrainingCommandModule._record_control_replies(operation, replies)
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
        TrainingCommandModule._record_control_replies("shutdown", [ClientReply("secret-token", "site-1", None, reply)])
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
