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

"""Controlled administrative transfer boundaries; real transport lives in fast integration."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.streaming.transfer_outcome import RefOutcome, compute_transfer_outcome
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server import binary_transfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import audit_context, reject
from nvflare.fuel.sec.audit import Auditor, AuditService


@pytest.fixture
def audit_file(monkeypatch, tmp_path):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    yield path
    auditor.close()


def _connection(tmp_path, bind=True, with_file=True):
    folder = tmp_path / "download" / "command-tx" / "job-1"
    folder.mkdir(parents=True)
    if with_file:
        (folder / "private-result.txt").write_text("private-content-marker")
    manager = MagicMock(idle_timeout=30, monitor_interval=5)
    manager.bind_download.return_value = bind
    cell = MagicMock()
    cell.get_fqcn.return_value = "server"
    engine = SimpleNamespace(get_cell=lambda: cell)
    return Connection(
        props={
            ConnProps.DOWNLOAD_DIR: str(tmp_path / "download"),
            ConnProps.ENGINE: engine,
            ConnProps.ADMIN_SERVER: SimpleNamespace(timeout=10, sess_mgr=manager),
            ConnProps.SESSION: SimpleNamespace(sess_id="private-session-marker"),
        }
    )


def _records(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        assert set(record) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
        assert all(isinstance(value, str) for value in record.values())
        assert "private-" not in json.dumps(record)
    return records


@pytest.mark.parametrize(
    "done_status,receiver_status,expected_action,expected_reason",
    [
        ("finished", "success", "completed", "all_receivers_succeeded"),
        ("finished", "failed", "failed", "receiver_failed"),
        ("deleted", None, "aborted", "deleted"),
        ("timeout", None, "failed", "timeout"),
    ],
)
def test_terminal_callback_keeps_only_immutable_attribution(
    monkeypatch, tmp_path, audit_file, done_status, receiver_status, expected_action, expected_reason
):
    captured = {}

    class Downloader:
        tx_id = "stream-tx"

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def add_object(self, obj, ref_id=None):
            return "ref-1"

        def delete_transaction(self):
            pass

    monkeypatch.setattr(binary_transfer, "ObjectDownloader", Downloader)
    conn = _connection(tmp_path)
    with audit_context(user="verified@test.org", request_id="request-1", command="download_job", job_id="job-1"):
        binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
    assert not audit_file.read_text(), "preparing a transfer does not establish completion"
    callback = captured["outcome_cb"]
    assert len(callback.args) == 1
    context = callback.args[0]
    with pytest.raises(FrozenInstanceError):
        context.user = "forged@test.org"
    assert not any(isinstance(value, Connection) for value in vars(context).values())
    conn.set_prop(ConnProps.USER_NAME, "changed@test.org")
    refs = (RefOutcome("ref-1", {"receiver": receiver_status}),) if receiver_status else ()
    outcome = compute_transfer_outcome("stream-tx", done_status, 1, refs)
    with audit_context(user="unrelated@test.org", request_id="unrelated-request"):
        callback(outcome)
    records = _records(audit_file)
    assert len(records) == 1
    assert records[0]["action"] == f"download.transfer.{expected_action}"
    assert records[0]["user"] == "verified@test.org"
    assert records[0]["reference"] == "request-1"
    assert records[0]["job"] == "job-1"
    assert f"reason={expected_reason}" in records[0]["message"]
    assert "scope" not in records[0]
    assert "receipt" not in records[0]["message"]
    assert "extract" not in records[0]["message"]


@pytest.mark.parametrize("bind,with_file,reason", [(False, True, "session_inactive"), (True, False, "no_data")])
def test_setup_rejections_do_not_invent_completion(monkeypatch, tmp_path, audit_file, bind, with_file, reason):
    downloader = MagicMock(tx_id="stream-tx")
    monkeypatch.setattr(binary_transfer, "ObjectDownloader", lambda **kwargs: downloader)
    conn = _connection(tmp_path, bind=bind, with_file=with_file)
    with audit_context(user="verified@test.org", request_id="request-1"):
        binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
    records = _records(audit_file)
    assert len(records) == 1
    assert records[0]["action"] == "download.setup.rejected"
    assert records[0]["message"] == f"reason={reason}"
    downloader.delete_transaction.assert_called_once_with()


def test_assembly_error_keeps_existing_exception_and_cleanup(monkeypatch, tmp_path, audit_file):
    downloader = MagicMock(tx_id="stream-tx")
    monkeypatch.setattr(binary_transfer, "ObjectDownloader", lambda **kwargs: downloader)
    monkeypatch.setattr(binary_transfer, "add_file", MagicMock(side_effect=RuntimeError("private-error-marker")))
    conn = _connection(tmp_path)
    with audit_context(user="verified@test.org", request_id="request-1"):
        with pytest.raises(RuntimeError, match="private-error-marker"):
            binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
    records = _records(audit_file)
    assert len(records) == 1
    assert records[0]["action"] == "download.setup.rejected"
    assert records[0]["message"] == "reason=setup_error"
    downloader.delete_transaction.assert_called_once_with()


def test_terminal_audit_preparation_failure_does_not_escape_or_repeat_cleanup(monkeypatch, tmp_path, audit_file):
    from nvflare.fuel.sec import admin_audit

    failed = MagicMock()
    monkeypatch.setattr(admin_audit, "_failure", failed)
    manager = MagicMock()
    folder = tmp_path / "temporary-download"
    folder.mkdir()
    (folder / "file.txt").write_text("private-contents")
    transfer = binary_transfer.BinaryTransfer()
    transfer._cleanup_tx("stream-tx", "finished", [], str(folder), manager, "private-session")

    class BrokenOutcome:
        @property
        def status(self):
            raise RuntimeError("private-preparation-failure")

    with audit_context(user="admin@test.org", request_id="request-1"):
        binary_transfer._audit_download_outcome(admin_audit.snapshot(), BrokenOutcome())
    manager.end_download.assert_called_once_with("private-session", "stream-tx")
    assert not folder.exists()
    failed.assert_called_once_with()
    assert not audit_file.read_text()


def test_callback_snapshot_failure_preserves_transfer_setup(monkeypatch, tmp_path, audit_file):
    from nvflare.fuel.sec import admin_audit

    captured = {}
    downloader = MagicMock(tx_id="stream-tx")

    def create(**kwargs):
        captured.update(kwargs)
        return downloader

    failed = MagicMock()
    monkeypatch.setattr(admin_audit, "_failure", failed)
    monkeypatch.setattr(binary_transfer, "ObjectDownloader", create)
    monkeypatch.setattr(binary_transfer, "snapshot", MagicMock(side_effect=RuntimeError("private-context-error")))
    monkeypatch.setattr(binary_transfer, "add_file", lambda *args, **kwargs: "ref-1")
    conn = _connection(tmp_path)
    binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
    assert captured["outcome_cb"] is None
    assert conn.buffer.meta["files"] == [["private-result.txt", "ref-1"]]
    downloader.delete_transaction.assert_not_called()
    failed.assert_called_once_with()
    assert not audit_file.read_text()


@pytest.mark.parametrize("bind,with_file,reason", [(False, True, "session_inactive"), (True, False, "no_data")])
def test_reply_failure_preserves_one_deepest_setup_rejection(
    monkeypatch, tmp_path, audit_file, bind, with_file, reason
):
    downloader = MagicMock(tx_id="stream-tx")
    monkeypatch.setattr(binary_transfer, "ObjectDownloader", lambda **kwargs: downloader)
    conn = _connection(tmp_path, bind=bind, with_file=with_file)
    conn.append_error = MagicMock(side_effect=RuntimeError("private-reply-error"))
    with audit_context(user="verified@test.org", request_id="request-1"):
        with pytest.raises(RuntimeError, match="private-reply-error"):
            binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
        # An enclosing command wrapper may observe the same propagated failure.
        # It must not create another rejection for this setup decision.
        reject("download.setup.rejected", reason="preparation_error")
    records = _records(audit_file)
    assert len(records) == 1
    assert records[0]["message"] == f"reason={reason}"
    assert records[0]["reference"] == "request-1"
    downloader.delete_transaction.assert_called_once_with()


def test_assembly_rejection_is_not_repeated_by_enclosing_wrapper(monkeypatch, tmp_path, audit_file):
    downloader = MagicMock(tx_id="stream-tx")
    monkeypatch.setattr(binary_transfer, "ObjectDownloader", lambda **kwargs: downloader)
    monkeypatch.setattr(binary_transfer, "add_file", MagicMock(side_effect=RuntimeError("private-assembly-error")))
    conn = _connection(tmp_path)
    with audit_context(user="verified@test.org", request_id="request-1"):
        with pytest.raises(RuntimeError, match="private-assembly-error"):
            binary_transfer.BinaryTransfer().download_folder(conn, "command-tx", "job-1")
        reject("download.setup.rejected", reason="preparation_error")
    records = _records(audit_file)
    assert len(records) == 1
    assert records[0]["message"] == "reason=setup_error"
    downloader.delete_transaction.assert_called_once_with()
