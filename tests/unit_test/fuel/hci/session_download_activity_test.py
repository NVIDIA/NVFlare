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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.streaming.download_service import ProduceRC, _PropKey
from nvflare.fuel.f3.streaming.file_downloader import FileDownloadable
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.file_transfer import FileTransferModule
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from nvflare.fuel.hci.server.binary_transfer import BinaryTransfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.sess import SessionManager
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    make_service_no_monitor,
    pull_request,
    run_monitor_once,
)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Cell:
    def __init__(self):
        self.events = []

    def fire_and_forget(self, **kwargs):
        self.events.append(kwargs)


def _new_session(manager, name, cert_exp=None):
    return manager.create_session(name, "org", "role", f"admin/{name}", cert_exp=cert_exp)


def _new_manager(monkeypatch, cell, clock, idle_timeout=1800):
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.time.time", clock)
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.threading.Thread.start", lambda _thread: None)
    return SessionManager(cell, idle_timeout=idle_timeout)


def test_progress_keeps_only_the_owning_session_alive_past_idle_timeout(monkeypatch):
    clock = _Clock()
    manager = _new_manager(monkeypatch, _Cell(), clock, idle_timeout=10)
    progressing = _new_session(manager, "progressing")
    idle = _new_session(manager, "idle")
    progressing_cancel = MagicMock()
    idle_cancel = MagicMock()
    assert manager.bind_download(progressing.sess_id, "tx-progressing", progressing_cancel)
    assert manager.bind_download(idle.sess_id, "tx-idle", idle_cancel)

    clock.advance(9)
    assert not manager.mark_download_active(idle.sess_id, "tx-progressing")
    assert not manager.mark_download_active(progressing.sess_id, "unbound-tx")
    assert manager.mark_download_active(progressing.sess_id, "tx-progressing")
    clock.advance(2)
    manager.check_sessions()

    assert manager.get_sessions() == [progressing]
    progressing_cancel.assert_not_called()
    idle_cancel.assert_called_once_with()

    clock.advance(9)
    assert manager.mark_download_active(progressing.sess_id, "tx-progressing")
    manager.check_sessions()
    assert manager.get_sessions() == [progressing]


def test_stalled_download_and_certificate_expiry_end_the_activity_scope(monkeypatch):
    clock = _Clock()
    cell = _Cell()
    manager = _new_manager(monkeypatch, cell, clock, idle_timeout=10)
    stalled = _new_session(manager, "stalled")
    expired_cert = _new_session(manager, "expired-cert", cert_exp=5)
    stalled_cancel = MagicMock()
    expired_cancel = MagicMock()
    assert manager.bind_download(stalled.sess_id, "tx-stalled", stalled_cancel)
    assert manager.bind_download(expired_cert.sess_id, "tx-expired", expired_cancel)

    clock.advance(5)
    assert not manager.mark_download_active(expired_cert.sess_id, "tx-expired")
    manager.check_sessions()
    expired_cancel.assert_called_once_with()

    clock.advance(6)
    manager.check_sessions()
    stalled_cancel.assert_called_once_with()
    assert manager.get_sessions() == []
    assert len(cell.events) == 2


def test_logout_cancels_bound_download_but_completed_download_is_detached(monkeypatch):
    cell = _Cell()
    manager = _new_manager(monkeypatch, cell, _Clock())
    active = _new_session(manager, "active")
    completed = _new_session(manager, "completed")
    active_cancel = MagicMock()
    completed_cancel = MagicMock()
    assert manager.bind_download(active.sess_id, "tx-active", active_cancel)
    assert manager.bind_download(completed.sess_id, "tx-completed", completed_cancel)

    manager.end_download(completed.sess_id, "tx-completed")
    manager.end_session_by_id(active.sess_id)
    manager.end_session_by_id(completed.sess_id)

    active_cancel.assert_called_once_with()
    completed_cancel.assert_not_called()
    assert cell.events == []


def test_session_teardown_isolates_notification_and_cancel_failures(monkeypatch):
    cell = MagicMock()
    cell.fire_and_forget.side_effect = RuntimeError("notification failed")
    manager = _new_manager(monkeypatch, cell, _Clock())
    session = _new_session(manager, "admin")
    failing_cancel = MagicMock(side_effect=RuntimeError("cancel failed"))
    successful_cancel = MagicMock()
    assert manager.bind_download(session.sess_id, "tx-failing", failing_cancel)
    assert manager.bind_download(session.sess_id, "tx-successful", successful_cancel)

    manager.end_session_by_id(session.sess_id, "idle timeout")

    failing_cancel.assert_called_once_with()
    successful_cancel.assert_called_once_with()
    assert manager.get_sessions() == []
    assert manager.downloads == {}


@pytest.mark.parametrize(
    (
        "admin_timeout",
        "session_idle_timeout",
        "session_monitor_interval",
        "expected_download_timeout",
        "expected_progress_interval",
    ),
    [
        (10, 1800, 5, 1805, 30.0),
        (2400, 1800, 5, 2400, 30.0),
        (20, 9, 5, 20, 3.0),
    ],
)
def test_binary_transfer_binds_verified_progress_and_detaches_on_failure(
    monkeypatch,
    tmp_path,
    admin_timeout,
    session_idle_timeout,
    session_monitor_interval,
    expected_download_timeout,
    expected_progress_interval,
):
    download_root = tmp_path / "download"
    source_folder = download_root / "command-tx" / "job-1"
    source_folder.mkdir(parents=True)
    (source_folder / "result.zip").write_bytes(b"result")

    created = {}

    class _Downloader:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.tx_id = "stream-tx"
            self.delete_transaction = MagicMock()

        def add_object(self, obj, ref_id=None):
            return "ref-1"

    monkeypatch.setattr("nvflare.fuel.hci.server.binary_transfer.ObjectDownloader", _Downloader)

    session_mgr = MagicMock(idle_timeout=session_idle_timeout, monitor_interval=session_monitor_interval)
    session_mgr.bind_download.return_value = True
    admin = SimpleNamespace(timeout=admin_timeout, sess_mgr=session_mgr)
    cell = MagicMock()
    cell.get_fqcn.return_value = "server"
    engine = MagicMock()
    engine.get_cell.return_value = cell
    session = SimpleNamespace(sess_id="session-1")
    conn = Connection(
        props={
            ConnProps.DOWNLOAD_DIR: str(download_root),
            ConnProps.ENGINE: engine,
            ConnProps.ADMIN_SERVER: admin,
            ConnProps.SESSION: session,
        }
    )

    BinaryTransfer().download_folder(conn, "command-tx", "job-1")

    session_mgr.bind_download.assert_called_once()
    assert created["timeout"] == expected_download_timeout
    assert created["progress_interval"] == expected_progress_interval
    progress_cb = created["progress_cb"]
    progress_cb(tx_id="stream-tx", bytes_done=0, state=TransferProgressState.ACTIVE)
    progress_cb(tx_id="stream-tx", bytes_done=10, state=TransferProgressState.ACTIVE)
    progress_cb(tx_id="stream-tx", bytes_done=10, state=TransferProgressState.COMPLETED)
    session_mgr.mark_download_active.assert_called_once_with("session-1", "stream-tx")

    created["transaction_done_cb"](
        "stream-tx",
        "timed_out",
        [],
        tx_path=created["tx_path"],
        session_mgr=created["session_mgr"],
        session_id=created["session_id"],
    )
    session_mgr.end_download.assert_called_once_with("session-1", "stream-tx")


def test_binary_transfer_cancels_bound_download_when_setup_fails(monkeypatch, tmp_path):
    download_root = tmp_path / "download"
    source_folder = download_root / "command-tx" / "job-1"
    source_folder.mkdir(parents=True)
    (source_folder / "result.zip").write_bytes(b"result")

    created = {}

    class _Downloader:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.tx_id = "stream-tx"
            self.delete_transaction = MagicMock()
            created["downloader"] = self

    monkeypatch.setattr("nvflare.fuel.hci.server.binary_transfer.ObjectDownloader", _Downloader)
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.binary_transfer.add_file", MagicMock(side_effect=RuntimeError("setup failed"))
    )

    session_mgr = MagicMock(idle_timeout=1800, monitor_interval=5)
    session_mgr.bind_download.return_value = True
    admin = SimpleNamespace(timeout=10, sess_mgr=session_mgr)
    cell = MagicMock()
    cell.get_fqcn.return_value = "server"
    engine = MagicMock()
    engine.get_cell.return_value = cell
    session = SimpleNamespace(sess_id="session-1")
    conn = Connection(
        props={
            ConnProps.DOWNLOAD_DIR: str(download_root),
            ConnProps.ENGINE: engine,
            ConnProps.ADMIN_SERVER: admin,
            ConnProps.SESSION: session,
        }
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        BinaryTransfer().download_folder(conn, "command-tx", "job-1")

    session_mgr.bind_download.assert_called_once()
    created["downloader"].delete_transaction.assert_called_once_with()


def test_result_download_reference_survives_default_retry_envelope_and_resumes(monkeypatch, tmp_path):
    clock = _Clock()
    session_mgr = _new_manager(monkeypatch, _Cell(), clock, idle_timeout=1800)
    session = _new_session(session_mgr, "admin")
    service = make_service_no_monitor()
    created = {}

    download_root = tmp_path / "download"
    source_folder = download_root / "command-tx" / "job-1"
    source_folder.mkdir(parents=True)
    source = source_folder / "result.bin"
    expected = b"0123456789abcdef"
    source.write_bytes(expected)

    class _Downloader:
        def __init__(
            self,
            cell,
            timeout,
            num_receivers,
            transaction_done_cb,
            progress_cb,
            progress_interval,
            **cb_kwargs,
        ):
            created["timeout"] = timeout
            self.tx_id = service.new_transaction(
                cell=cell,
                timeout=timeout,
                num_receivers=num_receivers,
                transaction_done_cb=transaction_done_cb,
                progress_cb=progress_cb,
                progress_interval=progress_interval,
                **cb_kwargs,
            )
            created["tx_id"] = self.tx_id

        def add_object(self, obj, ref_id=None):
            ref_id = service.add_object(self.tx_id, obj, ref_id=ref_id)
            created["ref_id"] = ref_id
            return ref_id

        def delete_transaction(self):
            service.delete_transaction(self.tx_id)

    def _add_small_file(downloader, file_name):
        return downloader.add_object(FileDownloadable(file_name, chunk_size=4))

    monkeypatch.setattr("nvflare.fuel.hci.server.binary_transfer.ObjectDownloader", _Downloader)
    monkeypatch.setattr("nvflare.fuel.hci.server.binary_transfer.add_file", _add_small_file)

    admin = SimpleNamespace(timeout=10, sess_mgr=session_mgr)
    cell = MagicMock()
    cell.get_fqcn.return_value = "server"
    engine = MagicMock()
    engine.get_cell.return_value = cell
    conn = Connection(
        props={
            ConnProps.DOWNLOAD_DIR: str(download_root),
            ConnProps.ENGINE: engine,
            ConnProps.ADMIN_SERVER: admin,
            ConnProps.SESSION: session,
        }
    )

    BinaryTransfer().download_folder(conn, "command-tx", "job-1")

    tx_id = created["tx_id"]
    ref_id = created["ref_id"]
    assert created["timeout"] == session_mgr.idle_timeout + session_mgr.monitor_interval

    first_reply = service._handle_download(pull_request(ref_id, "admin-cell"))
    assert first_reply.payload[_PropKey.STATUS] == ProduceRC.OK
    received = bytearray(first_reply.payload[_PropKey.DATA])
    committed_state = first_reply.payload[_PropKey.STATE]

    # Simulate all three timed-out requests failing to reach the producer. The final
    # permitted request starts at 29 seconds and can arrive near its 5-second deadline.
    for quiet_seconds in (7, 9, 13, 5):
        clock.advance(quiet_seconds)
        run_monitor_once(service, now=clock.now)
        session_mgr.check_sessions()
        assert service.get_transaction_id(ref_id) == tx_id

    state = committed_state
    while True:
        reply = service._handle_download(pull_request(ref_id, "admin-cell", state=state))
        status = reply.payload[_PropKey.STATUS]
        if status == ProduceRC.EOF:
            break
        assert status == ProduceRC.OK
        received.extend(reply.payload[_PropKey.DATA])
        state = reply.payload[_PropKey.STATE]

    assert len(received) == len(expected)
    assert hashlib.sha256(received).digest() == hashlib.sha256(expected).digest()

    # Legacy receivers settle on the monitor pass after the terminal reply has left
    # the request path. Drive that pass explicitly because this test suppresses the
    # real monitor thread.
    run_monitor_once(service, now=service._tx_table[tx_id].last_active_time)
    assert not source_folder.exists()
    assert session_mgr.downloads == {}


def test_progressing_download_outlives_idle_timeout_with_matching_size_and_hash(monkeypatch, tmp_path):
    clock = _Clock()
    manager = _new_manager(monkeypatch, _Cell(), clock, idle_timeout=5)
    session = _new_session(manager, "admin")
    source = tmp_path / "result.bin"
    expected = b"0123456789abcdef"
    source.write_bytes(expected)
    service = make_service_no_monitor()

    def progress_cb(tx_id, bytes_done, state, **kwargs):
        if bytes_done > 0 and state == TransferProgressState.ACTIVE:
            manager.mark_download_active(session.sess_id, tx_id)

    tx_id = service.new_transaction(
        cell=MagicMock(),
        timeout=100,
        num_receivers=1,
        progress_cb=progress_cb,
        progress_interval=0,
        transaction_done_cb=lambda finished_tx_id, *_args: manager.end_download(session.sess_id, finished_tx_id),
    )
    ref_id = service.add_object(tx_id, FileDownloadable(str(source), chunk_size=4))
    assert manager.bind_download(session.sess_id, tx_id, lambda: service.delete_transaction(tx_id))

    received = bytearray()
    state = None
    while True:
        reply = service._handle_download(pull_request(ref_id, "admin-cell", state=state))
        status = reply.payload[_PropKey.STATUS]
        if status == ProduceRC.EOF:
            break
        received.extend(reply.payload[_PropKey.DATA])
        state = reply.payload[_PropKey.STATE]
        clock.advance(4)
        manager.check_sessions()
        assert manager.get_sessions() == [session]

    assert clock.now > manager.idle_timeout
    assert len(received) == len(expected)
    assert hashlib.sha256(received).digest() == hashlib.sha256(expected).digest()
    service.delete_transaction(tx_id)
    assert manager.downloads == {}


def test_session_expiry_aborts_active_download():
    api = AdminAPI.__new__(AdminAPI)
    api.session_expired_reason = None
    api.session_abort_signal = Signal()
    api.debug = MagicMock()
    api.close = MagicMock()
    api.fire_session_event = MagicMock()

    message = SimpleNamespace(payload="idle timeout")
    api._handle_session_expired(message)

    assert api.session_expired_reason == "idle timeout"
    assert api.session_abort_signal.triggered
    assert api.session_abort_signal.value == "idle timeout"
    api.close.assert_called_once_with()


def test_failed_download_removes_partial_file(monkeypatch, tmp_path):
    partial_file = tmp_path / "partial"
    partial_file.write_bytes(b"incomplete")
    monkeypatch.setattr(
        "nvflare.fuel.hci.client.api.downloader.download_file",
        lambda **_kwargs: ("aborted", str(partial_file)),
    )
    api = AdminAPI.__new__(AdminAPI)
    api.cell = MagicMock()
    api.file_download_progress_timeout = 1
    api.session_abort_signal = Signal()
    api.logger = MagicMock()
    api._print_hci = MagicMock()

    assert api.download_file("server", "ref-1", str(tmp_path / "result")) is None
    assert not partial_file.exists()


def test_pull_folder_failure_removes_transaction_directory(tmp_path):
    module = FileTransferModule(str(tmp_path), str(tmp_path))
    tx_path = tmp_path / "job-1__tx-1"
    tx_path.mkdir()
    (tx_path / "previous-result").write_bytes(b"complete")
    failure = {ProtoKey.STATUS: APIStatus.ERROR_RUNTIME, ProtoKey.DETAILS: "download failed"}
    api = MagicMock()
    api.server_execute.return_value = {
        ProtoKey.STATUS: APIStatus.SUCCESS,
        ProtoKey.META: {
            MetaKey.FILES: [["result", "ref-1"]],
            MetaKey.TX_ID: "tx-1",
            MetaKey.SOURCE_FQCN: "server",
        },
    }
    api.do_command.return_value = failure
    command_entry = MagicMock()
    command_entry.full_command_name.return_value = "download_job"
    ctx = MagicMock()
    ctx.get_command_entry.return_value = command_entry
    ctx.get_api.return_value = api

    assert module.pull_folder(["pull_folder", "job-1"], ctx) == failure
    assert not tx_path.exists()


def test_logout_after_expiry_does_not_use_stopped_messenger():
    api = AdminAPI.__new__(AdminAPI)
    api.closed = True
    api.in_logout = False
    api.session_expired_reason = "idle timeout"
    api.server_execute = MagicMock()

    assert api.logout() is None
    api.server_execute.assert_not_called()
