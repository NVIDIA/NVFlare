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

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.streaming.download_service import ProduceRC, _PropKey
from nvflare.fuel.f3.streaming.file_downloader import FileDownloadable
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server.binary_transfer import BinaryTransfer
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.sess import SessionManager
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_service_no_monitor, pull_request


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


def test_binary_transfer_binds_verified_progress_and_detaches_on_failure(monkeypatch, tmp_path):
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

    session_mgr = MagicMock(idle_timeout=9)
    session_mgr.bind_download.return_value = True
    admin = SimpleNamespace(timeout=20, sess_mgr=session_mgr)
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
    assert created["progress_interval"] == 3.0
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


def test_logout_after_expiry_does_not_use_stopped_messenger():
    api = AdminAPI.__new__(AdminAPI)
    api.closed = True
    api.in_logout = False
    api.session_expired_reason = "idle timeout"
    api.server_execute = MagicMock()

    assert api.logout() is None
    api.server_execute.assert_not_called()
