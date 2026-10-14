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
import os
import stat
import struct
import threading
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.file_driver import (
    OWNER_MARKER,
    READ_CHUNK,
    FileConnection,
    FileDriver,
    _ConnConfig,
    _LogReader,
    _LogWriter,
    parse_file_url,
)
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Message, MessageReceiver
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN

APP_ID = 321
NODE_A = "file-comm-a"
NODE_B = "file-comm-b"

FAST_PARAMS = {"poll_interval": "0.005", "lease_interval": "0.2", "lease_timeout": "5"}


class _State:
    def __init__(self):
        self.ready = {NODE_A: Event(), NODE_B: Event()}
        self.disconnected = {NODE_A: Event(), NODE_B: Event()}
        self.received_from = {NODE_A: [], NODE_B: []}


class _Monitor(EndpointMonitor):
    def __init__(self, state: _State):
        self.state = state

    def state_change(self, endpoint: Endpoint):
        if endpoint.state == EndpointState.READY:
            self.state.ready[endpoint.name].set()
        elif endpoint.state == EndpointState.DISCONNECTED:
            self.state.disconnected[endpoint.name].set()


class _Receiver(MessageReceiver):
    def __init__(self, state: _State):
        self.state = state

    def process_message(self, endpoint: Endpoint, connection: Connection, app_id: int, message: Message):
        self.state.received_from[endpoint.name].append(message.payload)


def _make_comm(name: str, state: _State) -> Communicator:
    comm = Communicator(Endpoint(name, {}))
    comm.register_monitor(_Monitor(state))
    comm.register_message_receiver(APP_ID, _Receiver(state))
    return comm


def _wait_until(condition, timeout=15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _frame(payload: bytes) -> bytes:
    length = PREFIX_LEN + len(payload)
    return struct.pack(">I", length) + bytes(PREFIX_LEN - 4) + payload


@pytest.mark.parametrize("operation", ["remove", "stop"])
def test_passive_reader_finishes_before_shutdown_returns(tmp_path, operation):
    state = _State()
    server = _make_comm(NODE_A, state)
    client = _make_comm(NODE_B, state)
    reader_closing = Event()
    release_reader = Event()
    completed = Event()
    worker = None

    try:
        handle, url, _ = server.start_listener("shared-file", {"root_dir": str(tmp_path), **FAST_PARAMS})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert state.ready[NODE_B].wait(5)

        driver = server.conn_manager.connectors[handle].driver
        assert _wait_until(lambda: bool(driver.connections))
        conn = next(iter(driver.connections.values()))
        assert _wait_until(lambda: conn.reader.file is not None)
        reader_thread = next(
            t for t in threading.enumerate() if t.name == f"file_conn_{os.path.basename(conn.conn_dir)}"
        )
        original_close = conn.reader.close

        def delayed_close():
            reader_closing.set()
            assert release_reader.wait(5)
            original_close()

        conn.reader.close = delayed_close

        def shut_down():
            if operation == "remove":
                server.remove_connector(handle)
            else:
                server.stop()
            completed.set()

        worker = threading.Thread(target=shut_down)
        worker.start()
        assert reader_closing.wait(5)
        assert not completed.wait(0.1), "shutdown returned before the passive reader closed its file"
        release_reader.set()
        worker.join(5)
        assert completed.is_set()
        assert not reader_thread.is_alive()
        assert conn.reader.file is None
        assert state.disconnected[NODE_B].is_set()
    finally:
        release_reader.set()
        client.stop()
        server.stop()
        if worker is not None:
            worker.join(5)


def test_active_reader_shutdown_waits_for_cleanup_not_worker_exit(tmp_path):
    (tmp_path / "conns").mkdir()
    driver = FileDriver()
    driver.register_conn_monitor(MagicMock())
    params = {"url": f"shared-file://0{tmp_path}", **FAST_PARAMS}
    connector = ConnectorInfo("active", driver, params, Mode.ACTIVE, 0, 0, False, Event())
    keep_worker_alive = Event()
    connect_returned = Event()
    shutdown_returned = Event()
    errors = []

    def connect():
        try:
            driver.connect(connector)
            connect_returned.set()
            keep_worker_alive.wait()
        except Exception as ex:
            errors.append(ex)

    def shut_down():
        try:
            driver.shutdown()
        except Exception as ex:
            errors.append(ex)
        finally:
            shutdown_returned.set()

    worker = threading.Thread(target=connect, daemon=True)
    stopper = threading.Thread(target=shut_down, daemon=True)
    worker.start()
    try:
        assert _wait_until(lambda: bool(driver.connections), timeout=5)
        conn = next(iter(driver.connections.values()))
        stopper.start()
        assert shutdown_returned.wait(3), "shutdown waited for the long-lived worker after connection cleanup"
        assert connect_returned.wait(3)
        assert worker.is_alive()
        assert conn.reader.file is None
        assert not driver.connections
        assert not errors
    finally:
        keep_worker_alive.set()
        driver.stop_event.set()
        worker.join(5)
        if stopper.ident is not None:
            stopper.join(5)


@pytest.mark.parametrize("operation", ["remove", "stop"])
def test_active_shared_file_connector_shutdown_returns_with_idle_executor(tmp_path, operation):
    (tmp_path / "conns").mkdir()
    manager = ConnManager(Endpoint(f"active_file_{operation}"))
    driver = FileDriver()
    params = {"scheme": "shared-file", "url": f"shared-file://0{tmp_path}", **FAST_PARAMS}
    handle = manager.add_connector(driver, params, Mode.ACTIVE)
    completed = Event()
    errors = []

    def shut_down():
        try:
            if operation == "remove":
                manager.remove_connector(handle)
            else:
                manager.stop()
        except Exception as ex:
            errors.append(ex)
        finally:
            completed.set()

    stopper = threading.Thread(target=shut_down, daemon=True)
    try:
        manager.start()
        assert _wait_until(lambda: bool(driver.connections), timeout=5)
        stopper.start()
        assert completed.wait(3), f"{operation} waited for the conn_mgr executor worker to exit"
        assert not errors
        assert not driver.connections
    finally:
        connector = manager.connectors.get(handle)
        if connector:
            connector.stopped.set()
        driver.stop_event.set()
        # If the old reader.join() regression returns, retiring the pool releases
        # its worker so the failed test cannot strand a non-daemon thread.
        manager.conn_mgr_executor.shutdown(wait=False)
        if stopper.ident is not None:
            stopper.join(5)
        manager.stop()


class TestParseFileUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("shared-file://0/lustre/proj/cellnet", "/lustre/proj/cellnet"),
            ("shared-file://0/lustre/proj/cellnet/", "/lustre/proj/cellnet"),
            ("shared-file://0/a/b?poll_interval=0.05&fsync=False", "/a/b"),
            ("shared-file://0/a#frag", "/a"),
        ],
    )
    def test_valid_urls(self, url, expected):
        assert parse_file_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "shared-file://lustre/proj/cellnet",  # non-0 authority, path would be silently wrong
            "shared-file:///lustre/proj/cellnet",  # empty authority
            "shared-file://0",  # no path
            "shared-file://0/",  # root path
            "shared-file://localhost/a/b",
            "tcp://host:1234",
            "shared-file:relative/path",
        ],
    )
    def test_invalid_urls(self, url):
        with pytest.raises(CommError):
            parse_file_url(url)


class TestLogRoundtrip:
    def test_rotation_and_cleanup(self, tmp_path):
        cfg = _ConnConfig({"max_log_size": 100})
        writer = _LogWriter(str(tmp_path), "active", cfg)
        reader = _LogReader(str(tmp_path), "active", cfg)

        frames = [_frame(f"payload-{i}".encode()) for i in range(20)]
        received = []
        for frame in frames:
            writer.append(frame)
            received.extend(reader.read_frames())
        received.extend(reader.read_frames())
        writer.close()
        reader.close()

        assert received == frames
        logs = [f for f in os.listdir(tmp_path) if f.endswith(".log")]
        assert len(logs) <= 2

    def test_frame_larger_than_log_cap(self, tmp_path):
        cfg = _ConnConfig({"max_log_size": 64})
        writer = _LogWriter(str(tmp_path), "active", cfg)
        reader = _LogReader(str(tmp_path), "active", cfg)

        frames = [_frame(bytes(1000)), _frame(b"small"), _frame(bytes(2000))]
        received = []
        for frame in frames:
            writer.append(frame)
            received.extend(reader.read_frames())
        received.extend(reader.read_frames())

        assert received == frames

    def test_reader_recovers_from_stale_file_handle(self, tmp_path):
        cfg = _ConnConfig({})
        writer = _LogWriter(str(tmp_path), "active", cfg)
        reader = _LogReader(str(tmp_path), "active", cfg)

        writer.append(_frame(b"first"))
        assert reader.read_frames() == [_frame(b"first")]

        os.close(reader.file.fileno())
        writer.append(_frame(b"second"))
        assert reader.read_frames() == []
        assert reader.read_frames() == [_frame(b"second")]

    def test_partial_frame_visibility(self, tmp_path):
        cfg = _ConnConfig({})
        reader = _LogReader(str(tmp_path), "active", cfg)
        frame = _frame(b"split-delivery")

        with open(os.path.join(tmp_path, "active.0.log"), "ab") as f:
            f.write(frame[:10])
            f.flush()
            assert reader.read_frames() == []
            f.write(frame[10:])
            f.flush()
        assert reader.read_frames() == [frame]


class TestFileDriver:
    def test_message_exchange_and_close(self, tmp_path):
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        try:
            resources = {"root_dir": str(tmp_path), **FAST_PARAMS}
            _, url, _ = comm_a.start_listener("shared-file", resources)
            assert url.startswith("shared-file://0/")
            comm_a.start()

            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()

            assert state.ready[NODE_A].wait(15)
            assert state.ready[NODE_B].wait(15)

            comm_a.send(Endpoint(NODE_B), APP_ID, Message({}, b"hello-from-a"))
            comm_b.send(Endpoint(NODE_A), APP_ID, Message({}, b"hello-from-b"))

            assert _wait_until(lambda: state.received_from[NODE_A] and state.received_from[NODE_B])
            assert state.received_from[NODE_A] == [b"hello-from-a"]
            assert state.received_from[NODE_B] == [b"hello-from-b"]

            comm_b.stop()
            assert state.disconnected[NODE_B].wait(15)
        finally:
            comm_b.stop()
            comm_a.stop()

    def test_many_messages_with_rotation(self, tmp_path):
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        try:
            resources = {"root_dir": str(tmp_path), "max_log_size": "300", **FAST_PARAMS}
            _, url, _ = comm_a.start_listener("shared-file", resources)
            comm_a.start()
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()

            assert state.ready[NODE_A].wait(15) and state.ready[NODE_B].wait(15)

            expected = [f"msg-{i}".encode() for i in range(40)] + [bytes(100 * 1024)]
            for payload in expected:
                comm_b.send(Endpoint(NODE_A), APP_ID, Message({}, payload))

            assert _wait_until(lambda: len(state.received_from[NODE_B]) == len(expected))
            assert sorted(state.received_from[NODE_B]) == sorted(expected)

            conn_dirs = list((tmp_path / next(p for p in os.listdir(tmp_path)) / "conns").iterdir())
            assert len(conn_dirs) == 1
            logs = [f for f in os.listdir(conn_dirs[0]) if f.startswith("active") and f.endswith(".log")]
            assert len(logs) <= 2
        finally:
            comm_b.stop()
            comm_a.stop()

    def test_reconnect(self, tmp_path):
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        comm_b2 = None
        try:
            resources = {"root_dir": str(tmp_path), **FAST_PARAMS}
            _, url, _ = comm_a.start_listener("shared-file", resources)
            comm_a.start()
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()

            assert state.ready[NODE_B].wait(15)
            comm_b.stop()
            assert state.disconnected[NODE_B].wait(15)

            state.ready[NODE_A].clear()
            state.ready[NODE_B].clear()
            comm_b2 = _make_comm(NODE_B, state)
            comm_b2.add_connector(url, Mode.ACTIVE)
            comm_b2.start()

            assert state.ready[NODE_B].wait(15)
            assert state.ready[NODE_A].wait(15)
            comm_b2.send(Endpoint(NODE_A), APP_ID, Message({}, b"after-reconnect"))
            assert _wait_until(lambda: b"after-reconnect" in state.received_from[NODE_B])
        finally:
            comm_b.stop()
            if comm_b2:
                comm_b2.stop()
            comm_a.stop()

    def test_missing_root_dir(self):
        comm = _make_comm(NODE_A, _State())
        with pytest.raises(CommError):
            comm.start_listener("shared-file", {})

    def test_close_drains_pending_frames(self, tmp_path):
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        try:
            _, url, _ = comm_a.start_listener("shared-file", {"root_dir": str(tmp_path), **FAST_PARAMS})
            comm_a.start()
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()
            assert state.ready[NODE_A].wait(15) and state.ready[NODE_B].wait(15)

            expected = [f"tail-{i}".encode() for i in range(30)]
            for payload in expected:
                comm_b.send(Endpoint(NODE_A), APP_ID, Message({}, payload))
            comm_b.stop()

            assert _wait_until(lambda: len(state.received_from[NODE_B]) == len(expected))
            assert sorted(state.received_from[NODE_B]) == sorted(expected)
        finally:
            comm_b.stop()
            comm_a.stop()

    def test_drain_delivers_final_frame_larger_than_read_chunk(self, tmp_path):
        conn_dir = tmp_path / "conn"
        conn_dir.mkdir()
        frame = _frame(bytes(READ_CHUNK + 1040))
        (conn_dir / "active.0.log").write_bytes(frame)
        (conn_dir / "closed").touch()

        connector = ConnectorInfo("test", None, {}, Mode.PASSIVE, 0, 0, False, Event())
        conn = FileConnection(str(conn_dir), connector, _ConnConfig({"poll_interval": "0.005"}))
        received = []
        conn.register_frame_receiver(SimpleNamespace(process_frame=received.append))

        conn._drain(Event())

        assert received == [frame]

    def test_permissions_ignore_umask(self, tmp_path):
        old_umask = os.umask(0o022)
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        def _mode(path):
            return stat.S_IMODE(os.stat(path).st_mode)

        try:
            _, url, _ = comm_a.start_listener("shared-file", {"root_dir": str(tmp_path), **FAST_PARAMS})
            comm_a.start()
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()
            assert state.ready[NODE_A].wait(15) and state.ready[NODE_B].wait(15)

            listen_dir = tmp_path / next(p for p in os.listdir(tmp_path))
            assert _mode(listen_dir) == 0o770
            conn_dir = next((listen_dir / "conns").iterdir())
            assert _mode(conn_dir) == 0o770
            assert _mode(conn_dir / "active.0.log") == 0o660
        finally:
            os.umask(old_umask)
            comm_b.stop()
            comm_a.stop()

    def test_active_cleans_up_when_listener_dead(self, tmp_path):
        listen_dir = tmp_path / "cell"
        (listen_dir / "conns").mkdir(parents=True)
        state = _State()
        comm_b = _make_comm(NODE_B, state)
        url = f"shared-file://0{listen_dir}?poll_interval=0.005&lease_interval=0.2&lease_timeout=1"

        try:
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()
            assert _wait_until(lambda: len(os.listdir(listen_dir / "conns")) > 0, 5)
            assert _wait_until(lambda: len(os.listdir(listen_dir / "conns")) == 0, 15)
        finally:
            comm_b.stop()


class TestGetUrls:
    def test_gc_requires_ownership_marker(self, tmp_path):
        old = time.time() - 7200
        foreign = tmp_path / "lst_foreign"
        foreign.mkdir()
        (foreign / "data.txt").write_text("keep me")
        os.utime(foreign, (old, old))

        dead = tmp_path / "lst_dead"
        (dead / "conns").mkdir(parents=True)
        (dead / OWNER_MARKER).touch()
        (dead / "lease").touch()
        os.utime(dead / "lease", (old, old))

        FileDriver.get_urls("shared-file", {"root_dir": str(tmp_path)})

        assert foreign.exists() and (foreign / "data.txt").exists()
        assert _wait_until(lambda: not dead.exists(), 5)

    def test_reserved_chars_rejected(self, tmp_path):
        for ch in ("?", "#"):
            with pytest.raises(CommError):
                FileDriver.get_urls("shared-file", {"root_dir": str(tmp_path) + f"/bad{ch}dir"})


class TestConnConfig:
    def test_robust_param_parsing(self):
        cfg = _ConnConfig({"max_log_size": "1000000000.0", "poll_interval": "0", "fsync": "False"})
        assert cfg.max_log_size == 1_000_000_000
        assert cfg.poll_interval >= 0.001
        assert cfg.max_poll_interval >= cfg.poll_interval
        assert cfg.fsync is False
        assert _ConnConfig({"fsync": "true"}).fsync is True
        assert _ConnConfig({"dir_mode": "700", "file_mode": "600"}).dir_mode == 0o700
