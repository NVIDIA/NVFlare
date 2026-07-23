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
import time
from threading import Event

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.file_driver import (
    OWNER_MARKER,
    FileDriver,
    _ConnConfig,
    _LogReader,
    _LogWriter,
    parse_file_url,
)
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Message, MessageReceiver
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


class TestParseFileUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("file://0/lustre/proj/cellnet", "/lustre/proj/cellnet"),
            ("file://0/lustre/proj/cellnet/", "/lustre/proj/cellnet"),
            ("file://0/a/b?poll_interval=0.05&fsync=False", "/a/b"),
            ("file://0/a#frag", "/a"),
        ],
    )
    def test_valid_urls(self, url, expected):
        assert parse_file_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "file://lustre/proj/cellnet",  # non-0 authority, path would be silently wrong
            "file:///lustre/proj/cellnet",  # empty authority
            "file://0",  # no path
            "file://0/",  # root path
            "file://localhost/a/b",
            "tcp://host:1234",
            "file:relative/path",
        ],
    )
    def test_invalid_urls(self, url):
        with pytest.raises(CommError):
            parse_file_url(url)


class TestLogRoundtrip:
    def test_rotation_and_cleanup(self, tmp_path):
        cfg = _ConnConfig({"max_log_size": 100})
        writer = _LogWriter(str(tmp_path), "a2p", cfg)
        reader = _LogReader(str(tmp_path), "a2p", cfg)

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
        writer = _LogWriter(str(tmp_path), "a2p", cfg)
        reader = _LogReader(str(tmp_path), "a2p", cfg)

        frames = [_frame(bytes(1000)), _frame(b"small"), _frame(bytes(2000))]
        received = []
        for frame in frames:
            writer.append(frame)
            received.extend(reader.read_frames())
        received.extend(reader.read_frames())

        assert received == frames

    def test_partial_frame_visibility(self, tmp_path):
        cfg = _ConnConfig({})
        reader = _LogReader(str(tmp_path), "a2p", cfg)
        frame = _frame(b"split-delivery")

        with open(os.path.join(tmp_path, "a2p.0.log"), "ab") as f:
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
            _, url, _ = comm_a.start_listener("file", resources)
            assert url.startswith("file://0/")
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
            _, url, _ = comm_a.start_listener("file", resources)
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
            logs = [f for f in os.listdir(conn_dirs[0]) if f.startswith("a2p") and f.endswith(".log")]
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
            _, url, _ = comm_a.start_listener("file", resources)
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
            comm.start_listener("file", {})

    def test_close_drains_pending_frames(self, tmp_path):
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        try:
            _, url, _ = comm_a.start_listener("file", {"root_dir": str(tmp_path), **FAST_PARAMS})
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

    def test_permissions_ignore_umask(self, tmp_path):
        old_umask = os.umask(0o022)
        state = _State()
        comm_a = _make_comm(NODE_A, state)
        comm_b = _make_comm(NODE_B, state)

        def _mode(path):
            return stat.S_IMODE(os.stat(path).st_mode)

        try:
            _, url, _ = comm_a.start_listener("file", {"root_dir": str(tmp_path), **FAST_PARAMS})
            comm_a.start()
            comm_b.add_connector(url, Mode.ACTIVE)
            comm_b.start()
            assert state.ready[NODE_A].wait(15) and state.ready[NODE_B].wait(15)

            listen_dir = tmp_path / next(p for p in os.listdir(tmp_path))
            assert _mode(listen_dir) == 0o770
            conn_dir = next((listen_dir / "conns").iterdir())
            assert _mode(conn_dir) == 0o770
            assert _mode(conn_dir / "a2p.0.log") == 0o660
        finally:
            os.umask(old_umask)
            comm_b.stop()
            comm_a.stop()

    def test_active_cleans_up_when_listener_dead(self, tmp_path):
        listen_dir = tmp_path / "cell"
        (listen_dir / "conns").mkdir(parents=True)
        state = _State()
        comm_b = _make_comm(NODE_B, state)
        url = f"file://0{listen_dir}?poll_interval=0.005&lease_interval=0.2&lease_timeout=1"

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

        FileDriver.get_urls("file", {"root_dir": str(tmp_path)})

        assert foreign.exists() and (foreign / "data.txt").exists()
        assert _wait_until(lambda: not dead.exists(), 5)

    def test_reserved_chars_rejected(self, tmp_path):
        for ch in ("?", "#"):
            with pytest.raises(CommError):
                FileDriver.get_urls("file", {"root_dir": str(tmp_path) + f"/bad{ch}dir"})


class TestConnConfig:
    def test_robust_param_parsing(self):
        cfg = _ConnConfig({"max_log_size": "1000000000.0", "poll_interval": "0", "fsync": "False"})
        assert cfg.max_log_size == 1_000_000_000
        assert cfg.poll_interval >= 0.001
        assert cfg.max_poll_interval >= cfg.poll_interval
        assert cfg.fsync is False
        assert _ConnConfig({"fsync": "true"}).fsync is True
        assert _ConnConfig({"dir_mode": "700", "file_mode": "600"}).dir_mode == 0o700
