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

import threading
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.socket_conn import SocketConnection
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection


class FakeSocket:
    def __init__(self, send_returns=None):
        self.send_returns = list(send_returns or [])
        self.sent_chunks = []

    def getpeername(self):
        return ("127.0.0.1", 9001)

    def getsockname(self):
        return ("127.0.0.1", 9002)

    def fileno(self):
        return 99

    def shutdown(self, *_args, **_kwargs):
        return None

    def close(self):
        return None

    def send(self, view):
        self.sent_chunks.append(bytes(view))
        if self.send_returns:
            return self.send_returns.pop(0)
        return len(view)


class TestSocketConnectionSendTimeout:
    def _make_conn(self, monkeypatch, timeout_sec, send_returns=None):
        monkeypatch.setattr(CommConfigurator, "get_streaming_send_timeout", lambda self, default: float(timeout_sec))
        connector = SimpleNamespace(mode=Mode.ACTIVE, driver=SimpleNamespace(get_name=lambda: "tcp"))
        sock = FakeSocket(send_returns=send_returns)
        conn = SocketConnection(sock=sock, connector=connector, secure=False)
        return conn, sock

    def test_send_timeout_config_is_applied(self, monkeypatch):
        conn, _ = self._make_conn(monkeypatch, timeout_sec=1.75)
        assert abs(conn.send_timeout - 1.75) < 1e-9

    def test_send_frame_success_with_partial_writes(self, monkeypatch):
        conn, sock = self._make_conn(monkeypatch, timeout_sec=2.0, send_returns=[2, 3])

        monkeypatch.setattr(
            "nvflare.fuel.f3.drivers.socket_conn.select.select",
            lambda _r, _w, _x, _t: ([], [sock], []),
        )

        conn.send_frame(b"abcde")
        assert len(sock.sent_chunks) == 2
        # Negative path for close-on-timeout change: success send must not close.
        assert conn.closing is False

    def test_send_frame_timeout_when_socket_not_writable(self, monkeypatch):
        conn, sock = self._make_conn(monkeypatch, timeout_sec=0.01)

        monkeypatch.setattr(
            "nvflare.fuel.f3.drivers.socket_conn.select.select",
            lambda _r, _w, _x, _t: ([], [], []),
        )

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abc")

        assert ex.value.code == CommError.TIMEOUT
        assert "timeout" in str(ex.value).lower()

    def test_send_frame_timeout_closes_connection_to_avoid_frame_desync(self, monkeypatch):
        conn, sock = self._make_conn(monkeypatch, timeout_sec=0.01, send_returns=[2])

        select_calls = {"count": 0}

        def _select(_r, _w, _x, _t):
            select_calls["count"] += 1
            if select_calls["count"] == 1:
                return ([], [sock], [])
            return ([], [], [])

        monkeypatch.setattr("nvflare.fuel.f3.drivers.socket_conn.select.select", _select)

        close_calls = {"count": 0}
        original_close = conn.close

        def _close_spy():
            close_calls["count"] += 1
            original_close()

        monkeypatch.setattr(conn, "close", _close_spy)

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abcde")

        assert ex.value.code == CommError.TIMEOUT
        assert close_calls["count"] == 1
        assert conn.closing is True
        # First send() wrote partial bytes before timing out.
        assert len(sock.sent_chunks) == 1

    def test_send_frame_error_when_send_returns_zero(self, monkeypatch):
        conn, sock = self._make_conn(monkeypatch, timeout_sec=1.0, send_returns=[0])

        monkeypatch.setattr(
            "nvflare.fuel.f3.drivers.socket_conn.select.select",
            lambda _r, _w, _x, _t: ([], [sock], []),
        )

        close_calls = {"count": 0}
        original_close = conn.close

        def _close_spy():
            close_calls["count"] += 1
            original_close()

        monkeypatch.setattr(conn, "close", _close_spy)

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abc")

        assert ex.value.code == CommError.CLOSED
        assert "closed while sending" in str(ex.value).lower()
        # Negative path for close-on-timeout change: non-timeout comm errors should not force close.
        assert close_calls["count"] == 0
        assert conn.closing is False

    def test_send_frame_maps_timeouterror_to_timeout_and_closes(self, monkeypatch):
        conn, _ = self._make_conn(monkeypatch, timeout_sec=1.0)

        def _raise_timeout(*_args, **_kwargs):
            raise TimeoutError("socket timed out")

        monkeypatch.setattr(conn, "_send_with_timeout", _raise_timeout)

        close_calls = {"count": 0}
        original_close = conn.close

        def _close_spy():
            close_calls["count"] += 1
            original_close()

        monkeypatch.setattr(conn, "close", _close_spy)

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abc")

        assert ex.value.code == CommError.TIMEOUT
        assert "timeout" in str(ex.value).lower()
        assert close_calls["count"] == 1
        assert conn.closing is True

    def test_send_frame_maps_broken_pipe_to_closed(self, monkeypatch):
        conn, _ = self._make_conn(monkeypatch, timeout_sec=1.0)

        def _raise_closed(*_args, **_kwargs):
            raise BrokenPipeError("broken pipe")

        monkeypatch.setattr(conn, "_send_with_timeout", _raise_closed)

        close_calls = {"count": 0}
        original_close = conn.close

        def _close_spy():
            close_calls["count"] += 1
            original_close()

        monkeypatch.setattr(conn, "close", _close_spy)

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abc")

        assert ex.value.code == CommError.CLOSED
        assert "closed while sending" in str(ex.value).lower()
        assert close_calls["count"] == 0
        assert conn.closing is False

    def test_send_frame_wraps_unexpected_exception_as_error(self, monkeypatch):
        conn, _ = self._make_conn(monkeypatch, timeout_sec=1.0)

        def _raise_unexpected(*_args, **_kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(conn, "_send_with_timeout", _raise_unexpected)

        with pytest.raises(CommError) as ex:
            conn.send_frame(b"abc")

        assert ex.value.code == CommError.ERROR
        assert "error sending frame" in str(ex.value).lower()

    def test_send_frame_suppresses_error_when_closing(self, monkeypatch):
        conn, sock = self._make_conn(monkeypatch, timeout_sec=1.0, send_returns=[0])
        conn.closing = True

        monkeypatch.setattr(
            "nvflare.fuel.f3.drivers.socket_conn.select.select",
            lambda _r, _w, _x, _t: ([], [sock], []),
        )

        # should not raise because connection is already closing
        conn.send_frame(b"abc")

    def test_fix1_only_bounds_connection_lock_hold_time(self, monkeypatch):
        """Fix #1 only: bounded send timeout prevents prolonged HOL lock hold on one connection."""
        conn, sock = self._make_conn(monkeypatch, timeout_sec=0.03)
        sfm_conn = SfmConnection(conn=conn, local_endpoint=Endpoint("local"))

        # Socket never becomes writable -> send timeout path.
        monkeypatch.setattr(
            "nvflare.fuel.f3.drivers.socket_conn.select.select",
            lambda _r, _w, _x, _t: ([], [], []),
        )

        first_send_entered = threading.Event()
        original_send_with_timeout = conn._send_with_timeout

        def wrapped_send_with_timeout(frame, timeout_sec):
            first_send_entered.set()
            return original_send_with_timeout(frame, timeout_sec)

        conn._send_with_timeout = wrapped_send_with_timeout

        errors = []

        def do_send():
            try:
                sfm_conn.send_dict(Types.PING, 1, {"k": "v"})
            except Exception as ex:
                errors.append(ex)

        t1 = threading.Thread(target=do_send, daemon=True)
        t1.start()
        assert first_send_entered.wait(timeout=0.2)

        t2 = threading.Thread(target=do_send, daemon=True)
        t2.start()

        t1.join(timeout=0.5)
        t2.join(timeout=0.5)

        # Both sends should return quickly, rather than hanging for long periods.
        assert not t1.is_alive()
        assert not t2.is_alive()
        # First timeout closes the connection; subsequent send may be suppressed while closing.
        assert len(errors) == 1
