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

"""Live F3 transport regressions, run serially in the fast integration matrix."""

import socket
import threading
import time

import pytest

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm import heartbeat_monitor
from nvflare.fuel.f3.sfm.conn_manager import FRAME_THREAD_POOL_SIZE
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.prefix import Prefix


@pytest.mark.parametrize("scheme", ["tcp", "atcp", "grpc", "agrpc", "http"])
def test_dropped_peer_reconnects_with_consistent_endpoint_state(scheme):
    states = {"server": [], "client": []}
    received = threading.Event()

    class Monitor:
        def state_change(self, endpoint):
            states[endpoint.name].append(endpoint.state)

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == b"after reconnect":
                received.set()

    def wait_for(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    server.register_monitor(Monitor())
    client.register_monitor(Monitor())
    server.register_message_receiver(531, Receiver())
    try:
        _, url, _ = server.start_listener(scheme, {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert wait_for(lambda: states["server"] and states["client"]), states
        assert states["server"][-1] == EndpointState.READY, states
        assert states["client"][-1] == EndpointState.READY, states

        # Closing the active peer must clear both endpoint views and allow its
        # connector to retry. The next connection must carry data normally.
        next(iter(client.conn_manager.sfm_conns.values())).conn.close()
        for name in ("server", "client"):
            assert wait_for(lambda name=name: EndpointState.DISCONNECTED in states[name]), states
            assert wait_for(
                lambda name=name: len(states[name]) >= 3 and states[name][-1] == EndpointState.READY
            ), states

        assert len(server.conn_manager.sfm_conns) == 1
        assert len(client.conn_manager.sfm_conns) == 1
        client.send(Endpoint("server"), 531, Message({}, b"after reconnect"))
        assert received.wait(5)
    finally:
        client.stop()
        server.stop()


@pytest.mark.parametrize("scheme", ["tcp", "atcp", "grpc", "agrpc", "http"])
def test_receive_idle_timeout_closes_server_connection_and_reconnects(scheme, monkeypatch):
    monkeypatch.setattr(heartbeat_monitor, "HEARTBEAT_TICK", 0.01)
    monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 0.03)
    monkeypatch.setattr(CommConfigurator, "get_sfm_receive_idle_timeout", lambda self, default: 1.0)
    monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: 100.0)
    monkeypatch.setattr(CommConfigurator, "get_sfm_close_stalled_connection", lambda self, default=False: False)

    states = {"server": [], "client": []}
    received = threading.Event()
    healthy_pongs = {"count": 0, "first_received_at": None}
    healthy_pongs_lock = threading.Lock()
    dropped_ping = threading.Event()

    class Monitor:
        def state_change(self, endpoint):
            states[endpoint.name].append(endpoint.state)

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == b"after idle-timeout reconnect":
                received.set()

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def reconnected(name):
        try:
            disconnected = states[name].index(EndpointState.DISCONNECTED)
        except ValueError:
            return False
        return EndpointState.READY in states[name][disconnected + 1 :]

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    server.register_monitor(Monitor())
    client.register_monitor(Monitor())
    server.register_message_receiver(532, Receiver())

    client_conn_manager = client.conn_manager
    original_process_frame = client_conn_manager.process_frame

    def observe_healthy_pongs(sfm_conn, frame):
        if Prefix.from_bytes(frame).type == Types.PONG:
            with healthy_pongs_lock:
                if healthy_pongs["first_received_at"] is None:
                    healthy_pongs["first_received_at"] = time.monotonic()
                healthy_pongs["count"] += 1
        original_process_frame(sfm_conn, frame)

    client_conn_manager.process_frame = observe_healthy_pongs
    client_heartbeat_monitor = client_conn_manager.heartbeat_monitor
    original_client_heartbeat_check = client_heartbeat_monitor._check_heartbeat

    try:
        _, url, _ = server.start_listener(scheme, {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()

        assert wait_for(lambda: states["server"] and states["client"]), states
        assert states["server"][-1] == EndpointState.READY, states
        assert states["client"][-1] == EndpointState.READY, states

        def healthy_heartbeats_span_idle_timeout():
            with healthy_pongs_lock:
                count = healthy_pongs["count"]
                first_received_at = healthy_pongs["first_received_at"]
            return count >= 3 and first_received_at is not None and time.monotonic() - first_received_at >= 1.2

        assert wait_for(
            healthy_heartbeats_span_idle_timeout, timeout=3
        ), f"{scheme} did not exchange healthy heartbeats across the idle timeout"
        assert EndpointState.DISCONNECTED not in states["server"], states

        # Keep the client's socket open, but stop it sending frames and ignore
        # server pings. This reproduces a silent established peer rather than
        # an ordinary transport close.
        client_heartbeat_monitor._check_heartbeat = lambda: None

        def ignore_server_ping(sfm_conn, frame):
            if Prefix.from_bytes(frame).type == Types.PING:
                dropped_ping.set()
                return
            observe_healthy_pongs(sfm_conn, frame)

        client_conn_manager.process_frame = ignore_server_ping
        assert dropped_ping.wait(3), f"{scheme} server did not send a heartbeat"
        assert wait_for(lambda: EndpointState.DISCONNECTED in states["server"]), states

        # Let the active connector retry and verify the closed server-side
        # connection does not leave the peer permanently disconnected.
        client_heartbeat_monitor._check_heartbeat = original_client_heartbeat_check
        client_conn_manager.process_frame = original_process_frame
        assert wait_for(lambda: reconnected("server") and reconnected("client"), timeout=10), states

        client.send(Endpoint("server"), 532, Message({}, b"after idle-timeout reconnect"))
        assert received.wait(5), f"{scheme} did not deliver data after reconnect"
    finally:
        client_heartbeat_monitor._check_heartbeat = original_client_heartbeat_check
        client_conn_manager.process_frame = original_process_frame
        client.stop()
        server.stop()


def test_async_tcp_receive_timeout_aborts_queued_output_and_reconnects(monkeypatch):
    monkeypatch.setattr(heartbeat_monitor, "HEARTBEAT_TICK", 0.01)
    monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 0.03)
    monkeypatch.setattr(CommConfigurator, "get_sfm_receive_idle_timeout", lambda self, default: 5.0)
    monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: 100.0)
    monkeypatch.setattr(CommConfigurator, "get_sfm_close_stalled_connection", lambda self, default=False: False)

    states = {"server": [], "client": []}
    received = threading.Event()
    forced_close_queued_bytes = []
    client_conn = None

    class Monitor:
        def state_change(self, endpoint):
            states[endpoint.name].append(endpoint.state)

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == b"after queued-output timeout":
                received.set()

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def reconnected(name):
        try:
            disconnected = states[name].index(EndpointState.DISCONNECTED)
        except ValueError:
            return False
        return EndpointState.READY in states[name][disconnected + 1 :]

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    server.register_monitor(Monitor())
    client.register_monitor(Monitor())
    server.register_message_receiver(533, Receiver())
    client_heartbeat_monitor = client.conn_manager.heartbeat_monitor
    original_client_heartbeat_check = client_heartbeat_monitor._check_heartbeat

    try:
        _, url, _ = server.start_listener("atcp", {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()

        assert wait_for(lambda: states["server"] and states["client"]), states
        assert states["server"][-1] == EndpointState.READY, states
        assert states["client"][-1] == EndpointState.READY, states

        client_sfm_conn = next(iter(client.conn_manager.sfm_conns.values()))
        client_conn = client_sfm_conn.conn
        client_heartbeat_monitor._check_heartbeat = lambda: None
        client_conn.writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)

        paused = threading.Event()
        transport = client_conn.writer.transport
        client_conn.aio_ctx.get_event_loop().call_soon_threadsafe(lambda: (transport.pause_reading(), paused.set()))
        assert paused.wait(5), "client did not pause its async TCP reader"

        server_sfm_conn = next(iter(server.conn_manager.sfm_conns.values()))
        server_sfm_conn.conn.writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        server_sfm_conn.record_receive_activity()
        server_transport = server_sfm_conn.conn.writer.transport
        server_loop = server_sfm_conn.conn.aio_ctx.get_event_loop()

        def queued_output_size():
            result = []
            measured = threading.Event()

            def read_size():
                try:
                    result.append(server_transport.get_write_buffer_size())
                except RuntimeError:
                    result.append(0)
                finally:
                    measured.set()

            server_loop.call_soon_threadsafe(read_size)
            assert measured.wait(2), "server event loop did not report queued output"
            return result[0]

        original_force_close = server_sfm_conn.conn.force_close

        def record_queued_bytes_and_force_close():
            forced_close_queued_bytes.append(queued_output_size())
            original_force_close()

        server_sfm_conn.conn.force_close = record_queued_bytes_and_force_close
        server.send(Endpoint("client"), 533, Message({}, b"x" * (32 * 1024 * 1024)))
        assert wait_for(
            lambda: queued_output_size() > 0, timeout=3
        ), "server did not queue output while the client reader was paused"
        assert not server_sfm_conn.receive_idle_timeout_triggered
        assert wait_for(lambda: server_sfm_conn.receive_idle_timeout_triggered, timeout=7), states
        assert wait_for(lambda: bool(forced_close_queued_bytes), timeout=2), "transport force-close did not run"
        assert forced_close_queued_bytes and forced_close_queued_bytes[0] > 0, forced_close_queued_bytes

        resumed = threading.Event()
        client_conn.aio_ctx.get_event_loop().call_soon_threadsafe(lambda: (transport.resume_reading(), resumed.set()))
        assert resumed.wait(5), "client did not resume its async TCP reader"
        for name in ("server", "client"):
            assert wait_for(lambda name=name: EndpointState.DISCONNECTED in states[name], timeout=5), states
        assert wait_for(lambda: reconnected("server") and reconnected("client"), timeout=10), states

        client.send(Endpoint("server"), 533, Message({}, b"after queued-output timeout"))
        assert received.wait(5), "message was not delivered after async TCP reconnect"
    finally:
        client_heartbeat_monitor._check_heartbeat = original_client_heartbeat_check
        if client_conn is not None:
            resumed = threading.Event()
            client_conn.aio_ctx.get_event_loop().call_soon_threadsafe(
                lambda: (client_conn.writer.transport.resume_reading(), resumed.set())
            )
            resumed.wait(2)
        client.stop()
        server.stop()


def test_ping_response_runs_when_application_frame_workers_are_saturated():
    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    ready = {"server": threading.Event(), "client": threading.Event()}

    class Monitor:
        def __init__(self, ready_event):
            self.ready_event = ready_event

        def state_change(self, endpoint):
            if endpoint.state == EndpointState.READY:
                self.ready_event.set()

    server.register_monitor(Monitor(ready["server"]))
    client.register_monitor(Monitor(ready["client"]))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    release_workers = threading.Event()
    try:
        _, url, _ = server.start_listener("tcp", {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert ready["server"].wait(10), "server did not accept client connection"
        assert ready["client"].wait(10), "client did not finish the handshake"

        client_manager = client.conn_manager
        workers_entered = threading.Event()
        entered_lock = threading.Lock()
        entered = {"count": 0}

        def hold_application_worker():
            with entered_lock:
                entered["count"] += 1
                if entered["count"] == FRAME_THREAD_POOL_SIZE:
                    workers_entered.set()
            release_workers.wait()

        for _ in range(FRAME_THREAD_POOL_SIZE):
            client_manager.frame_mgr_executor.submit(hold_application_worker)
        assert workers_entered.wait(10), f"only {entered['count']} frame workers started"

        server_sfm_conn = next(iter(server.conn_manager.sfm_conns.values()))
        client_sfm_conn = next(iter(client_manager.sfm_conns.values()))
        pong_send_started = threading.Event()
        original_send_heartbeat_async = client_sfm_conn.send_heartbeat_async

        def observe_pong_send(frame_type, data=None):
            if frame_type == Types.PONG:
                pong_send_started.set()
            return original_send_heartbeat_async(frame_type, data)

        client_sfm_conn.send_heartbeat_async = observe_pong_send
        previous_activity = server_sfm_conn.get_last_activity()
        server_sfm_conn.send_heartbeat(Types.PING)
        assert pong_send_started.wait(10), "PING was not handled while application workers were busy"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and server_sfm_conn.get_last_activity() <= previous_activity:
            time.sleep(0.01)
        assert server_sfm_conn.get_last_activity() > previous_activity, "PONG was queued behind application handlers"
        assert client_sfm_conn.sfm_endpoint is not None
    finally:
        release_workers.set()
        client.stop()
        server.stop()


def test_async_tcp_slow_frame_progress_prevents_receive_idle_timeout(monkeypatch):
    monkeypatch.setattr(heartbeat_monitor, "HEARTBEAT_TICK", 0.01)
    monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 0.05)
    monkeypatch.setattr(CommConfigurator, "get_sfm_receive_idle_timeout", lambda self, default: 0.25)

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    ready = {"server": threading.Event(), "client": threading.Event()}
    received = threading.Event()
    states = []
    payload = b"slow frame" * 200_000

    class Monitor:
        def state_change(self, endpoint):
            if endpoint.state == EndpointState.READY:
                ready[endpoint.name].set()
            states.append(endpoint.state)

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == payload:
                received.set()

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server.register_monitor(Monitor())
    client.register_monitor(Monitor())
    server.register_message_receiver(534, Receiver())
    try:
        _, url, _ = server.start_listener("atcp", {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert ready["server"].wait(10), "server async TCP connection did not become ready"
        assert ready["client"].wait(10), "client async TCP connection did not become ready"

        server_conn = next(iter(server.conn_manager.sfm_conns.values()))
        client_conn = next(iter(client.conn_manager.sfm_conns.values()))
        # This test sends in one direction, so the sender has no inbound application traffic
        # while its write is in progress. Keep its monitor running but leave enough idle time
        # there to isolate the receiver-side progress timeout under test.
        client.conn_manager.heartbeat_monitor.receive_idle_timeout = 100.0
        server_conn.conn.writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        original_progress = server_conn.conn.frame_receiver.on_receive_progress
        progress_count = 0
        progress_lock = threading.Lock()

        def slow_progress():
            nonlocal progress_count
            original_progress()
            with progress_lock:
                progress_count += 1
            time.sleep(0.06)

        server_conn.conn.frame_receiver.on_receive_progress = slow_progress
        sender = threading.Thread(target=client.send, args=(Endpoint("server"), 534, Message({}, payload)))
        started = time.monotonic()
        sender.start()

        assert received.wait(20), "slow frame was not delivered"
        sender.join(5)
        elapsed = time.monotonic() - started
        with progress_lock:
            observed_progress = progress_count

        assert not sender.is_alive(), "client send did not complete"
        assert elapsed > 0.25, f"test transfer did not exceed the receive idle timeout ({elapsed:.3f}s)"
        assert observed_progress > 1, f"reader did not report incremental progress ({observed_progress})"
        assert server_conn.receive_idle_timeout_triggered is False, states
        assert EndpointState.DISCONNECTED not in states, states
        assert client_conn.sfm_endpoint is not None
    finally:
        client.stop()
        server.stop()


def test_blocked_ping_send_does_not_stop_other_connection_deadlines(monkeypatch):
    monkeypatch.setattr(heartbeat_monitor, "HEARTBEAT_TICK", 0.01)
    monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 0.03)
    monkeypatch.setattr(CommConfigurator, "get_sfm_receive_idle_timeout", lambda self, default: 10.0)

    server = Communicator(Endpoint("server"))
    clients = [Communicator(Endpoint(f"client-{index}")) for index in range(2)]
    ready = threading.Event()
    connected = set()

    class Monitor:
        def state_change(self, endpoint):
            if endpoint.state == EndpointState.READY:
                connected.add(endpoint.name)
                if len(connected) == 2:
                    ready.set()

    server.register_monitor(Monitor())
    monitor = server.conn_manager.heartbeat_monitor
    original_check_heartbeat = monitor._check_heartbeat
    monitor._check_heartbeat = lambda: None
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    try:
        _, url, _ = server.start_listener("tcp", {"host": "127.0.0.1", "port": port})
        server.start()
        for client in clients:
            client.add_connector(url, Mode.ACTIVE)
            client.start()
        assert ready.wait(10), connected

        conns = list(server.conn_manager.sfm_conns.values())
        blocked_conn, expired_conn = conns
        with blocked_conn.activity_lock:
            blocked_conn.last_activity = time.monotonic() - 1
        with expired_conn.activity_lock:
            expired_conn.last_activity = time.monotonic() - 20
        blocked_conn.lock.acquire()
        try:
            monitor._check_heartbeat = original_check_heartbeat
            monitor.curr_time = time.monotonic()
            original_check_heartbeat()
            assert expired_conn.receive_idle_timeout_triggered, "deadline scan waited for a blocked heartbeat send"
        finally:
            blocked_conn.lock.release()
    finally:
        for client in clients:
            client.stop()
        server.stop()


def test_blocked_synchronous_tcp_send_does_not_block_receive_deadline(monkeypatch):
    monkeypatch.setattr(heartbeat_monitor, "HEARTBEAT_TICK", 0.01)
    monkeypatch.setattr(CommConfigurator, "get_heartbeat_interval", lambda self, default: 0.03)
    monkeypatch.setattr(CommConfigurator, "get_sfm_receive_idle_timeout", lambda self, default: 0.5)
    monkeypatch.setattr(CommConfigurator, "get_sfm_send_stall_timeout", lambda self, default: 100.0)

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    states = {"server": [], "client": []}
    ready = {"server": threading.Event(), "client": threading.Event()}
    reader_blocked = threading.Event()
    release_reader = threading.Event()
    received = threading.Event()

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def reconnected(name):
        try:
            disconnected = states[name].index(EndpointState.DISCONNECTED)
        except ValueError:
            return False
        return EndpointState.READY in states[name][disconnected + 1 :]

    class Monitor:
        def state_change(self, endpoint):
            states[endpoint.name].append(endpoint.state)
            if endpoint.state == EndpointState.READY:
                ready[endpoint.name].set()

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == b"after synchronous send timeout":
                received.set()

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server.register_monitor(Monitor())
    client.register_monitor(Monitor())
    server.register_message_receiver(535, Receiver())
    sender = None
    try:
        _, url, _ = server.start_listener("tcp", {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert ready["server"].wait(10), "server TCP connection did not become ready"
        assert ready["client"].wait(10), "client TCP connection did not become ready"

        server_conn = next(iter(server.conn_manager.sfm_conns.values()))
        client_conn = next(iter(client.conn_manager.sfm_conns.values()))
        client.conn_manager.heartbeat_monitor._check_heartbeat = lambda: None
        client_conn.conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        server_conn.conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

        original_process_frame = client_conn.conn.frame_receiver.process_frame

        def block_client_reader(frame):
            if Prefix.from_bytes(frame).type == Types.PING:
                reader_blocked.set()
                release_reader.wait(10)
            original_process_frame(frame)

        client_conn.conn.frame_receiver.process_frame = block_client_reader
        server_conn.send_heartbeat(Types.PING)
        assert reader_blocked.wait(5), "client did not pause after the heartbeat frame"

        # The peer keeps its TCP socket open but stops draining it. The large synchronous
        # write then owns SfmConnection.lock while the watchdog must still check the deadline.
        server_conn.record_receive_activity()

        def send_large_frame():
            server.send(Endpoint("client"), 535, Message({}, b"x" * (32 * 1024 * 1024)))

        sender = threading.Thread(target=send_large_frame)
        sender.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server_conn.get_send_stall_seconds() <= 0:
            time.sleep(0.01)
        assert server_conn.get_send_stall_seconds() > 0, "large TCP write did not start"
        send_lock_acquired = server_conn.lock.acquire(blocking=False)
        if send_lock_acquired:
            server_conn.lock.release()
        assert not send_lock_acquired, "large TCP write did not hold the send lock"

        assert wait_for(lambda: server_conn.receive_idle_timeout_triggered, timeout=5), (
            "heartbeat watchdog blocked behind the synchronous TCP send",
            states,
        )
        assert wait_for(lambda: EndpointState.DISCONNECTED in states["client"], timeout=5), states
        assert wait_for(lambda: not sender.is_alive(), timeout=5), "forced close did not release the TCP writer"

        release_reader.set()
        assert wait_for(lambda: reconnected("server") and reconnected("client"), timeout=10), states
        client.send(Endpoint("server"), 535, Message({}, b"after synchronous send timeout"))
        assert received.wait(5), "message was not delivered after synchronous TCP reconnect"
    finally:
        release_reader.set()
        client.stop()
        server.stop()
