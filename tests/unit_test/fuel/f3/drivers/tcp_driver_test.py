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

import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.f3.drivers import tcp_driver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context, parse_url
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver, TcpStreamServer
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


@pytest.mark.timeout(10)
@pytest.mark.parametrize("timeout", [None, 1, "1"])
def test_connect_finishing_after_shutdown_closes_socket(monkeypatch, timeout):
    driver = TcpDriver()
    driver.register_conn_monitor(MagicMock())
    with socket.socket() as listener, ThreadPoolExecutor(1) as executor:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        params = {"host": "127.0.0.1", "port": listener.getsockname()[1], "connect_timeout": timeout}
        connector = ConnectorInfo("test", driver, params, Mode.ACTIVE, 0, 0, False, Event())
        add_connection = driver.add_connection

        def stop_before_registration(connection):
            assert connection.sock.gettimeout() is None
            connector.stopped.set()
            driver.shutdown()  # The newly connected socket is not registered yet.
            add_connection(connection)

        monkeypatch.setattr(driver, "add_connection", stop_before_registration)
        future = executor.submit(driver.connect, connector)
        connection, _ = listener.accept()
        with connection:
            future.result(timeout=3)  # Must finish while the remote socket remains open.
            connection.settimeout(1)
            assert connection.recv(1) == b""
        assert not driver.connections


@pytest.mark.timeout(10)
@pytest.mark.parametrize("pause_at", ["construction", "stop_check"])
def test_listener_starting_during_shutdown_releases_socket(monkeypatch, pause_at):
    driver = TcpDriver()
    connector = ConnectorInfo(
        "test", driver, {"scheme": "tcp", "host": "127.0.0.1", "port": 0}, Mode.PASSIVE, 0, 0, False, Event()
    )
    paused, release = Event(), Event()
    servers = []

    def pause():
        paused.set()
        assert release.wait(3)

    def construct(*args):
        server = TcpStreamServer(*args)
        servers.append(server)
        if pause_at == "construction":
            pause()
        return server

    is_stopped = connector.stopped.is_set

    def check_stopped():
        pause()
        return is_stopped()

    monkeypatch.setattr(tcp_driver, "TcpStreamServer", construct)
    if pause_at == "stop_check":
        monkeypatch.setattr(connector.stopped, "is_set", check_stopped)
    listener = Thread(target=driver.listen, args=(connector,), daemon=True)
    shutdown = Thread(target=driver.shutdown, daemon=True)
    try:
        listener.start()
        assert paused.wait(2)
        connector.stopped.set()
        shutdown.start()
        if pause_at == "construction":
            shutdown.join(2)
            assert not shutdown.is_alive()
        release.set()
        listener.join(2)
        shutdown.join(2)
        assert not listener.is_alive(), "listener started after shutdown"
        assert not shutdown.is_alive(), "shutdown waited for a listener that never started"
        assert servers[0].socket.fileno() == -1
    finally:
        release.set()
        if listener.is_alive() and servers:
            servers[0].shutdown()
        listener.join(2)
        if shutdown.is_alive() and servers:
            # Release a waiter stranded by a regressed early return before serve_forever().
            servers[0]._BaseServer__is_shut_down.set()
        if shutdown.ident is not None:
            shutdown.join(2)
        for server in servers:
            server.server_close()


@pytest.fixture
def tls_listener(tmp_path):
    key, public_key = generate_keys()
    cert = generate_cert(Identity("localhost"), Identity("localhost"), key, public_key, ca=True)
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(serialize_cert(cert))
    key_file.write_bytes(serialize_pri_key(key))
    params = {"ca_cert": str(cert_file), "server_cert": str(cert_file), "server_key": str(key_file)}
    driver = TcpDriver()
    connector = ConnectorInfo(
        "test", driver, dict(params, host="127.0.0.1", port=0, scheme="stcp"), Mode.PASSIVE, 0, 0, False, Event()
    )
    driver.register_conn_monitor(MagicMock())
    driver.server = TcpStreamServer(driver, connector)
    listener = Thread(target=driver.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    listener.start()
    try:
        yield driver, connector
    finally:
        connector.stopped.set()
        driver.shutdown()
        listener.join(3)
        assert not listener.is_alive()


def test_idle_tls_peer_cannot_pin_or_survive_listener_shutdown(tls_listener):
    driver, connector = tls_listener
    handshaking = Event()
    do_handshake = ssl.SSLSocket.do_handshake

    def handshake_and_signal(sock, *args, **kwargs):
        if sock.server_side:
            handshaking.set()
        return do_handshake(sock, *args, **kwargs)

    shutdown = Thread(target=driver.shutdown, daemon=True)
    with patch.object(ssl.SSLSocket, "do_handshake", handshake_and_signal):
        with socket.create_connection(driver.server.server_address, timeout=2) as peer:
            assert handshaking.wait(2)
            connector.stopped.set()
            shutdown.start()
            shutdown.join(2)
            assert not shutdown.is_alive(), "an idle TLS handshake blocked shutdown"
            assert driver.server.socket.fileno() == -1
            # The driver owns and closes the pending handshake, even though no
            # completed connection has entered its connection registry yet.
            assert peer.recv(1) == b""


def test_tls_handshake_honors_longer_url_connection_timeout(tls_listener):
    driver, connector = tls_listener
    connector.params.update(parse_url("stcp://localhost:0?connect_timeout=10"))
    context = get_ssl_context(connector.params, ssl_server=False)
    with socket.create_connection(driver.server.server_address, timeout=2) as peer:
        time.sleep(5.2)  # A legitimate handshake can exceed the former fixed five-second limit.
        with context.wrap_socket(peer) as secured_peer:
            secured_peer.sendall(b"x")


def test_idle_tls_peer_is_closed_at_configured_timeout(tls_listener):
    driver, connector = tls_listener
    connector.params["connect_timeout"] = 0.1
    with socket.create_connection(driver.server.server_address, timeout=2) as peer:
        assert peer.recv(1) == b""
