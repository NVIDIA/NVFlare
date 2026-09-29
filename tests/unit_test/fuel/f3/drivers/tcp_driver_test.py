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

from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context, parse_url
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver, TcpStreamServer
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


@pytest.mark.timeout(10)
def test_connect_finishing_after_shutdown_closes_socket(monkeypatch):
    driver = TcpDriver()
    driver.register_conn_monitor(MagicMock())
    with socket.socket() as listener, ThreadPoolExecutor(1) as executor:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        params = {"host": "127.0.0.1", "port": listener.getsockname()[1], "connect_timeout": 1}
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
            # The accepted socket can still complete TLS after the shutdown snapshot.
            context = get_ssl_context(connector.params, ssl_server=False)
            with context.wrap_socket(peer) as secured_peer:
                assert secured_peer.recv(1) == b""


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
