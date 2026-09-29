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
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
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


def test_idle_tls_peer_cannot_pin_listener_shutdown(tmp_path):
    key, public_key = generate_keys()
    cert = generate_cert(Identity("localhost"), Identity("localhost"), key, public_key, ca=True)
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(serialize_cert(cert))
    key_file.write_bytes(serialize_pri_key(key))
    params = {"ca_cert": str(cert_file), "server_cert": str(cert_file), "server_key": str(key_file)}
    connector = MagicMock(params=dict(params, host="127.0.0.1", port=0, scheme="stcp"), stopped=Event())
    driver = TcpDriver()
    driver.register_conn_monitor(MagicMock())
    listening = Event()
    serve = TcpStreamServer.serve_forever

    def serve_and_signal(server):
        listening.set()
        serve(server, poll_interval=0.05)

    listener = Thread(target=driver.listen, args=(connector,), daemon=True)
    shutdown = Thread(target=driver.shutdown, daemon=True)
    peer = None
    try:
        with patch.object(TcpStreamServer, "serve_forever", serve_and_signal):
            listener.start()
            assert listening.wait(2)
            peer = socket.create_connection(driver.server.server_address, timeout=2)
            time.sleep(0.1)  # leave the peer in the TLS handshake, sending no bytes
            shutdown.start()
            shutdown.join(2)
            assert not shutdown.is_alive(), "an idle TLS handshake blocked shutdown"
    finally:
        if peer:
            peer.close()
        shutdown.join(3)
        listener.join(3)
        if driver.server:
            driver.server.server_close()
