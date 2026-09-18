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
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver


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
