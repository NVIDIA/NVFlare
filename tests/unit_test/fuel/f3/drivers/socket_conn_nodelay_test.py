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
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.socket_conn import SocketConnection


def _make_connector():
    return SimpleNamespace(mode=Mode.ACTIVE, driver=SimpleNamespace(get_name=lambda: "tcp"))


@pytest.fixture
def tcp_socket_pair():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(listener.getsockname())
    accepted, _ = listener.accept()
    listener.close()
    try:
        yield client, accepted
    finally:
        client.close()
        accepted.close()


def test_tcp_no_delay_set_on_both_sides_by_default(tcp_socket_pair):
    client, accepted = tcp_socket_pair

    for sock in (client, accepted):
        # initial value is platform-dependent (macOS may pre-enable it on loopback)
        SocketConnection(sock, _make_connector())
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0


def test_tcp_no_delay_can_be_disabled(tcp_socket_pair, monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_tcp_no_delay", lambda self, default=True: False)
    client, _ = tcp_socket_pair
    initial = client.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

    SocketConnection(client, _make_connector())

    assert client.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == initial


def test_tcp_no_delay_skips_non_inet_sockets():
    left, right = socket.socketpair()
    try:
        # must not raise even though unix sockets have no TCP options
        SocketConnection(left, _make_connector())
    finally:
        left.close()
        right.close()
