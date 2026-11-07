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

from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.utils.network_utils import get_open_ports


def _fake_sockets():
    sockets = [Mock(), Mock()]
    for index, sock in enumerate(sockets):
        sock.getsockname.return_value = ("0.0.0.0", 20000 + index)
    return sockets


def test_get_open_ports_closes_sockets_after_success():
    sockets = _fake_sockets()
    with patch("nvflare.fuel.utils.network_utils.socket.socket", side_effect=sockets):
        assert get_open_ports(2) == [20000, 20001]
    for sock in sockets:
        sock.close.assert_called_once_with()


@pytest.mark.parametrize("failure_step", ["bind", "listen", "getsockname"])
def test_get_open_ports_closes_sockets_after_probe_failure(failure_step):
    sockets = _fake_sockets()
    getattr(sockets[1], failure_step).side_effect = OSError("socket failure")
    with patch("nvflare.fuel.utils.network_utils.socket.socket", side_effect=sockets):
        with pytest.raises(OSError, match="socket failure"):
            get_open_ports(2)
    for sock in sockets:
        sock.close.assert_called_once_with()


def test_get_open_ports_closes_existing_socket_if_creation_fails():
    first_socket = _fake_sockets()[0]
    with patch("nvflare.fuel.utils.network_utils.socket.socket", side_effect=[first_socket, OSError("no sockets")]):
        with pytest.raises(OSError, match="no sockets"):
            get_open_ports(2)
    first_socket.close.assert_called_once_with()


def test_get_open_ports_closes_sockets_when_not_enough_valid_ports():
    sockets = _fake_sockets()
    sockets[1].getsockname.return_value = ("0.0.0.0", 0)
    with patch("nvflare.fuel.utils.network_utils.socket.socket", side_effect=sockets):
        with pytest.raises(RuntimeError, match="Needed 2 but got 1"):
            get_open_ports(2)
    for sock in sockets:
        sock.close.assert_called_once_with()
