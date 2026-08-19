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
from unittest.mock import MagicMock

from nvflare.private.fed.client.fed_client_base import FederatedClientBase


def _make_client():
    client = FederatedClientBase.__new__(FederatedClientBase)
    client._shutdown_lock = threading.Lock()
    client.communicator = SimpleNamespace(heartbeat_done=False)
    client.cell = MagicMock()
    client.engine = None
    client.client_name = "site-1"
    client.logger = MagicMock()
    client.terminate = MagicMock()
    client.logout_client = MagicMock()
    return client


def test_send_request_before_shutdown_skips_after_close():
    client = _make_client()
    reply = MagicMock()
    client.cell.send_request.return_value = reply

    assert client.send_request_before_shutdown(topic="terminal_outcome") is reply

    client.close()

    assert client.communicator.heartbeat_done is True
    assert client.send_request_before_shutdown(topic="terminal_outcome") is None
    client.cell.send_request.assert_called_once_with(topic="terminal_outcome")
    client.logout_client.assert_called_once()


def test_close_waits_for_terminal_request_before_logout():
    client = _make_client()
    request_started = threading.Event()
    release_request = threading.Event()
    close_entered = threading.Event()

    def send_request(**_kwargs):
        request_started.set()
        assert release_request.wait(timeout=1.0)
        return MagicMock()

    client.cell.send_request.side_effect = send_request
    client.terminate.side_effect = close_entered.set

    request_thread = threading.Thread(target=client.send_request_before_shutdown, kwargs={"topic": "outcome"})
    request_thread.start()
    assert request_started.wait(timeout=1.0)

    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    assert not close_entered.wait(timeout=0.1)
    client.logout_client.assert_not_called()

    release_request.set()
    request_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert not close_thread.is_alive()
    assert client.communicator.heartbeat_done is True
    client.logout_client.assert_called_once()
