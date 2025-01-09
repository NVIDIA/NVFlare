# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.fuel.f3.streaming.tools.utils import RX_CELL, TEST_CHANNEL, TEST_TOPIC, TX_CELL, make_buffer
from nvflare.fuel.utils.network_utils import get_open_ports

WAIT_SEC = 10


class State:
    def __init__(self):
        self.done = threading.Event()
        self.result = None


class TestStreamCell:
    @pytest.fixture(scope="session")
    def port(self):
        return get_open_ports(1)[0]

    @pytest.fixture(scope="session")
    def state(self):
        return State()

    @pytest.fixture(scope="session")
    def server_cell(self, port, state):
        listening_url = f"tcp://localhost:{port}"
        cell = CoreCell(RX_CELL, listening_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        stream_cell.register_blob_cb(TEST_CHANNEL, TEST_TOPIC, self.blob_cb, state=state)
        cell.start()

        yield stream_cell
        cell.stop()

    @pytest.fixture(scope="session")
    def client_cell(self, port, state):
        connect_url = f"tcp://localhost:{port}"
        cell = CoreCell(TX_CELL, connect_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        cell.start()

        yield stream_cell
        cell.stop()

    def test_streaming_blob(self, server_cell, client_cell, state):

        size = 64 * 1024 * 1024 + 123
        buffer = make_buffer(size)

        send_future = client_cell.send_blob(TEST_CHANNEL, TEST_TOPIC, RX_CELL, Message(None, buffer))
        bytes_sent = send_future.result()
        assert bytes_sent == len(buffer)

        if not state.done.wait(timeout=30):
            raise Exception("Data not received after 30 seconds")

        assert buffer == state.result

    def test_streaming_buffer_list(self, server_cell, client_cell, state):

        size = 64 * 1024 * 1024 + 123
        buffer = make_buffer(size)
        buf_list = []
        interval = int(size / 4)
        buf_list.append(buffer[0:interval])
        buf_list.append(buffer[interval : 2 * interval])
        buf_list.append(buffer[2 * interval : 3 * interval])
        buf_list.append(buffer[3 * interval : size])

        send_future = client_cell.send_blob(TEST_CHANNEL, TEST_TOPIC, RX_CELL, Message(None, buf_list))
        bytes_sent = send_future.result()
        assert bytes_sent == len(buffer)

        if not state.done.wait(timeout=30):
            raise Exception("Data not received after 30 seconds")

        assert buffer == state.result

    def blob_cb(self, future: StreamFuture, **kwargs):
        state = kwargs.get("state")
        state.result = future.result()
        state.done.set()
