# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import time

import pytest

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.net_utils import parse_url
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Message, MessageReceiver

log = logging.getLogger(__name__)

APP_ID = 123
NODE_A = "Communicator A"
NODE_B = "Communicator B"
MESSAGE_FROM_A = "Test message from a"
MESSAGE_FROM_B = "Test message from b"


class Monitor(EndpointMonitor):
    def __init__(self, tester):
        self.tester = tester

    def state_change(self, endpoint: Endpoint):
        if endpoint.state == EndpointState.READY:
            if endpoint.name == NODE_A:
                self.tester.a_ready = True
            else:
                self.tester.b_ready = True


class Receiver(MessageReceiver):
    def __init__(self, tester):
        self.tester = tester

    def process_message(self, endpoint: Endpoint, connection: Connection, app_id: int, message: Message):
        text = message.payload.decode("utf-8")
        if endpoint.name == NODE_A:
            assert text == MESSAGE_FROM_A
            self.tester.a_received = True
        else:
            assert text == MESSAGE_FROM_B
            self.tester.b_received = True


class TestCommunicator:
    @pytest.fixture
    def comm_a(self):
        local_endpoint = Endpoint(NODE_A, {"foo": "test"})
        comm = Communicator(local_endpoint)
        comm.register_monitor(Monitor(self))
        comm.register_message_receiver(APP_ID, Receiver(self))
        self.a_ready = False
        self.a_received = False
        return comm

    @pytest.fixture
    def comm_b(self):
        local_endpoint = Endpoint(NODE_B, {"bar": 123})
        comm = Communicator(local_endpoint)
        comm.register_monitor(Monitor(self))
        comm.register_message_receiver(APP_ID, Receiver(self))
        self.b_ready = False
        self.b_received = False
        return comm

    @pytest.mark.parametrize(
        "scheme, port_range",
        [
            ("tcp", "2000-3000"),
        ],
    )
    def test_sfm_message(self, comm_a, comm_b, scheme, port_range):

        handle1, url = comm_a.start_listener(scheme, {"ports": port_range})
        comm_a.start()

        # Check port is in the range
        parts = port_range.split("-")
        lo = int(parts[0])
        hi = int(parts[1])
        params = parse_url(url)
        port = int(params.get("port"))
        assert lo <= port <= hi

        comm_b.add_connector(url, Mode.ACTIVE)
        comm_b.start()

        while not self.a_ready or not self.b_ready:
            log.info("Waiting for both endpoints to be ready")
            time.sleep(0.1)

        comm_a.send(Endpoint(NODE_B), APP_ID, Message({}, MESSAGE_FROM_A.encode("utf-8")))

        comm_b.send(Endpoint(NODE_A), APP_ID, Message({}, MESSAGE_FROM_B.encode("utf-8")))

        time.sleep(1)

        assert self.a_received and self.b_received

        comm_b.stop()
        comm_a.stop()
