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

import logging
import time
from threading import Event

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


class CommState:
    def __init__(self):
        self.a_ready_event = Event()
        self.a_received_event = Event()
        self.b_ready_event = Event()
        self.b_received_event = Event()


class Monitor(EndpointMonitor):
    def __init__(self, comm_state: CommState):
        self.comm_state = comm_state

    def state_change(self, endpoint: Endpoint):
        if endpoint.state == EndpointState.READY:
            if endpoint.name == NODE_A:
                self.comm_state.a_ready_event.set()
            else:
                self.comm_state.b_ready_event.set()


class Receiver(MessageReceiver):
    def __init__(self, comm_state: CommState):
        self.comm_state = comm_state

    def process_message(self, endpoint: Endpoint, connection: Connection, app_id: int, message: Message):
        text = message.payload.decode("utf-8")
        if endpoint.name == NODE_A:
            assert text == MESSAGE_FROM_A
            self.comm_state.a_received_event.set()
        else:
            assert text == MESSAGE_FROM_B
            self.comm_state.b_received_event.set()


def get_comm_a(comm_state):
    local_endpoint = Endpoint(NODE_A, {"foo": "test"})
    comm = Communicator(local_endpoint)
    comm.register_monitor(Monitor(comm_state))
    comm.register_message_receiver(APP_ID, Receiver(comm_state))
    return comm


def get_comm_b(comm_state):
    local_endpoint = Endpoint(NODE_B, {"bar": 123})
    comm = Communicator(local_endpoint)
    comm.register_monitor(Monitor(comm_state))
    comm.register_message_receiver(APP_ID, Receiver(comm_state))
    return comm


class TestCommunicator:
    @pytest.mark.parametrize(
        "scheme, port_range",
        [
            ("tcp", "2000-3000"),
            ("grpc", "3000-4000"),
            # ("http", "3000-4000"), # TODO: HTTP is not working properly
            # ("atcp", "3000-4000"), # TODO: This test is hanging with Python 3.12
        ],
    )
    def test_sfm_message(self, scheme, port_range):
        comm_state = CommState()
        comm_a = get_comm_a(comm_state)
        comm_b = get_comm_b(comm_state)

        _, url, _ = comm_a.start_listener(scheme, {"ports": port_range})
        comm_a.start()

        # Check port is in the range
        if port_range:
            parts = port_range.split("-")
            lo = int(parts[0])
            hi = int(parts[1])
            params = parse_url(url)
            port = int(params.get("port"))
            assert lo <= port <= hi

        comm_b.add_connector(url, Mode.ACTIVE)
        comm_b.start()

        while not comm_state.a_ready_event.wait(10) or not comm_state.b_ready_event.wait(10):
            log.info("Waiting for both endpoints to be ready")
            time.sleep(0.1)

        comm_a.send(Endpoint(NODE_B), APP_ID, Message({}, MESSAGE_FROM_A.encode("utf-8")))
        comm_b.send(Endpoint(NODE_A), APP_ID, Message({}, MESSAGE_FROM_B.encode("utf-8")))

        time.sleep(1)

        assert comm_state.a_received_event.is_set() and comm_state.b_received_event.is_set()

        comm_b.stop()
        comm_a.stop()
