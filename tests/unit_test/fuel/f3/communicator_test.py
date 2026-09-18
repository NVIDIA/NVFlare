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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import parse_url
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Message, MessageReceiver
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.f3.sfm.constants import FLARE_PROTOCOL_VERSION, HandshakeKeys, Types
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection

log = logging.getLogger(__name__)

APP_ID = 123
NODE_A = "communicator-a"
NODE_B = "communicator-b"
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
    def test_data_during_handshake_does_not_close_connection(self, monkeypatch):
        comm = Communicator(Endpoint("server"))
        peer = MagicMock(connector=SimpleNamespace(params={}))
        peer.get_conn_properties.return_value = {}
        sender = SfmConnection(peer, Endpoint("site-1"))
        incoming = SfmConnection(peer, comm.local_endpoint)
        receiver = MagicMock()
        comm.register_message_receiver(APP_ID, receiver)
        sender.send_data(APP_ID, 1, {}, b"first request")
        data = peer.send_frame.call_args.args[0]
        original_update = comm.conn_manager.update_endpoint

        def delayed_update(*args):
            # READY has been sent, but attachment has not finished on its frame worker.
            comm.conn_manager.frame_mgr_executor.submit(comm.conn_manager.process_frame_task, incoming, data).result(
                timeout=2
            )
            peer.close.assert_not_called()
            receiver.process_message.assert_not_called()
            original_update(*args)

        monkeypatch.setattr(comm.conn_manager, "update_endpoint", delayed_update)
        try:
            sender.send_handshake(Types.HELLO)
            comm.conn_manager.process_frame_task(incoming, peer.send_frame.call_args.args[0])
            assert incoming.sfm_endpoint is not None
            comm.conn_manager.process_frame_task(incoming, data)
            receiver.process_message.assert_called_once()
            peer.close.assert_not_called()
        finally:
            comm.stop()

    @pytest.mark.parametrize("quiet", [False, True])
    def test_connector_retry_logging(self, caplog, quiet):
        stopped = Event()
        driver = MagicMock()
        driver.connect.side_effect = ConnectionRefusedError("server unavailable")
        params = {DriverParams.URL.value: "tcp://localhost:8002"}
        if quiet:
            params[DriverParams.QUIET_RECONNECT.value] = True
        connector = ConnectorInfo("test", driver, params, Mode.ACTIVE, 0, 0, False, stopped)
        waits = []

        def wait(delay):
            waits.append(delay)
            if len(waits) == 4:
                stopped.set()

        stopped.wait = wait
        with caplog.at_level(logging.DEBUG, logger="nvflare.fuel.f3.sfm.conn_manager"):
            ConnManager.start_connector_task(connector)
        assert waits == [1, 2, 4, 8]
        records = [r for r in caplog.records if "failed with exception" in r.message or "Retrying" in r.message]
        assert len(records) == 8
        assert max(r.levelno for r in records) == (logging.DEBUG if quiet else logging.ERROR)

    @pytest.mark.parametrize("frame_type", [Types.HELLO, Types.READY])
    @pytest.mark.parametrize("version", [None, 3, "2", 2.0])
    def test_protocol_gate_preserves_existing_connection(self, frame_type, version):
        comm = Communicator(Endpoint("server"))
        peer = MagicMock(connector=SimpleNamespace(params={}))
        peer.get_conn_properties.return_value = {}
        sender = SfmConnection(peer, Endpoint("site-1"))
        active = SfmConnection(peer, comm.local_endpoint)
        try:
            sender.send_handshake(frame_type)
            comm.conn_manager.process_frame_task(active, peer.send_frame.call_args.args[0])
            endpoint = comm.find_endpoint("site-1")
            assert endpoint.state == EndpointState.READY
            assert endpoint.get_prop(HandshakeKeys.FLARE_PROTOCOL) == FLARE_PROTOCOL_VERSION
            data = {HandshakeKeys.ENDPOINT_NAME: "site-1"}
            if version is not None:
                data[HandshakeKeys.FLARE_PROTOCOL] = version
            sender.send_dict(frame_type, 1, data)
            rejected = SfmConnection(MagicMock(), comm.local_endpoint)
            comm.conn_manager.process_frame_task(rejected, peer.send_frame.call_args.args[0])
            rejected.conn.close.assert_called_once()
            rejected.conn.send_frame.assert_not_called()
            peer.close.assert_not_called()
            assert comm.find_endpoint("site-1") is endpoint
            assert comm.conn_manager.get_connections("site-1") == [active]
            assert rejected.sfm_endpoint is None
        finally:
            comm.stop()

    @pytest.mark.parametrize(
        "scheme, port_range",
        [
            ("tcp", "2000-3000"),
            ("grpc", "3000-4000"),
            ("http", "4000-5000"),
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
