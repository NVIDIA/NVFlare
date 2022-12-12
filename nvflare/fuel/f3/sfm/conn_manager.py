# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional

import uuid

import msgpack

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState, FrameReceiver, Bytes
from nvflare.fuel.f3.drivers.driver import Driver, ConnMonitor, Mode
from nvflare.fuel.f3.drivers.prefix import Prefix, PREFIX_LEN
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import MessageReceiver, Headers, Message
from nvflare.fuel.f3.sfm.constants import Types, HandshakeKeys
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection
from nvflare.fuel.f3.sfm.sfm_endpoint import SfmEndpoint

THREAD_POOL_SIZE = 16

log = logging.getLogger(__name__)


@dataclass
class Transport:
    """A listener or connector"""
    name: str
    driver: Driver
    mode: Mode
    total_conns: int
    curr_conns: int


class ConnManager:
    """SFM connection manager
    The class is responsible for maintaining state of SFM connections and pumping data through them
    """

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint

        # List of active transports (listener and connector)
        self.transports: List[Transport] = []

        # A dict of SFM connections, key is connection name
        self.connections: Dict[str, SfmConnection] = {}

        # A dict of SfmEndpoint for finding endpoint by name
        self.endpoints: Dict[str, SfmEndpoint] = {}

        # A list of Endpoint monitors
        self.monitors: List[EndpointMonitor] = []

        # App/receiver mapping
        self.receivers: Dict[int, MessageReceiver] = {}

        self.started = False
        self.executor = ThreadPoolExecutor(THREAD_POOL_SIZE, "conn_mgr")

    def add_transport(self, driver: Driver, mode: Mode) -> str:
        name = str(uuid.uuid4())
        transport = Transport(name, driver, mode, 0, 0)
        driver.register_conn_monitor(SfmConnMonitor(self, transport))
        self.transports.append(transport)

        log.debug(f"Transport {name} with driver {driver.get_name()} is created in {mode.name} mode")

        if self.started:
            self.start_transport(transport)

        return name

    def remove_transport(self, name: str):
        for index, trans in enumerate(self.transports):
            if name == trans.name:
                trans.driver.shutdown()
                self.transports.pop(index)
                log.info(f"Transport {name} with driver {trans.driver.get_name()} is removed")
                return

        log.error(f"Unknown transport name: {name}")

    def start(self):
        for trans in self.transports:
            self.start_transport(trans)

    def stop(self):
        pass

    def find_endpoint(self, name: str) -> Optional[Endpoint]:

        sfm_endpoint = self.endpoints.get(name)
        if not sfm_endpoint:
            log.debug("Endpoint {name} doesn't exist")
            return None

        return sfm_endpoint.endpoint

    def send_message(self, endpoint: Endpoint, app_id: int, headers: Headers, payload: Bytes):
        """Send a message to endpoint for app

        This method is similar to an HTTP request or RPC call.

        Args:
            endpoint: An endpoint to send the request to
            app_id: Application ID
            headers: headers, optional
            payload: message payload, optional

        Raises:
            CommError: If any error happens while sending the data
        """

        sfm_endpoint = self.endpoints.get(endpoint.name)
        if not sfm_endpoint:
            raise CommError(CommError.CLOSED, f"Endpoint {endpoint.name} not available, may be disconnected")

        state = sfm_endpoint.endpoint.state
        if state != EndpointState.READY:
            raise CommError(CommError.NOT_READY, f"Endpoint {endpoint.name} is not ready: {state}")

        stream_id = sfm_endpoint.next_stream_id()

        # When multiple connections, round-robin by stream ID
        sfm_conn = sfm_endpoint.get_connection(stream_id)
        if not sfm_conn:
            log.error("Logic error, ready endpoint has no connections")
            raise CommError(CommError.ERROR, f"Endpoint {endpoint.name} has no connection")

        sfm_conn.send_data(app_id, stream_id, headers, payload)

    def register_message_receiver(self, app_id: int, receiver: MessageReceiver):
        if self.receivers.get(app_id):
            raise CommError(CommError.BAD_DATA, f"Receiver for app {app_id} is already registered")

        self.receivers[app_id] = receiver

    def add_endpoint_monitor(self, monitor: EndpointMonitor):
        self.monitors.append(monitor)

    # Internal methods

    def start_transport(self, transport: Transport):
        """Start transport in a new thread"""

        self.executor.submit(self.start_transport_task, transport)

    def start_transport_task(self, transport: Transport):
        """Start transport in a new thread"""
        if transport.mode == Mode.ACTIVE:
            starter = transport.driver.connect
        else:
            starter = transport.driver.listen

        connected = False
        while not connected:
            try:
                starter({})
                connected = True
            except Exception as ex:
                log.error(f"Connection failed: {ex}")
                log.debug(traceback.format_exc())
                time.sleep(5)

        log.info(f"Transport {transport.driver.get_name()}:{transport.name} is started in {transport.mode.name} mode")

    def conn_state_change(self, connection: Connection, transport: Transport):
        state = connection.state

        if state == ConnState.CONNECTED:
            self.handle_new_connection(connection, transport)
            transport.total_conns += 1
            transport.curr_conns += 1
        elif state == ConnState.CLOSED:
            self.close_connection(connection)
            transport.curr_conns -= 1
        else:
            log.error(f"Unknown state: {state}")

    def process_frame(self, sfm_conn: SfmConnection, frame: Bytes):

        prefix = Prefix.from_bytes(frame)

        if prefix.header_len == 0:
            headers = None
        else:
            headers = msgpack.unpackb(frame[PREFIX_LEN:PREFIX_LEN + prefix.header_len])

        if prefix.type in (Types.HELLO, Types.READY):
            if prefix.type == Types.HELLO:
                sfm_conn.send_handshake(Types.READY)

            data = self.get_dict_payload(prefix, frame)
            self.update_endpoint(sfm_conn, data)

        elif prefix.type == Types.DATA:
            if prefix.length > PREFIX_LEN+prefix.header_len:
                payload = frame[PREFIX_LEN+prefix.header_len:]
            else:
                payload = None

            message = Message(headers, payload)
            receiver = self.receivers.get(prefix.app_id)
            if receiver:
                receiver.process_message(sfm_conn.endpoint.endpoint, prefix.app_id, message)
            else:
                log.debug(f"No receiver registered for App ID {prefix.app_id}, message ignored")

        else:
            log.error(f"Received unsupported frame type {prefix.type} on {sfm_conn.get_name()}")

    def update_endpoint(self, sfm_conn: SfmConnection, data):

        endpoint_name = data.pop(HandshakeKeys.ENDPOINT_NAME)
        if endpoint_name == self.local_endpoint.name:
            raise CommError(CommError.BAD_DATA, f"Duplicate endpoint name {endpoint_name}")

        endpoint = Endpoint(endpoint_name, data)
        endpoint.state = EndpointState.READY

        sfm_endpoint = self.endpoints.get(endpoint_name)
        if sfm_endpoint:
            old_state = sfm_endpoint.endpoint.state
            sfm_endpoint.endpoint = endpoint
        else:
            old_state = EndpointState.IDLE
            sfm_endpoint = SfmEndpoint(endpoint)

        sfm_endpoint.add_connection(sfm_conn)
        sfm_conn.endpoint = sfm_endpoint
        self.endpoints[sfm_endpoint.endpoint.name] = sfm_endpoint

        if endpoint.state != old_state:
            self.notify_monitors(endpoint)

    def notify_monitors(self, endpoint: Endpoint):

        if not self.monitors:
            log.debug("No endpoint monitor registered")
            return

        for monitor in self.monitors:
            monitor.state_change(endpoint)

    @staticmethod
    def get_dict_payload(prefix, frame):
        mv = memoryview(frame)
        return msgpack.unpackb(mv[(PREFIX_LEN+prefix.header_len):])

    def handle_new_connection(self, connection: Connection, transport: Transport):

        sfm_conn = SfmConnection(connection, self.local_endpoint, transport.mode)
        self.connections[sfm_conn.get_name()] = sfm_conn
        connection.register_frame_receiver(SfmFrameReceiver(self, sfm_conn))

        if sfm_conn.mode == Mode.ACTIVE:
            sfm_conn.send_handshake(Types.HELLO)

    def close_connection(self, connection: Connection):
        pass


class SfmConnMonitor(ConnMonitor):

    def __init__(self, conn_manager: ConnManager, transport: Transport):
        self.conn_manager = conn_manager
        self.transport = transport

    def state_change(self, connection: Connection):
        self.conn_manager.conn_state_change(connection, self.transport)


class SfmFrameReceiver(FrameReceiver):

    def __init__(self, conn_manager: ConnManager, conn: SfmConnection):
        self.conn_manager = conn_manager
        self.conn = conn

    def process_frame(self, frame: Bytes):
        self.conn_manager.process_frame(self.conn, frame)
