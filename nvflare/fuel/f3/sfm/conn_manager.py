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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import msgpack

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection, ConnState, FrameReceiver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver import ConnMonitor, Driver
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import ssl_required
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Headers, Message, MessageReceiver
from nvflare.fuel.f3.sfm.constants import HandshakeKeys, Types
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection
from nvflare.fuel.f3.sfm.sfm_endpoint import SfmEndpoint
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.security.logging import secure_format_exception, secure_format_traceback

FRAME_THREAD_POOL_SIZE = 100
CONN_THREAD_POOL_SIZE = 16
INIT_WAIT = 1
MAX_WAIT = 60
SILENT_RECONNECT_TIME = 5
SELF_ADDR = "0.0.0.0:0"

log = logging.getLogger(__name__)

handle_lock = threading.Lock()
handle_count = 0


def get_handle():
    global handle_lock, handle_count
    with handle_lock:
        handle_count += 1

    return "CH%05d" % handle_count


class ConnManager(ConnMonitor):
    """SFM connection manager
    The class is responsible for maintaining state of SFM connections and pumping data through them
    """

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint

        # Active connectors
        self.connectors: Dict[str, ConnectorInfo] = {}

        # A dict of SFM connections, key is connection name
        self.sfm_conns: Dict[str, SfmConnection] = {}

        # A dict of SfmEndpoint for finding endpoint by name
        self.sfm_endpoints: Dict[str, SfmEndpoint] = {}

        # A list of Endpoint monitors
        self.monitors: List[EndpointMonitor] = []

        # App/receiver mapping
        self.receivers: Dict[int, MessageReceiver] = {}

        self.started = False
        self.conn_mgr_executor = ThreadPoolExecutor(CONN_THREAD_POOL_SIZE, "conn_mgr")
        self.frame_mgr_executor = ThreadPoolExecutor(FRAME_THREAD_POOL_SIZE, "frame_mgr")
        self.lock = threading.Lock()
        self.null_conn = NullConnection()
        stats = StatsPoolManager.get_pool("sfm_send_frame")
        if not stats:
            stats = StatsPoolManager.add_time_hist_pool(
                "sfm_send_frame", "SFM send_frame time in secs", scope=local_endpoint.name
            )
        self.send_frame_stats = stats

    def add_connector(self, driver: Driver, params: dict, mode: Mode) -> str:

        # Validate parameters
        capabilities = driver.capabilities()
        support_ssl = capabilities.get(DriverCap.SUPPORT_SSL, False)

        if ssl_required(params) and not support_ssl:
            scheme = params.get(DriverParams.SCHEME.value, "Unknown")
            raise CommError(
                CommError.BAD_CONFIG,
                f"Connector with scheme {scheme} requires SSL but " f"driver {driver.get_name()} doesn't support it",
            )

        handle = get_handle()
        connector = ConnectorInfo(handle, driver, params, mode, 0, 0, False, False)
        driver.register_conn_monitor(self)
        with self.lock:
            self.connectors[handle] = connector

        log.debug(f"Connector {connector} is created")

        if self.started:
            self.start_connector(connector)

        return handle

    def remove_connector(self, handle: str):
        with self.lock:
            connector = self.connectors.pop(handle, None)
            if connector:
                connector.stopping = True
                connector.driver.shutdown()
                log.debug(f"Connector {connector} is removed")
            else:
                log.error(f"Unknown connector handle: {handle}")

    def start(self):
        with self.lock:
            for handle in sorted(self.connectors.keys()):
                connector = self.connectors[handle]
                if not connector.started:
                    self.start_connector(connector)

        self.started = True

    def stop(self):

        with self.lock:
            for handle in sorted(self.connectors.keys()):
                connector = self.connectors[handle]
                connector.stopping = True
                connector.driver.shutdown()

        self.conn_mgr_executor.shutdown(True)
        self.frame_mgr_executor.shutdown(True)

    def find_endpoint(self, name: str) -> Optional[Endpoint]:

        sfm_endpoint = self.sfm_endpoints.get(name)
        if not sfm_endpoint:
            log.debug(f"Endpoint {name} doesn't exist")
            return None

        return sfm_endpoint.endpoint

    def remove_endpoint(self, name: str):

        sfm_endpoint = self.sfm_endpoints.get(name)
        if not sfm_endpoint:
            log.debug(f"Endpoint {name} doesn't exist or already removed")
            return

        for sfm_conn in sfm_endpoint.connections:
            sfm_conn.conn.close()

        self.sfm_endpoints.pop(name)
        log.debug(f"Endpoint {name} is removed")

    def get_connections(self, name: str) -> Optional[List[SfmConnection]]:

        sfm_endpoint = self.sfm_endpoints.get(name)
        if not sfm_endpoint:
            log.debug("Endpoint {name} doesn't exist")
            return None

        return sfm_endpoint.connections

    def send_message(self, endpoint: Endpoint, app_id: int, headers: Headers, payload: BytesAlike):
        """Send a message to endpoint for app

        The message is asynchronous, no response is expected.

        Args:
            endpoint: An endpoint to send the message to
            app_id: Application ID
            headers: headers, optional
            payload: message payload, optional

        Raises:
            CommError: If any error happens while sending the data
        """

        if endpoint.name == self.local_endpoint.name:
            self.send_loopback_message(endpoint, app_id, headers, payload)
            return

        sfm_endpoint = self.sfm_endpoints.get(endpoint.name)
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

        # TODO: If multiple connections, should retry a diff connection on errors
        start = time.perf_counter()

        sfm_conn.send_data(app_id, stream_id, headers, payload)

        self.send_frame_stats.record_value(
            category=sfm_conn.conn.connector.driver.get_name(), value=time.perf_counter() - start
        )

    def register_message_receiver(self, app_id: int, receiver: MessageReceiver):
        if self.receivers.get(app_id):
            raise CommError(CommError.BAD_CONFIG, f"Receiver for app {app_id} is already registered")

        self.receivers[app_id] = receiver

    def add_endpoint_monitor(self, monitor: EndpointMonitor):
        self.monitors.append(monitor)

    # Internal methods

    def start_connector(self, connector: ConnectorInfo):
        """Start connector in a new thread"""

        if connector.started:
            return

        log.info(f"Connector {connector} is starting")

        self.conn_mgr_executor.submit(self.start_connector_task, connector)

    @staticmethod
    def start_connector_task(connector: ConnectorInfo):
        """Start connector in a new thread
        This function will loop as long as connector is not stopped
        """

        connector.started = True
        if connector.mode == Mode.ACTIVE:
            starter = connector.driver.connect
        else:
            starter = connector.driver.listen

        wait = INIT_WAIT
        while not connector.stopping:
            start_time = time.time()
            try:
                starter(connector)
            except Exception as ex:
                fail_msg = (
                    f"Connector {connector} failed with exception {type(ex).__name__}: {secure_format_exception(ex)}"
                )
                if wait < SILENT_RECONNECT_TIME:
                    log.debug(fail_msg)
                else:
                    log.error(fail_msg)

            if connector.stopping:
                log.debug(f"Connector {connector} has stopped")
                break

            # After a long run, resetting wait
            run_time = time.time() - start_time
            if run_time > MAX_WAIT:
                log.debug(f"Driver for {connector} had a long run ({run_time} sec), resetting wait")
                wait = INIT_WAIT

            reconnect_msg = f"Retrying {connector} in {wait} seconds"
            # First few retries may happen in normal shutdown, show it as debug
            if wait < SILENT_RECONNECT_TIME:
                log.debug(reconnect_msg)
            else:
                log.info(reconnect_msg)

            time.sleep(wait)
            # Exponential backoff
            wait *= 2
            if wait > MAX_WAIT:
                wait = MAX_WAIT

    def state_change(self, connection: Connection):
        try:
            state = connection.state
            connector = connection.connector
            if state == ConnState.CONNECTED:
                self.handle_new_connection(connection)
                with self.lock:
                    connector.total_conns += 1
                    connector.curr_conns += 1
            elif state == ConnState.CLOSED:
                self.close_connection(connection)
                with self.lock:
                    connector.curr_conns -= 1
            else:
                log.error(f"Unknown state: {state}")
        except Exception as ex:
            log.error(f"Error handling state change: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())

    def process_frame_task(self, sfm_conn: SfmConnection, frame: BytesAlike):

        try:
            prefix = Prefix.from_bytes(frame)
            log.debug(f"Received frame: {prefix} on {sfm_conn.conn}")

            if prefix.header_len == 0:
                headers = None
            else:
                headers = msgpack.unpackb(frame[PREFIX_LEN : PREFIX_LEN + prefix.header_len])

            if prefix.type in (Types.HELLO, Types.READY):
                if prefix.type == Types.HELLO:
                    sfm_conn.send_handshake(Types.READY)

                data = self.get_dict_payload(prefix, frame)
                self.update_endpoint(sfm_conn, data)

            elif prefix.type == Types.DATA:
                if prefix.length > PREFIX_LEN + prefix.header_len:
                    payload = frame[PREFIX_LEN + prefix.header_len :]
                else:
                    payload = None

                message = Message(headers, payload)
                receiver = self.receivers.get(prefix.app_id)
                if receiver:
                    receiver.process_message(sfm_conn.sfm_endpoint.endpoint, sfm_conn.conn, prefix.app_id, message)
                else:
                    log.debug(f"No receiver registered for App ID {prefix.app_id}, message ignored")

            else:
                log.error(f"Received unsupported frame type {prefix.type} on {sfm_conn.get_name()}")
        except Exception as ex:
            log.error(f"Error processing frame: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())

    def process_frame(self, sfm_conn: SfmConnection, frame: BytesAlike):
        self.frame_mgr_executor.submit(self.process_frame_task, sfm_conn, frame)

    def update_endpoint(self, sfm_conn: SfmConnection, data: dict):

        endpoint_name = data.pop(HandshakeKeys.ENDPOINT_NAME)
        if not endpoint_name:
            raise CommError(CommError.BAD_DATA, f"Handshake without endpoint name for connection {sfm_conn.get_name()}")

        if endpoint_name == self.local_endpoint.name:
            raise CommError(
                CommError.BAD_DATA, f"Duplicate endpoint name {endpoint_name} for connection {sfm_conn.get_name()}"
            )

        endpoint = Endpoint(endpoint_name, data)
        endpoint.state = EndpointState.READY
        conn_props = sfm_conn.conn.get_conn_properties()
        if conn_props:
            endpoint.conn_props.update(conn_props)

        sfm_endpoint = self.sfm_endpoints.get(endpoint_name)
        if sfm_endpoint:
            old_state = sfm_endpoint.endpoint.state
            sfm_endpoint.endpoint = endpoint
        else:
            old_state = EndpointState.IDLE
            sfm_endpoint = SfmEndpoint(endpoint)

        sfm_endpoint.add_connection(sfm_conn)
        sfm_conn.sfm_endpoint = sfm_endpoint
        self.sfm_endpoints[endpoint_name] = sfm_endpoint

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
        return msgpack.unpackb(mv[(PREFIX_LEN + prefix.header_len) :])

    def handle_new_connection(self, connection: Connection):

        sfm_conn = SfmConnection(connection, self.local_endpoint)
        with self.lock:
            self.sfm_conns[sfm_conn.get_name()] = sfm_conn

        connection.register_frame_receiver(SfmFrameReceiver(self, sfm_conn))

        if connection.connector.mode == Mode.ACTIVE:
            sfm_conn.send_handshake(Types.HELLO)

    def close_connection(self, connection: Connection):

        with self.lock:
            name = connection.name
            if name not in self.sfm_conns:
                log.debug(f"Connection {name} has closed with no endpoint assigned")
                return

            sfm_conn = self.sfm_conns.pop(name)
            sfm_endpoint = sfm_conn.sfm_endpoint
            if sfm_endpoint is None:
                log.debug(f"Connection {name} is closed before SFM handshake")
                return

            old_state = sfm_endpoint.endpoint.state
            sfm_endpoint.remove_connection(sfm_conn)

            state = EndpointState.READY if sfm_endpoint.connections else EndpointState.DISCONNECTED
            sfm_endpoint.endpoint.state = state
            if old_state != state:
                self.notify_monitors(sfm_endpoint.endpoint)

    def send_loopback_message(self, endpoint: Endpoint, app_id: int, headers: Headers, payload: BytesAlike):
        """Send message to itself"""

        # Call receiver in a different thread to avoid deadlock
        self.frame_mgr_executor.submit(self.loopback_message_task, endpoint, app_id, headers, payload)

    def loopback_message_task(self, endpoint: Endpoint, app_id: int, headers: Headers, payload: BytesAlike):

        receiver = self.receivers.get(app_id)
        if not receiver:
            log.debug(f"No receiver registered for App ID {app_id}, loopback message ignored")
            return

        try:
            receiver.process_message(endpoint, self.null_conn, app_id, Message(headers, payload))
        except Exception as ex:
            log.error(f"Loopback message error: {secure_format_exception(ex)}")


class SfmFrameReceiver(FrameReceiver):
    def __init__(self, conn_manager: ConnManager, conn: SfmConnection):
        self.conn_manager = conn_manager
        self.conn = conn

    def process_frame(self, frame: BytesAlike):
        try:
            self.conn_manager.process_frame(self.conn, frame)
        except Exception as ex:
            log.error(f"Error processing frame: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())


class NullConnection(Connection):
    """A mock connection used for loopback messages"""

    def __init__(self):
        connector = ConnectorInfo("Null", None, {}, Mode.ACTIVE, 0, 0, False, False)
        super().__init__(connector)

    def get_conn_properties(self) -> dict:
        return {DriverParams.LOCAL_ADDR.value: SELF_ADDR, DriverParams.PEER_ADDR.value: SELF_ADDR}

    def close(self):
        pass

    def send_frame(self, frame: BytesAlike):
        raise CommError(CommError.NOT_SUPPORTED, "Can't send data on Null connection")
