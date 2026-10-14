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
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from typing import Dict, List, Optional

import msgpack
from cryptography import x509

from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver, get_param, is_admin_listener, is_mtls_connection
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection, ConnState, FrameReceiver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver import ConnMonitor, Driver
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import ssl_required
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.message import Message, MessageReceiver
from nvflare.fuel.f3.sfm.constants import FLARE_PROTOCOL_VERSION, HandshakeKeys, Types
from nvflare.fuel.f3.sfm.heartbeat_monitor import HeartbeatMonitor
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection
from nvflare.fuel.f3.sfm.sfm_endpoint import SfmEndpoint
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.utils.admin_name_utils import is_valid_admin_client_name
from nvflare.fuel.utils.buffer_list import BufferList
from nvflare.security.logging import secure_format_exception, secure_format_traceback

FRAME_THREAD_POOL_SIZE = 100
CONN_THREAD_POOL_SIZE = 16
INIT_WAIT = 1
MAX_WAIT = 10
SILENT_RECONNECT_TIME = 5
SELF_ADDR = "0.0.0.0:0"

log = logging.getLogger(__name__)

handle_lock = threading.Lock()
handle_count = 0


def _peer_cert(conn_props: dict):
    der = get_param(conn_props, DriverParams.PEER_CERT)
    return x509.load_der_x509_certificate(der) if der else None


def get_handle():
    global handle_lock, handle_count
    with handle_lock:
        handle_count += 1

    return "CH%05d" % handle_count


class ConnManager(ConnMonitor):
    """SFM connection manager
    The class is responsible for maintaining state of SFM connections and pumping data through them
    """

    def __init__(self, local_endpoint: Endpoint, identity_resolver: CellIdentityResolver = None):
        self.local_endpoint = local_endpoint
        self.identity_resolver = identity_resolver if identity_resolver else CellIdentityResolver(local_endpoint.name)

        # Active connectors
        self.connectors: Dict[str, ConnectorInfo] = {}
        # A connector remains owned until its driver has finished shutdown,
        # including when removal races a manager-wide stop.
        self._connector_cleanups: Dict[str, Future] = {}

        # A dict of SFM connections, key is connection name
        self.sfm_conns: Dict[str, SfmConnection] = {}

        # A dict of SfmEndpoint for finding endpoint by name
        self.sfm_endpoints: Dict[str, SfmEndpoint] = {}

        # A list of Endpoint monitors
        self.monitors: List[EndpointMonitor] = []

        # App/receiver mapping
        self.receivers: Dict[int, MessageReceiver] = {}

        self.started = False
        self.stopped = False
        self.conn_mgr_executor = ThreadPoolExecutor(CONN_THREAD_POOL_SIZE, "conn_mgr")
        self.frame_mgr_executor = ThreadPoolExecutor(FRAME_THREAD_POOL_SIZE, "frame_mgr")
        self.lock = threading.Lock()
        self._endpoint_notifications = deque()
        self._notification_condition = threading.Condition(self.lock)
        self._notification_thread = None
        self._notification_stopping = False
        self.null_conn = NullConnection()
        stats = StatsPoolManager.get_pool("sfm_send_frame")
        if not stats:
            stats = StatsPoolManager.add_time_hist_pool(
                "sfm_send_frame", "SFM send_frame time in secs", scope=local_endpoint.name
            )
        self.send_frame_stats = stats
        self.heartbeat_monitor = HeartbeatMonitor(self.sfm_conns)

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
        connector = ConnectorInfo(handle, driver, params, mode, 0, 0, False, threading.Event())
        driver.register_conn_monitor(self)
        with self.lock:
            if self.stopped:
                raise CommError(CommError.CLOSED, "Connection manager is stopped")
            self.connectors[handle] = connector

        log.debug(f"Connector {connector} is created")

        if self.started:
            self.start_connector(connector)

        return handle

    def remove_connector(self, handle: str):
        with self.lock:
            completion = self._connector_cleanups.get(handle)
            connector = None if completion else self.connectors.pop(handle, None)
            if connector:
                completion = Future()
                self._connector_cleanups[handle] = completion
        if connector:
            connector.stopped.set()
            try:
                driver_completion = connector.driver.shutdown()
                if isinstance(driver_completion, Future):
                    driver_completion.result()
            except BaseException as ex:
                completion.set_exception(ex)
                raise
            else:
                completion.set_result(None)
            finally:
                with self.lock:
                    self._connector_cleanups.pop(handle, None)
            log.debug(f"Connector {connector} is removed")
        elif completion:
            completion.result()
            with self.lock:
                self.connectors.pop(handle, None)
        else:
            log.error(f"Unknown connector handle: {handle}")

    def start(self):
        with self.lock:
            for handle in sorted(self.connectors.keys()):
                connector = self.connectors[handle]
                if not connector.started:
                    self.start_connector(connector)

        self.heartbeat_monitor.start()

        self.started = True

    def stop(self):

        self.heartbeat_monitor.stop()

        with self.lock:
            self.stopped = True
            connectors = []
            for handle, connector in self.connectors.items():
                if handle not in self._connector_cleanups:
                    completion = Future()
                    self._connector_cleanups[handle] = completion
                    connectors.append((connector, completion))
            completions = list(self._connector_cleanups.values())
        driver_completions = []
        first_error = None
        for connector, completion in connectors:
            connector.stopped.set()
            try:
                driver_completion = connector.driver.shutdown()
            except Exception as ex:
                completion.set_exception(ex)
                continue
            if isinstance(driver_completion, Future):
                driver_completions.append((driver_completion, completion))
            else:
                completion.set_result(None)

        # Request cancellation from every driver before waiting for any one.
        for driver_completion, completion in driver_completions:
            try:
                driver_completion.result()
            except Exception as ex:
                completion.set_exception(ex)
            else:
                completion.set_result(None)

        for completion in completions:
            try:
                completion.result()
            except Exception as ex:
                if first_error is None:
                    first_error = ex

        for executor in (self.conn_mgr_executor, self.frame_mgr_executor):
            try:
                executor.shutdown(True)
            except Exception as ex:
                if first_error is None:
                    first_error = ex

        with self._notification_condition:
            self._notification_stopping = True
            self._notification_condition.notify_all()
            notifier = self._notification_thread
        if notifier and notifier is not threading.current_thread():
            notifier.join()

        if first_error is not None:
            raise first_error

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

    def send_message(self, endpoint: Endpoint, app_id: int, headers: Optional[dict], payload: BytesAlike):
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

        # Flatten buffer list so drivers don't have to deal with it
        if isinstance(payload, list):
            flat_payload = BufferList(payload).flatten()
        else:
            flat_payload = payload

        if endpoint.name == self.local_endpoint.name:
            self.send_loopback_message(endpoint, app_id, headers, flat_payload)
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

        sfm_conn.send_data(app_id, stream_id, headers, flat_payload)

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

        try:
            self.conn_mgr_executor.submit(self.start_connector_task, connector)
        except RuntimeError:
            log.debug("Connector start skipped — executor already shut down")

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
        quiet_reconnect = connector.params.get(DriverParams.QUIET_RECONNECT.value, False)
        while not connector.stopped.is_set():
            start_time = time.time()
            try:
                starter(connector)
            except Exception as ex:
                fail_msg = (
                    f"Connector {connector} failed with exception {type(ex).__name__}: {secure_format_exception(ex)}"
                )
                if quiet_reconnect or wait < SILENT_RECONNECT_TIME:
                    log.debug(fail_msg)
                else:
                    log.error(fail_msg)

            if connector.stopped.is_set():
                log.debug(f"Connector {connector} has stopped")
                break

            # After a long run, resetting wait
            run_time = time.time() - start_time
            if run_time > MAX_WAIT:
                log.debug(f"Driver for {connector} had a long run ({run_time} sec), resetting wait")
                wait = INIT_WAIT

            reconnect_msg = f"Retrying {connector} in {wait} seconds"
            # First few retries may happen in normal shutdown, show it as debug
            if quiet_reconnect or wait < SILENT_RECONNECT_TIME:
                log.debug(reconnect_msg)
            else:
                log.info(reconnect_msg)

            connector.stopped.wait(wait)

            # Exponential backoff
            wait *= 2
            if wait > MAX_WAIT:
                wait = MAX_WAIT

    def state_change(self, connection: Connection):
        try:
            state = connection.state
            connector = connection.connector
            if state == ConnState.CONNECTED:
                log.info(f"Connection {connection} is created: PID: {os.getpid()}")
                self.handle_new_connection(connection)
                with self.lock:
                    connector.total_conns += 1
                    connector.curr_conns += 1
            elif state == ConnState.CLOSED:
                log.info(f"Connection {connection} is closed PID: {os.getpid()}")
                self.close_connection(connection)
                with self.lock:
                    connector.curr_conns -= 1
            else:
                log.error(f"Unknown state: {state}")
        except Exception as ex:
            log.error(f"Error handling state change: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())

    def process_frame_task(self, sfm_conn: SfmConnection, frame: BytesAlike):

        if self.stopped:
            return

        try:
            prefix = Prefix.from_bytes(frame)
            log.debug(f"Received frame: {prefix} on {sfm_conn.conn}")

            if prefix.header_len == 0:
                headers = None
            else:
                headers = msgpack.unpackb(frame[PREFIX_LEN : PREFIX_LEN + prefix.header_len])

            if prefix.type in (Types.HELLO, Types.READY):
                data = self.get_dict_payload(prefix, frame)
                version = data.get(HandshakeKeys.FLARE_PROTOCOL)
                if type(version) is not int or version != FLARE_PROTOCOL_VERSION:
                    name = data.get(HandshakeKeys.ENDPOINT_NAME)
                    log.warning(f"Rejecting endpoint {name}: incompatible FLARE protocol {version!r}")
                    sfm_conn.conn.close()
                    # Report the failed attempt without touching an existing endpoint of the same name.
                    endpoint = Endpoint(name, data)
                    endpoint.state = EndpointState.ERROR
                    with self.lock:
                        self._queue_endpoint_notification(endpoint)
                    return

                if prefix.type == Types.HELLO:
                    sfm_conn.send_handshake(Types.READY)

                self.update_endpoint(sfm_conn, data)
            elif prefix.type == Types.PING:
                sfm_conn.send_heartbeat(Types.PONG)
            elif prefix.type == Types.PONG:
                log.debug(f"PONG received for {sfm_conn.conn}")
                # No action is needed for PONG. The last_activity is already updated
            elif prefix.type == Types.DATA:
                if sfm_conn.sfm_endpoint is None:
                    # READY may arrive before the concurrent handshake task finishes attaching the endpoint.
                    log.debug("Ignoring DATA before endpoint attachment")
                    return
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
        except RuntimeError as ex:
            if self.stopped:
                log.debug(f"Frame processing interrupted by shutdown: {secure_format_exception(ex)}")
            else:
                log.error(f"Error processing frame: {secure_format_exception(ex)}")
                log.debug(secure_format_traceback())
        except Exception as ex:
            log.error(f"Error processing frame: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())

    def process_frame(self, sfm_conn: SfmConnection, frame: BytesAlike):
        if self.stopped:
            log.debug(f"Frame received after shutdown for connection {sfm_conn.get_name()}")
            return

        try:
            self.frame_mgr_executor.submit(self.process_frame_task, sfm_conn, frame)
        except RuntimeError:
            log.debug(f"Frame received after shutdown for connection {sfm_conn.get_name()}")

    def update_endpoint(self, sfm_conn: SfmConnection, data: dict):

        endpoint_name = data.pop(HandshakeKeys.ENDPOINT_NAME)
        if not endpoint_name:
            raise CommError(CommError.BAD_DATA, f"Handshake without endpoint name for connection {sfm_conn.get_name()}")

        endpoint_name = FQCN.normalize(endpoint_name)
        err = FQCN.validate(endpoint_name)
        if err:
            sfm_conn.conn.close()
            raise CommError(
                CommError.BAD_DATA,
                f"Invalid endpoint name '{endpoint_name}' for connection {sfm_conn.get_name()}: {err}",
            )

        if endpoint_name == self.local_endpoint.name:
            raise CommError(
                CommError.BAD_DATA, f"Duplicate endpoint name {endpoint_name} for connection {sfm_conn.get_name()}"
            )

        conn_props = sfm_conn.conn.get_conn_properties()
        # Passive mTLS connections must always present the connecting peer's CN.
        # Active mTLS connections enforce the same binding when the driver exposes
        # a real peer CN; gRPC active-side connections may report "N/A"/None.
        if is_mtls_connection(sfm_conn.conn.connector.params):
            peer_cn = get_param(conn_props, DriverParams.PEER_CN)
            if sfm_conn.conn.connector.mode == Mode.PASSIVE or (peer_cn and peer_cn != "N/A"):
                if (
                    is_valid_admin_client_name(endpoint_name)
                    and sfm_conn.conn.connector.mode == Mode.PASSIVE
                    and not is_admin_listener(sfm_conn.conn.connector.params)
                ):
                    sfm_conn.conn.close()
                    raise CommError(
                        CommError.BAD_DATA,
                        f"Admin endpoint '{endpoint_name}' can only connect through an admin listener",
                    )
                try:
                    self.identity_resolver.require_match(
                        endpoint_name,
                        peer_cn,
                        f"connection {sfm_conn.get_name()}",
                        peer_cert=_peer_cert(conn_props),
                    )
                except ValueError as ex:
                    sfm_conn.conn.close()
                    raise CommError(CommError.BAD_DATA, str(ex))

        endpoint = Endpoint(endpoint_name, data)
        endpoint.state = EndpointState.READY
        if conn_props:
            endpoint.conn_props.update(conn_props)

        # Closing the connection removes it from sfm_conns under this same lock.
        # A frame worker that passed its initial stopped check must not attach
        # that connection or publish READY after the close notification.
        with self.lock:
            if (
                self.stopped
                or sfm_conn.closed
                or getattr(sfm_conn.conn, "state", None) == ConnState.CLOSED
                or (sfm_conn.registered and self.sfm_conns.get(sfm_conn.get_name()) is not sfm_conn)
            ):
                return

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
                self._queue_endpoint_notification(endpoint)

    def _queue_endpoint_notification(self, endpoint: Endpoint):
        """Queue a state snapshot while holding self.lock, in mutation order."""
        if not self.monitors or self._notification_stopping:
            return
        self._endpoint_notifications.append(copy(endpoint))
        if self._notification_thread is None:
            self._notification_thread = threading.Thread(
                target=self._deliver_endpoint_notifications, name="endpoint_notifications", daemon=True
            )
            self._notification_thread.start()
        self._notification_condition.notify()

    def _deliver_endpoint_notifications(self):
        # The queue preserves mutation order. No manager or driver lock is held
        # when invoking callbacks, which may remove connectors or stop the cell.
        while True:
            with self._notification_condition:
                while not self._endpoint_notifications and not self._notification_stopping:
                    self._notification_condition.wait()
                if not self._endpoint_notifications:
                    return
                endpoint = self._endpoint_notifications.popleft()
            try:
                self.notify_monitors(endpoint)
            except Exception as ex:
                log.error(f"Endpoint monitor failed: {secure_format_exception(ex)}")
                log.debug(secure_format_traceback())

    def notify_monitors(self, endpoint: Endpoint):

        if not self.monitors:
            log.debug("No endpoint monitor registered")
            return

        for monitor in self.monitors:
            monitor.state_change(endpoint)

    @staticmethod
    def get_dict_payload(prefix, frame):
        mv = frame if isinstance(frame, memoryview) else memoryview(frame)
        return msgpack.unpackb(mv[(PREFIX_LEN + prefix.header_len) :])

    def handle_new_connection(self, connection: Connection):

        sfm_conn = SfmConnection(connection, self.local_endpoint)
        with self.lock:
            sfm_conn.registered = True
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
            sfm_conn.closed = True
            sfm_endpoint = sfm_conn.sfm_endpoint
            if sfm_endpoint is None:
                log.debug(f"Connection {name} is closed before SFM handshake")
                return

            old_state = sfm_endpoint.endpoint.state
            sfm_endpoint.remove_connection(sfm_conn)

            state = EndpointState.READY if sfm_endpoint.connections else EndpointState.DISCONNECTED
            sfm_endpoint.endpoint.state = state
            if old_state != state:
                self._queue_endpoint_notification(sfm_endpoint.endpoint)

    def send_loopback_message(self, endpoint: Endpoint, app_id: int, headers: Optional[dict], payload: BytesAlike):
        """Send message to itself"""

        if self.stopped:
            return

        # Call receiver in a different thread to avoid deadlock
        try:
            self.frame_mgr_executor.submit(self.loopback_message_task, endpoint, app_id, headers, payload)
        except RuntimeError as e:
            log.debug(f"Loopback submit skipped: {e}")

    def loopback_message_task(self, endpoint: Endpoint, app_id: int, headers: Optional[dict], payload: BytesAlike):

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
        self.conn.last_activity = time.time()

        try:
            self.conn_manager.process_frame(self.conn, frame)
        except Exception as ex:
            log.error(f"Error processing frame: {secure_format_exception(ex)}")
            log.debug(secure_format_traceback())


class NullConnection(Connection):
    """A mock connection used for loopback messages"""

    def __init__(self):
        connector = ConnectorInfo("Null", None, {}, Mode.ACTIVE, 0, 0, False, threading.Event())
        super().__init__(connector)

    def get_conn_properties(self) -> dict:
        return {DriverParams.LOCAL_ADDR.value: SELF_ADDR, DriverParams.PEER_ADDR.value: SELF_ADDR}

    def close(self):
        pass

    def send_frame(self, frame: BytesAlike):
        raise CommError(CommError.NOT_SUPPORTED, "Can't send data on Null connection")
