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
import asyncio
import logging
import random
import socket
import ssl
from ssl import SSLContext
from typing import List, Any, Union, Optional

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams, Connector
from nvflare.fuel.f3.drivers.prefix import Prefix
from nvflare.fuel.f3.sfm.conn_manager import Mode

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8
LO_PORT = 1025
HI_PORT = 65535
MAX_ITER_SIZE = 10
RANDOM_TRIES = 20


class WsConnection(Connection):

    def __init__(self, websocket: Any, loop, connector: Connector):
        super().__init__(connector)
        self.websocket = websocket
        self.queue = asyncio.Queue(QUEUE_SIZE)
        self.loop = loop
        self.closing = False

    def get_conn_properties(self) -> dict:
        addr = self.websocket.remote_address
        if addr:
            return {"peer_host": addr[0], "peer_port": addr[1]}
        else:
            return {}

    def close(self):
        self.closing = True
        asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        # Can't do asyncio send directly. Append to the queue
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(frame), self.loop)
        except BaseException as ex:
            log.error(f"Error sending frame: {ex}")


class HttpDriver(Driver):

    def __init__(self):
        super().__init__()
        self.connections = {}
        self.loop = None
        self.stop_event = None
        self.connector = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["http", "https", "ws", "wss"]

    def listen(self, connector: Connector):
        self.connector = connector
        self.start_event_loop(Mode.PASSIVE)

    def connect(self, connector: Connector):
        self.connector = connector
        self.start_event_loop(Mode.ACTIVE)

    def shutdown(self):
        if not self.loop:
            return

        self.stop_event.set_result(None)

        for _, v in self.connections.items():
            v.close()

        self.stop_event.set_result(None)
        self.loop.stop()
        self.loop.close()

    async def async_shutdown(self):
        for _, v in self.connections.items():
            v.close()

        self.stop_event.set_result(None)
        self.loop.stop()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "https"

        host = resources.get("host") if resources else None
        if not host:
            host = "localhost"

        port = HttpDriver.get_open_port(resources)
        if not port:
            raise CommError(CommError.BAD_CONFIG, "Can't find an open port in the specified range")

        # Always listen on all interfaces
        listening_url = f"{scheme}://0:{port}"
        connect_url = f"{scheme}://{host}:{port}"

        return connect_url, listening_url

    # Internal methods

    def start_event_loop(self, mode: Mode):
        if mode != self.connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector.handle}")

        asyncio.run(self.event_loop(mode))

    async def event_loop(self, mode: Mode):
        self.loop = asyncio.get_running_loop()
        self.stop_event = self.loop.create_future()

        if mode == Mode.ACTIVE:
            coroutine = self.async_connect()
        else:
            coroutine = self.async_listen()

        await coroutine

    async def async_connect(self):

        params = self.connector.params

        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        ssl_context = self.get_ssl_context(params, False)
        if ssl_context:
            scheme = "wss"
        else:
            scheme = "ws"
        async with websockets.connect(f"{scheme}://{host}:{port}", ssl=ssl_context, max_size=1000000000) as ws:
            conn = WsConnection(ws, self.loop, self.connector)
            self.add_connection(conn)
            await self.read_write_loop(conn)

    async def async_listen(self):
        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)
        ssl_context = self.get_ssl_context(params, True)
        async with websockets.serve(self.handler, host, port, ssl=ssl_context, max_size=1000000000):
            await self.stop_event

    async def handler(self, websocket):
        conn = WsConnection(websocket, self.loop, self.connector)
        self.add_connection(conn)
        await self.read_write_loop(conn)

    def add_connection(self, conn: WsConnection):
        self.connections[conn.name] = conn
        if not self.conn_monitor:
            log.error(f"Connection monitor not registered for driver {self.get_name()}")
        else:
            log.debug(f"Connection {self.get_name()}:{conn.name} is open")
            conn.state = ConnState.CONNECTED
            self.conn_monitor.state_change(conn)

    async def reader(self, conn: WsConnection):
        while not conn.closing:
            # Reading from websocket and call receiver CB
            frame = await conn.websocket.recv()
            if log.isEnabledFor(logging.DEBUG):
                prefix = Prefix.from_bytes(frame)
                log.debug(f"Received frame: {prefix} on {self.get_name()}:{conn.name}")

            if conn.frame_receiver:
                conn.frame_receiver.process_frame(frame)
            else:
                log.error("Frame receiver not registered")

    async def writer(self, conn: WsConnection):
        while not conn.closing:
            # Read from queue and send to websocket
            frame = await conn.queue.get()
            if log.isEnabledFor(logging.DEBUG):
                prefix = Prefix.from_bytes(frame)
                log.debug(f"Sending frame: {prefix} on {self.get_name()}:{conn.name}")

            await conn.websocket.send(frame)
            # This is to yield control. See bug: https://github.com/aaugustin/websockets/issues/865
            await asyncio.sleep(0)

    async def read_write_loop(self, conn: WsConnection):
        """Pumping data on the connection"""

        await asyncio.gather(self.reader(conn), self.writer(conn))

        conn.state = ConnState.CLOSED
        if self.conn_monitor:
            self.conn_monitor.state_change(conn)

        log.debug(f"Connection {self.get_name()}:{conn.name} is closed")

    @staticmethod
    def get_ssl_context(params: dict, server: bool) -> Optional[SSLContext]:
        scheme = params.get(DriverParams.SCHEME.value)
        if scheme not in ("https", "wss"):
            return None

        ca_path = params.get(DriverParams.CA_CERT.value)
        if server:
            cert_path = params.get(DriverParams.SERVER_CERT.value)
            key_path = params.get(DriverParams.SERVER_KEY.value)
        else:
            cert_path = params.get(DriverParams.CLIENT_CERT.value)
            key_path = params.get(DriverParams.CLIENT_KEY.value)

        if not all([ca_path, cert_path, key_path]):
            raise CommError(CommError.BAD_CONFIG, f"Certificate parameters are required for scheme {scheme}")

        if server:
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        else:
            ctx = ssl.create_default_context()

        # This feature is only supported on 3.7+
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = False
        ctx.load_verify_locations(ca_path)
        ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

        return ctx

    @staticmethod
    def parse_range(entry: Any):

        if isinstance(entry, int):
            return range(entry, entry + 1)

        parts = entry.split("-")
        if len(parts) == 1:
            num = int(parts[0])
            return range(num, num + 1)
        lo = int(parts[0]) if parts[0] else LO_PORT
        hi = int(parts[1]) if parts[1] else HI_PORT
        return range(lo, hi + 1)

    @staticmethod
    def parse_ports(ranges: Any) -> list:
        all_ranges = []
        if isinstance(ranges, list):
            for r in ranges:
                all_ranges.append(HttpDriver.parse_range(r))
        else:
            all_ranges.append(HttpDriver.parse_range(ranges))

        return all_ranges

    @staticmethod
    def check_port(port) -> bool:
        result = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", port))
            result = True
        except Exception as e:
            log.debug(f"Port {port} binding error: {e}")
        s.close()

        return result

    @staticmethod
    def get_open_port(resources: dict) -> Optional[int]:

        port = resources.get(DriverParams.PORT)
        if port:
            return port

        ports = resources.get(DriverParams.PORTS)
        if not ports:
            port = random.randint(LO_PORT, HI_PORT)
            return port

        all_ports = HttpDriver.parse_ports(ports)

        for port_range in all_ports:
            if len(port_range) <= MAX_ITER_SIZE:
                for port in port_range:
                    if HttpDriver.check_port(port):
                        return port
            else:
                for i in range(RANDOM_TRIES):
                    port = random.randint(port_range.start, port_range.stop)
                    if HttpDriver.check_port(port):
                        return port

        return None
