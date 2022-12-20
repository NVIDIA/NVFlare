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
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from ssl import SSLContext
from typing import List, Any, Union, Optional

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams
from nvflare.fuel.f3.drivers.prefix import Prefix
from nvflare.fuel.f3.sfm.conn_manager import Mode

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8
LO_PORT = 1025
HI_PORT = 65535


class WsConnection(Connection):

    def __init__(self, websocket: Any, loop, mode: Mode):
        super().__init__()
        self.websocket = websocket
        self.mode = mode
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
        self.executor = ThreadPoolExecutor(THREAD_POOL_SIZE, "http_driver")
        self.loop = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["http", "https", "ws", "wss"]

    def listen(self, params: dict):
        self.start_event_loop(params, Mode.PASSIVE)

    def connect(self, params: dict):
        self.start_event_loop(params, Mode.ACTIVE)

    def shutdown(self):
        for _, v in self.connections.items():
            v.close()
        self.executor.shutdown(False)

    def get_connect_url(self, scheme: str, resources: dict):
        return self.get_url(Mode.ACTIVE, scheme, resources)

    def get_listening_url(self, scheme: str, resources: dict):
        return self.get_url(Mode.PASSIVE, scheme, resources)

    # Internal methods

    def get_url(self, mode: Mode, scheme: str, resources: dict):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "https"

        host = resources.get("host") if resources else None
        if not host:
            host = "localhost" if mode == Mode.ACTIVE else "0"

        port = self.get_open_port(resources)
        if not port:
            raise CommError(CommError.BAD_CONFIG, "Can't find an open port in the specified range")

        return f"{scheme}://{host}:{port}"

    def start_event_loop(self, params: dict, mode: Mode):
        self.executor.submit(self.event_loop, params, mode).result()

    def event_loop(self, params: dict, mode: Mode):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        if mode == Mode.ACTIVE:
            coroutine = self.async_connect(params)
        else:
            coroutine = self.async_listen(params)

        self.loop.run_until_complete(coroutine)
        self.loop.run_forever()

    async def async_connect(self, params: dict):

        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        ssl_context = self.get_ssl_context(params, False)
        if ssl_context:
            scheme = "wss"
        else:
            scheme = "ws"
        async with websockets.connect(f"{scheme}://{host}:{port}", ssl=ssl_context) as ws:
            conn = WsConnection(ws, self.loop, Mode.ACTIVE)
            self.add_connection(conn)
            await self.read_write_loop(conn)

    async def async_listen(self, params: dict):
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)
        ssl_context = self.get_ssl_context(params, True)
        async with websockets.serve(self.handler, host, port, ssl=ssl_context):
            await asyncio.Future()  # run forever

    async def handler(self, websocket):
        conn = WsConnection(websocket, self.loop, Mode.PASSIVE)
        self.add_connection(conn)
        await self.read_write_loop(conn)

    def add_connection(self, conn: WsConnection):
        self.connections[conn.name] = conn
        if not self.conn_monitor:
            log.error(f"Connection monitor not registered for driver {self.get_name()}")
        else:
            log.debug(f"Connection {self.get_name()}:{conn.name} is open")
            conn.state = ConnState.CONNECTED
            # Call the monitor in a diff thread to avoid deadlock
            self.executor.submit(self.conn_monitor.state_change, conn)

    async def reader(self, conn: WsConnection):
        while not conn.closing:
            # Reading from websocket and call receiver CB
            async for frame in conn.websocket:
                if log.isEnabledFor(logging.DEBUG):
                    prefix = Prefix.from_bytes(frame)
                    log.debug(f"Received frame: {prefix} on {self.get_name()}:{conn.name}")

                if conn.frame_receiver:
                    self.executor.submit(conn.frame_receiver.process_frame, frame)
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

    def parse_ports(self, ranges: Any) -> iter:
        all_ranges = []
        if isinstance(ranges, list):
            for r in ranges:
                all_ranges.append(self.parse_range(r))
        else:
            all_ranges.append(self.parse_range(ranges))

        return chain.from_iterable(all_ranges)

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

    def get_open_port(self, resources: dict) -> Optional[int]:

        port = resources.get(DriverParams.PORT)
        if port:
            return port

        ports = resources.get(DriverParams.PORTS)
        if not ports:
            port = random.randint(LO_PORT, HI_PORT)
            all_ports = range(port, HI_PORT+1)
        else:
            all_ports = self.parse_ports(ports)

        for port in all_ports:
            if self.check_port(port):
                return port

        return None
