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
import ssl
from concurrent.futures import ThreadPoolExecutor
from ssl import SSLContext
from typing import List, Any, Union

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams
from nvflare.fuel.f3.drivers.prefix import Prefix
from nvflare.fuel.f3.sfm.conn_manager import Mode

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8


class WsConnection(Connection):

    def __init__(self, websocket: Any, loop, mode: Mode):
        super().__init__()
        self.websocket = websocket
        self.mode = mode
        self.queue = asyncio.Queue(QUEUE_SIZE)
        self.loop = loop
        self.closing = False

    def get_conn_properties(self) -> dict:
        host, port = self.websocket.remote_address
        return {"peer_host": host, "peer_port": port}

    def close(self):
        self.closing = True

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        # Can't do asyncio send directly. Append to the queue
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(frame), self.loop).result()
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
        pass

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
    def get_ssl_context(params: dict, server: bool) -> SSLContext:
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
