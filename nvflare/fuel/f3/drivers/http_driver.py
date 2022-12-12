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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Union, Optional
from urllib.parse import urlparse

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, CommonKeys
from nvflare.fuel.f3.drivers.prefix import Prefix
from nvflare.fuel.f3.sfm.conn_manager import Mode

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8


class WsConnection(Connection):

    def __init__(self, websocket: Any, mode: Mode):
        super().__init__()
        self.websocket = websocket
        self.mode = mode
        self.queue = asyncio.Queue(QUEUE_SIZE)
        self.loop = asyncio.get_event_loop()
        self.closing = False

    def get_conn_properties(self) -> dict:
        pass

    def close(self):
        self.closing = True

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        # Can't do asyncio send directly. Append to the queue
        asyncio.run_coroutine_threadsafe(self.queue.put(frame), self.loop).result()


class HttpDriver(Driver):

    def __init__(self, conn_props: dict, resource_policy: Optional[dict] = None):
        super().__init__(conn_props, resource_policy)
        self.connections = {}
        self.scheme, self.host, self.port = self.find_host_and_port()
        self.executor = ThreadPoolExecutor(THREAD_POOL_SIZE, "http_driver")
        self.loop = asyncio.get_event_loop()

    def supported_transports(self) -> List[str]:
        return ["http", "https", "ws", "wss"]

    def listen(self, properties: Optional[dict] = None):
        self.start_event_loop(Mode.PASSIVE)

    def connect(self, properties: Optional[dict] = None):
        self.start_event_loop(Mode.ACTIVE)

    def shutdown(self):
        pass

    def start_event_loop(self, mode: Mode):
        self.executor.submit(self.event_loop, mode)
        # self.event_loop(mode)

    def event_loop(self, mode: Mode):
        if mode == Mode.ACTIVE:
            coroutine = self.async_connect()
        else:
            coroutine = self.async_listen()

        self.loop.run_until_complete(coroutine)
        self.loop.run_forever()

    async def async_connect(self):
        async with websockets.connect(f"ws://{self.host}:{self.port}") as ws:
            conn = WsConnection(ws, Mode.ACTIVE)
            self.add_connection(conn)
            await self.read_write_loop(conn)

    async def async_listen(self):
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    async def handler(self, websocket):
        conn = WsConnection(websocket, Mode.PASSIVE)
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
                # Do sync call to guarantee the order
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

    def find_host_and_port(self) -> (str, int):

        url = self.conn_props.get(CommonKeys.URL)

        if url:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in self.supported_transports():
                raise CommError(CommError.NOT_SUPPORTED, f"scheme is not supported by this driver: {parsed_url.scheme}")

            scheme = parsed_url.scheme
            parts = parsed_url.netloc.split(":")
            host = None
            port = 0
            if len(parts) >= 1:
                host = parts[0]
            if len(parts) >= 2:
                port = int(parts[1])
        else:
            scheme = "http"
            host = self.conn_props.get(CommonKeys.HOST)
            port_str = self.conn_props.get(CommonKeys.PORT)
            if port_str:
                port = int(port_str)
            else:
                port = 0

        if port == 0:
            # should check availability of the port
            port = random.randint(10000, 63000)

        return scheme, host, port
