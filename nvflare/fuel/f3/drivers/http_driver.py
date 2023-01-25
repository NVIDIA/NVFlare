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
from typing import List, Any, Union

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams, Connector
from nvflare.fuel.f3.drivers.prefix import Prefix
from nvflare.fuel.f3.sfm.conn_manager import Mode

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8
MAX_MSG_SIZE = 2000000000   # 1GB


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

        self.loop.stop()
        pending_tasks = asyncio.all_tasks(self.loop)
        for task in pending_tasks:
            task.cancel()
        asyncio.sleep(0)
        self.loop.close()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "https"

        host = resources.get("host") if resources else None
        if not host:
            host = "localhost"

        port = net_utils.get_open_tcp_port(resources)
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

        ssl_context = net_utils.get_ssl_context(params, False)
        if ssl_context:
            scheme = "wss"
        else:
            scheme = "ws"
        async with websockets.connect(f"{scheme}://{host}:{port}", ssl=ssl_context, max_size=MAX_MSG_SIZE) as ws:
            conn = WsConnection(ws, self.loop, self.connector)
            self.add_connection(conn)
            await self.read_write_loop(conn)

    async def async_listen(self):
        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)
        ssl_context = net_utils.get_ssl_context(params, True)
        async with websockets.serve(self.handler, host, port, ssl=ssl_context, max_size=MAX_MSG_SIZE):
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
