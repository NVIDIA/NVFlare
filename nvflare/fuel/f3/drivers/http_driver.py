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
from typing import List, Any, Dict

import websockets

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.connection import Connection, BytesAlike
from nvflare.fuel.f3.drivers.driver import Connector
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import get_tcp_urls
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

    def send_frame(self, frame: BytesAlike):
        # Can't do asyncio send directly. Append to the queue
        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(frame), self.loop)
        except BaseException as ex:
            log.error(f"Error sending frame: {ex}")


class HttpDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        self.loop = None
        self.stop_event = None
        self.connector = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["http", "https", "ws", "wss"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: True,
            DriverCap.SUPPORT_SSL.value: True
        }

    def listen(self, connector: Connector):
        self.connector = connector
        self.start_event_loop(Mode.PASSIVE)

    def connect(self, connector: Connector):
        self.connector = connector
        self.start_event_loop(Mode.ACTIVE)

    def shutdown(self):
        self.close_all()

        if not self.loop:
            return

        self.stop_event.set_result(None)

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

        return get_tcp_urls(scheme, resources)

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
            self.close_connection(conn)

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
        self.close_connection(conn)

    @staticmethod
    async def reader(conn: WsConnection):
        while not conn.closing:
            # Reading from websocket and call receiver CB
            frame = await conn.websocket.recv()
            conn.process_frame(frame)

    @staticmethod
    async def writer(conn: WsConnection):
        while not conn.closing:
            # Read from queue and send to websocket
            frame = await conn.queue.get()

            await conn.websocket.send(frame)
            # This is to yield control. See bug: https://github.com/aaugustin/websockets/issues/865
            await asyncio.sleep(0)

    async def read_write_loop(self, conn: WsConnection):
        """Pumping data on the connection"""
        await asyncio.gather(self.reader(conn), self.writer(conn))

        log.debug(f"Connection {self.get_name()}:{conn.name} is closed")
