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
from typing import List, Dict, Any

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.connnector import Mode
from nvflare.fuel.f3.drivers.driver import Connector
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.aio_connection import AioConnection
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver

log = logging.getLogger(__name__)


class AioTcpDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        self.aio_ctx = AioContext.get_global_context()
        self.server = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["atcp", "satcp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: False,
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
        if self.server:
            self.server.close()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return TcpDriver.get_urls(scheme, resources)

    # Internal methods

    def start_event_loop(self, mode: Mode):
        if mode != self.connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector}")

        self.aio_ctx.run_coro(self.event_loop(mode)).result()

    async def event_loop(self, mode: Mode):

        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        if mode == Mode.ACTIVE:
            coroutine = self.async_connect(host, port)
        else:
            coroutine = self.async_listen(host, port)

        await coroutine

    async def async_connect(self, host, port):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            await self.create_connection(reader, writer)
        except BaseException as ex:
            log.error(f"Connecting failed for {self.connector}: {ex}")

    async def async_listen(self, host, port):
        try:
            self.server = await asyncio.start_server(self.create_connection, host, port)
            async with self.server:
                await self.server.serve_forever()
        except BaseException as ex:
            log.error(f"Listening failed for {self.connector}: {ex}")

    async def create_connection(self, reader, writer):
        conn = AioConnection(self.connector, self.aio_ctx, reader, writer)
        self.add_connection(conn)
        await conn.read_loop()
        self.close_connection(conn)




