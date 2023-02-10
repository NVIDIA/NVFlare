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
from nvflare.fuel.f3.drivers.driver_params import DriverCap
from nvflare.fuel.f3.drivers.aio_conn import AioConnection
from nvflare.fuel.f3.drivers.uds_driver import UdsDriver

log = logging.getLogger(__name__)


class AioUdsDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        self.aio_ctx = AioContext.get_global_context()
        self.server = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["auds"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: False,
            DriverCap.SUPPORT_SSL.value: False
        }

    def listen(self, connector: Connector):
        self._run(connector, Mode.PASSIVE)

    def connect(self, connector: Connector):
        self._run(connector, Mode.ACTIVE)

    def shutdown(self):
        self.close_all()

        if self.server:
            self.server.close()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return UdsDriver.get_urls(scheme, resources)

    # Internal methods

    def _run(self, connector: Connector, mode: Mode):
        self.connector = connector
        if mode != self.connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector}")

        self.aio_ctx.run_coro(self._async_run(mode)).result()

    async def _async_run(self, mode: Mode):

        socket_path = UdsDriver.get_socket_path(self.connector.params)

        if mode == Mode.ACTIVE:
            coroutine = self._uds_connect(socket_path)
        else:
            coroutine = self._uds_listen(socket_path)

        await coroutine

    async def _uds_connect(self, socket_path: str):
        reader, writer = await asyncio.open_unix_connection(socket_path)
        await self._create_connection(reader, writer)

    async def _uds_listen(self, socket_path: str):
        self.server = await asyncio.start_unix_server(self._create_connection, socket_path)
        async with self.server:
            await self.server.serve_forever()

    async def _create_connection(self, reader, writer):
        conn = AioConnection(self.connector, self.aio_ctx, reader, writer)
        self.add_connection(conn)
        await conn.read_loop()
        self.close_connection(conn)
