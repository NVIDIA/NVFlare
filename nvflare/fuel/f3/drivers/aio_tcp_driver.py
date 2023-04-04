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
import asyncio
import logging
from concurrent.futures import CancelledError
from typing import Any, Dict, List

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.aio_conn import AioConnection
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver

log = logging.getLogger(__name__)


class AioTcpDriver(BaseDriver):
    def __init__(self):
        super().__init__()
        self.aio_ctx = AioContext.get_global_context()
        self.server = None
        self.ssl_context = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["atcp", "satcp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.HEARTBEAT.value: False, DriverCap.SUPPORT_SSL.value: True}

    def listen(self, connector: ConnectorInfo):
        self._run(connector, Mode.PASSIVE)

    def connect(self, connector: ConnectorInfo):
        self._run(connector, Mode.ACTIVE)

    def shutdown(self):
        self.close_all()

        if self.server:
            self.server.close()
            # This will wake up the event loop to end the server
            self.aio_ctx.run_coro(asyncio.sleep(0))

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return TcpDriver.get_urls(scheme, resources)

    # Internal methods

    def _run(self, connector: ConnectorInfo, mode: Mode):
        self.connector = connector
        if mode != self.connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector}")

        try:
            self.aio_ctx.run_coro(self._async_run(mode)).result()
        except CancelledError:
            log.debug(f"Connector {self.connector} is cancelled")

    async def _async_run(self, mode: Mode):

        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        if mode == Mode.ACTIVE:
            coroutine = self._tcp_connect(host, port)
        else:
            coroutine = self._tcp_listen(host, port)

        await coroutine

    async def _tcp_connect(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=False)
        reader, writer = await asyncio.open_connection(host, port, ssl=self.ssl_context)
        await self._create_connection(reader, writer)

    async def _tcp_listen(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=True)
        self.server = await asyncio.start_server(self._create_connection, host, port, ssl=self.ssl_context)
        async with self.server:
            await self.server.serve_forever()

    async def _create_connection(self, reader, writer):
        conn = AioConnection(self.connector, self.aio_ctx, reader, writer, self.ssl_context is not None)
        self.add_connection(conn)
        await conn.read_loop()
        self.close_connection(conn)
