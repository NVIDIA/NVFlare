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
from typing import Any, Dict, List

import websockets
from websockets.exceptions import ConnectionClosedOK

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.driver import ConnectorInfo
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE, get_tcp_urls
from nvflare.fuel.f3.sfm.conn_manager import Mode
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)

THREAD_POOL_SIZE = 8


class WsConnection(Connection):
    def __init__(self, websocket: Any, aio_context: AioContext, connector: ConnectorInfo, secure: bool):
        super().__init__(connector)
        self.websocket = websocket
        self.aio_context = aio_context
        self.closing = False
        self.secure = secure
        self.conn_props = self._get_socket_properties()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True
        self.aio_context.run_coro(self.websocket.close())

    def send_frame(self, frame: BytesAlike):
        self.aio_context.run_coro(self._async_send_frame(frame))

    def _get_socket_properties(self) -> dict:
        conn_props = {}

        addr = self.websocket.remote_address
        if addr:
            conn_props[DriverParams.PEER_ADDR.value] = f"{addr[0]}:{addr[1]}"

        addr = self.websocket.local_address
        if addr:
            conn_props[DriverParams.LOCAL_ADDR.value] = f"{addr[0]}:{addr[1]}"

        peer_cert = self.websocket.transport.get_extra_info("peercert")
        if peer_cert:
            cn = get_certificate_common_name(peer_cert)
        else:
            if self.secure:
                cn = "N/A"
            else:
                cn = None

        if cn:
            conn_props[DriverParams.PEER_CN.value] = cn

        return conn_props

    async def _async_send_frame(self, frame: BytesAlike):
        try:
            await self.websocket.send(frame)
            # This is to yield control. See bug: https://github.com/aaugustin/websockets/issues/865
            await asyncio.sleep(0)
        except Exception as ex:
            log.error(f"Error sending frame for connection {self}: {secure_format_exception(ex)}")


class AioHttpDriver(BaseDriver):
    """Async HTTP driver using websocket extension"""

    def __init__(self):
        super().__init__()
        self.aio_context = AioContext.get_global_context()
        self.stop_event = None
        self.ssl_context = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["http", "https", "ws", "wss"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

    def listen(self, connector: ConnectorInfo):
        self._event_loop(Mode.PASSIVE, connector)

    def connect(self, connector: ConnectorInfo):
        self._event_loop(Mode.ACTIVE, connector)

    def shutdown(self):
        self.close_all()

        if self.stop_event:
            self.stop_event.set_result(None)

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "https"

        return get_tcp_urls(scheme, resources)

    # Internal methods

    def _event_loop(self, mode: Mode, connector: ConnectorInfo):
        self.connector = connector
        if mode != connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector}")

        self.aio_context.run_coro(self._async_event_loop(mode)).result()

    async def _async_event_loop(self, mode: Mode):

        self.stop_event = self.aio_context.get_event_loop().create_future()

        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        if mode == Mode.ACTIVE:
            coroutine = self._async_connect(host, port)
        else:
            coroutine = self._async_listen(host, port)

        await coroutine

    async def _async_connect(self, host, port):

        self.ssl_context = net_utils.get_ssl_context(self.connector.params, False)
        if self.ssl_context:
            scheme = "wss"
        else:
            scheme = "ws"
        async with websockets.connect(f"{scheme}://{host}:{port}", ssl=self.ssl_context, max_size=MAX_FRAME_SIZE) as ws:
            await self._handler(ws)

    async def _async_listen(self, host, port):
        self.ssl_context = net_utils.get_ssl_context(self.connector.params, True)

        async with websockets.serve(self._handler, host, port, ssl=self.ssl_context, max_size=MAX_FRAME_SIZE):
            await self.stop_event

    async def _handler(self, websocket):
        conn = None
        try:
            conn = WsConnection(websocket, self.aio_context, self.connector, self.ssl_context)
            self.add_connection(conn)
            await self._read_loop(conn)
            self.close_connection(conn)
        except ConnectionClosedOK as ex:
            conn_info = str(conn) if conn else "N/A"
            log.debug(f"Connection {conn_info} is closed by peer: {secure_format_exception(ex)}")

    @staticmethod
    async def _read_loop(conn: WsConnection):
        while not conn.closing:
            # Reading from websocket and call receiver CB
            frame = await conn.websocket.recv()
            conn.process_frame(frame)
