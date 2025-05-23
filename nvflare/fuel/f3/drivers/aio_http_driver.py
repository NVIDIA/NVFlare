# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
from typing import Any, Dict, List

import aiohttp
from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import StreamResponse

from nvflare.fuel.f3.comm_config_utils import requires_secure_connection
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.driver import ConnectorInfo
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import get_tcp_urls
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)

WS_PATH = "f3"
MAX_FRAME_SIZE = 2 * 1024 * 1024 * 1024  # Set it to 2GB


class WsConnection(Connection):
    def __init__(self, websocket: Any, aio_context: AioContext, connector: ConnectorInfo, ssl_context):
        super().__init__(connector)
        self.websocket = websocket
        self.aio_context = aio_context
        self.closing = False
        self.ssl_context = ssl_context

        self.conn_props = self._get_ws_properties()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True
        self.aio_context.run_coro(self.websocket.close())

    def send_frame(self, frame: BytesAlike):
        self.aio_context.run_coro(self._async_send_frame(frame))

    def _get_ws_properties(self) -> dict:

        conn_props = {}
        local_sock = self.websocket.get_extra_info("sockname")
        if local_sock:
            conn_props[DriverParams.LOCAL_ADDR.value] = f"{local_sock[0]}:{local_sock[1]}"

        peer_sock = self.websocket.get_extra_info("peername")
        if peer_sock:
            conn_props[DriverParams.PEER_ADDR.value] = f"{peer_sock[0]}:{peer_sock[1]}"

        peer_cert = self.websocket.get_extra_info("peercert")
        if peer_cert:
            cn = get_certificate_common_name(peer_cert)
        else:
            cn = "N/A" if self.ssl_context else None

        if cn:
            conn_props[DriverParams.PEER_CN.value] = cn

        return conn_props

    async def _async_send_frame(self, frame: BytesAlike):
        try:
            await self.websocket.send_bytes(frame)
        except Exception as ex:
            log.error(f"Error sending frame for connection {self}, closing: {secure_format_exception(ex)}")
            self.close()


class AioHttpDriver(BaseDriver):
    """Async HTTP driver using aiohttp library"""

    def __init__(self):
        super().__init__()
        self.aio_context = AioContext.get_global_context()
        self.loop = self.aio_context.get_event_loop()
        self.ssl_context = None
        self.stop_event = self.loop.create_future()
        self.app = None
        self.site = None
        self.runner = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["http", "https", "ws", "wss"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

    def listen(self, connector: ConnectorInfo):
        self.connector = connector
        self.ssl_context = net_utils.get_ssl_context(self.connector.params, True)

        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        self.app = web.Application(client_max_size=MAX_FRAME_SIZE)
        self.app.router.add_get(f"/{WS_PATH}", self._websocket_handler)

        async def setup():
            self.runner = web.AppRunner(self.app, access_log=None)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, host, port, ssl_context=self.ssl_context)
            await self.site.start()
            await self.stop_event

        self.aio_context.run_coro(setup()).result()

    def connect(self, connector: ConnectorInfo):
        self.connector = connector
        self.ssl_context = net_utils.get_ssl_context(self.connector.params, False)

        async def async_connect():
            params = connector.params
            host = params.get(DriverParams.HOST.value)
            port = params.get(DriverParams.PORT.value)
            scheme = "wss" if self.ssl_context else "ws"
            url = f"{scheme}://{host}:{port}/{WS_PATH}"

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, ssl_context=self.ssl_context) as ws:
                    await self._connection_handler(ws)

        self.aio_context.run_coro(async_connect()).result()

    def shutdown(self):
        self.aio_context.run_coro(self._async_shutdown())

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = requires_secure_connection(resources)
        if secure:
            scheme = "https"

        return get_tcp_urls(scheme, resources)

    # Internal methods

    async def _connection_handler(self, websocket):
        conn = None
        try:
            conn = WsConnection(websocket, self.aio_context, self.connector, self.ssl_context)
            self.add_connection(conn)
            await self._read_loop(conn)
            self.close_connection(conn)
        except Exception as ex:
            conn_info = str(conn) if conn else "N/A"
            log.error(f"Connection {conn_info} is closed due to error: {secure_format_exception(ex)}")

    async def _websocket_handler(self, request: Request) -> StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_SIZE)
        await ws.prepare(request)
        await self._connection_handler(ws)
        return ws

    @staticmethod
    async def _read_loop(conn: WsConnection):

        async for msg in conn.websocket:
            if msg.type == aiohttp.WSMsgType.BINARY:
                conn.process_frame(msg.data)
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                log.info(f"{conn} is closed by peer")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.error(f"{conn} is closed due to error: {conn.websocket.exception()}")
                break
            else:
                log.info(f"Unknown message type {msg.type} received, ignored")

            if conn.closing:
                log.info(f"Connection {conn} is closed by calling close()")
                break

    async def _async_shutdown(self):
        self.close_all()

        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        if self.app:
            await self.app.shutdown()
            await self.app.cleanup()
            self.app = None

        if self.stop_event:
            self.stop_event.set_result(None)
