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
import logging
from asyncio import CancelledError, IncompleteReadError, StreamReader, StreamWriter

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)


class AioConnection(Connection):
    def __init__(
        self,
        connector: ConnectorInfo,
        aio_ctx: AioContext,
        reader: StreamReader,
        writer: StreamWriter,
        secure: bool = False,
    ):
        super().__init__(connector)
        self.reader = reader
        self.writer = writer
        self.aio_ctx = aio_ctx
        self.closing = False
        self.secure = secure
        self.conn_props = self._get_aio_properties()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True

        if not self.writer:
            return

        self.writer.close()
        self.aio_ctx.run_coro(self.writer.wait_closed())

    def send_frame(self, frame: BytesAlike):
        try:
            self.aio_ctx.run_coro(self._async_send_frame(frame))
        except Exception as ex:
            log.error(f"Error calling send coroutine for connection {self}: {secure_format_exception(ex)}")

    async def read_loop(self):
        try:
            while not self.closing:
                frame = await self._async_read_frame()
                self.process_frame(frame)

        except IncompleteReadError:
            if log.isEnabledFor(logging.DEBUG):
                closer = "locally" if self.closing else "by peer"
                log.debug(f"Connection {self} is closed {closer}")
        except CancelledError as error:
            log.debug(f"Connection {self} is closed by peer: {error}")
        except Exception as ex:
            log.error(f"Read error for connection {self}: {secure_format_exception(ex)}")

    # Internal methods

    async def _async_send_frame(self, frame: BytesAlike):
        try:
            self.writer.write(frame)
            await self.writer.drain()
        except Exception as ex:
            if not self.closing:
                log.error(f"Error sending frame for connection {self}: {secure_format_exception(ex)}")

    async def _async_read_frame(self):

        prefix_buf = await self.reader.readexactly(PREFIX_LEN)
        prefix = Prefix.from_bytes(prefix_buf)

        # Prefix only message
        if prefix.length == PREFIX_LEN:
            return prefix_buf

        if prefix.length > MAX_FRAME_SIZE:
            raise CommError(CommError.BAD_DATA, f"Frame exceeds limit ({prefix.length} > {MAX_FRAME_SIZE}")

        remaining = await self.reader.readexactly(prefix.length - PREFIX_LEN)

        return prefix_buf + remaining

    def _get_aio_properties(self) -> dict:

        conn_props = {}
        if not self.writer:
            return conn_props

        fileno = 0
        local_addr = self.writer.get_extra_info("sockname", "")
        if isinstance(local_addr, tuple):
            local_addr = f"{local_addr[0]}:{local_addr[1]}"
        else:
            sock = self.writer.get_extra_info("socket", None)
            if sock:
                fileno = sock.fileno()
            local_addr = f"{local_addr}:{fileno}"

        peer_addr = self.writer.get_extra_info("peername", "")
        if isinstance(peer_addr, tuple):
            peer_addr = f"{peer_addr[0]}:{peer_addr[1]}"
        else:
            peer_addr = f"{peer_addr}:{fileno}"

        conn_props[DriverParams.LOCAL_ADDR.value] = local_addr
        conn_props[DriverParams.PEER_ADDR.value] = peer_addr

        peer_cert = self.writer.get_extra_info("peercert")
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
