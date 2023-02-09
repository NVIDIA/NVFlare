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
import logging
from asyncio import StreamReader, StreamWriter, IncompleteReadError

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.connection import Connection, BytesAlike
from nvflare.fuel.f3.drivers.driver import Connector
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.prefix import Prefix, PREFIX_LEN

log = logging.getLogger(__name__)

MAX_FRAME_SIZE = 1024*1024*1024


class AioConnection(Connection):

    def __init__(self, connector: Connector, aio_ctx: AioContext, reader: StreamReader, writer: StreamWriter):
        super().__init__(connector)
        self.reader = reader
        self.writer = writer
        self.aio_ctx = aio_ctx
        self.closing = False

    def get_conn_properties(self) -> dict:

        conn_props = {}
        if not self.writer:
            return conn_props

        local_addr = self.writer.get_extra_info("sockname", "")
        if isinstance(local_addr, tuple):
            local_addr = f"{local_addr[0]}:{local_addr[1]}"

        peer_addr = self.writer.get_extra_info("peername", "")
        if isinstance(peer_addr, tuple):
            peer_addr = f"{peer_addr[0]}:{peer_addr[1]}"

        conn_props[DriverParams.LOCAL_ADDR.value] = local_addr
        conn_props[DriverParams.PEER_ADDR.value] = peer_addr

        return conn_props

    def close(self):
        self.closing = True

        if not self.writer:
            return

        self.writer.close()
        self.aio_ctx.run_coro(self.writer.wait_closed())

    def send_frame(self, frame: BytesAlike):
        try:
            self.aio_ctx.run_coro(self._async_send_frame(frame))
        except BaseException as ex:
            log.error(f"Error calling send coroutine for connection {self}: {ex}")

    async def read_loop(self):
        try:
            while not self.closing:
                # Reading from websocket and call receiver CB
                frame = await self._async_read_frame()
                if log.isEnabledFor(logging.DEBUG):
                    prefix = Prefix.from_bytes(frame)
                    log.debug(f"Received frame: {prefix} on {self}")

                if self.frame_receiver:
                    self.frame_receiver.process_frame(frame)
                else:
                    log.error("Frame receiver not registered")
        except IncompleteReadError:
            if log.isEnabledFor(logging.DEBUG):
                closer = "locally" if self.closing else "by peer"
                log.debug(f"Connection {self} is closed {closer}")
        except BaseException as ex:
            log.error(f"Read error for connection {self}: {ex}")

    # Internal methods

    async def _async_send_frame(self, frame: BytesAlike):
        try:
            self.writer.write(frame)
            await self.writer.drain()
        except BaseException as ex:
            log.error(f"Error sending frame for connection {self}: {ex}")

    async def _async_read_frame(self):

        prefix_buf = await self.reader.readexactly(PREFIX_LEN)
        prefix = Prefix.from_bytes(prefix_buf)

        # Prefix only message
        if prefix.length == PREFIX_LEN:
            return prefix_buf

        if prefix.length > MAX_FRAME_SIZE:
            raise CommError(CommError.BAD_DATA, f"Frame exceeds limit ({prefix.length} > {MAX_FRAME_SIZE}")

        remaining = await self.reader.readexactly(prefix.length-PREFIX_LEN)

        return prefix_buf + remaining


