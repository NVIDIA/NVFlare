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
import socket
from socketserver import BaseRequestHandler
from typing import Any, Union

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers.driver import ConnectorInfo
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)


class SocketConnection(Connection):
    def __init__(self, sock: Any, connector: ConnectorInfo, secure: bool = False):
        super().__init__(connector)
        self.sock = sock
        self.secure = secure
        self.closing = False
        self.conn_props = self._get_socket_properties()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True

        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError as error:
                log.debug(f"Connection {self} is already closed: {error}")

            self.sock.close()

    def send_frame(self, frame: BytesAlike):
        try:
            self.sock.sendall(frame)
        except Exception as ex:
            if not self.closing:
                raise CommError(CommError.ERROR, f"Error sending frame on conn {self}: {secure_format_exception(ex)}")

    def read_loop(self):
        try:
            self.read_frame_loop()
        except CommError as error:
            if error.code == CommError.CLOSED:
                log.debug(f"Connection {self.name} is closed by peer")
            else:
                log.debug(f"Connection {self.name} is closed due to error: {error}")
        except Exception as ex:
            if self.closing:
                log.debug(f"Connection {self.name} is closed")
            else:
                log.debug(f"Connection {self.name} is closed due to error: {secure_format_exception(ex)}")

    def read_frame_loop(self):
        # read_frame throws exception on stale/bad connection so this is not a dead loop
        while not self.closing:
            frame = self.read_frame()
            self.process_frame(frame)

    def read_frame(self) -> BytesAlike:

        prefix_buf = bytearray(PREFIX_LEN)
        self.read_into(prefix_buf, 0, PREFIX_LEN)
        prefix = Prefix.from_bytes(prefix_buf)

        if prefix.length == PREFIX_LEN:
            return prefix_buf

        if prefix.length > MAX_FRAME_SIZE:
            raise CommError(CommError.BAD_DATA, f"Frame exceeds limit ({prefix.length} > {MAX_FRAME_SIZE}")

        frame = bytearray(prefix.length)
        frame[0:PREFIX_LEN] = prefix_buf
        self.read_into(frame, PREFIX_LEN, prefix.length - PREFIX_LEN)

        return frame

    def read_into(self, buffer: BytesAlike, offset: int, length: int):
        if isinstance(buffer, memoryview):
            view = buffer
        else:
            view = memoryview(buffer)

        if offset:
            view = view[offset:]

        remaining = length
        while remaining:
            n = self.sock.recv_into(view, remaining)
            if n == 0:
                raise CommError(CommError.CLOSED, f"Connection {self.name} is closed by peer")
            view = view[n:]
            remaining -= n

    @staticmethod
    def _format_address(addr: Union[str, tuple], fileno: int) -> str:

        if isinstance(addr, tuple):
            result = f"{addr[0]}:{addr[1]}"
        else:
            result = f"{addr}:{fileno}"

        return result

    def _get_socket_properties(self) -> dict:
        conn_props = {}

        try:
            peer = self.sock.getpeername()
            fileno = self.sock.fileno()
        except OSError as ex:
            peer = "N/A"
            fileno = 0
            log.debug(f"getpeername() error: {secure_format_exception(ex)}")

        conn_props[DriverParams.PEER_ADDR.value] = self._format_address(peer, fileno)

        local = self.sock.getsockname()
        conn_props[DriverParams.LOCAL_ADDR.value] = self._format_address(local, fileno)

        if self.secure:
            cert = self.sock.getpeercert()
            if cert:
                cn = get_certificate_common_name(cert)
            else:
                cn = "N/A"
            conn_props[DriverParams.PEER_CN.value] = cn

        return conn_props


class ConnectionHandler(BaseRequestHandler):
    def handle(self):

        # noinspection PyUnresolvedReferences
        connection = SocketConnection(self.request, self.server.connector, self.server.ssl_context)
        # noinspection PyUnresolvedReferences
        driver = self.server.driver

        driver.add_connection(connection)
        connection.read_loop()
        driver.close_connection(connection)
