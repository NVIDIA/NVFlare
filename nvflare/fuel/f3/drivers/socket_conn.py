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
import socket
from socketserver import BaseRequestHandler
from typing import Any

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import Connection, BytesAlike
from nvflare.fuel.f3.drivers.driver import Connector
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.prefix import PREFIX_LEN, Prefix
from nvflare.fuel.hci.security import get_certificate_common_name

log = logging.getLogger(__name__)

MAX_FRAME_SIZE = 1024*1024*1024


class SocketConnection(Connection):

    def __init__(self, sock: Any, connector: Connector, secure: bool = False):
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
        except BaseException as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {ex}")

    def read_loop(self):
        try:
            self.read_frame_loop()
        except CommError:
            log.info(f"Connection {self.name} is closed by peer")
        except BaseException as ex:
            if self.closing:
                log.debug(f"Connection {self.name} is closed")
            else:
                log.error(f"Connection {self.name} is closed due to error: {ex}")

    def read_frame_loop(self):

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
        self.read_into(frame, PREFIX_LEN, prefix.length-PREFIX_LEN)

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

    def _get_socket_properties(self) -> dict:
        conn_props = {}

        fileno = 0
        try:
            peer = self.sock.getpeername()
            if isinstance(peer, tuple):
                peer_addr = f"{peer[0]}:{peer[1]}"
            else:
                fileno = self.sock.fileno()
                peer_addr = f"{peer}:{fileno}"
        except OSError as ex:
            peer_addr = "N/A"
            log.debug(f"getpeername() error: {ex}")

        conn_props[DriverParams.PEER_ADDR.value] = peer_addr

        local = self.sock.getsockname()
        if isinstance(local, tuple):
            local_addr = f"{local[0]}:{local[1]}"
        else:
            local_addr = f"{local}:{fileno}"

        conn_props[DriverParams.LOCAL_ADDR.value] = local_addr

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
        conn_props = {DriverParams.LOCAL_ADDR.value: self.server.local_addr}
        if self.client_address:
            if isinstance(self.client_address, tuple):
                peer_addr = f"{self.client_address[0]}:{self.client_address[1]}"
            else:
                peer_addr = self.client_address
        else:
            # ThreadingUnixStreamServer doesn't set client_address
            # noinspection PyUnresolvedReferences
            peer_addr = self.server.path
        conn_props[DriverParams.PEER_ADDR.value] = peer_addr

        # noinspection PyUnresolvedReferences
        if self.server.ssl_context:
            cn = get_certificate_common_name(self.request.getpeercert())
            if cn:
                conn_props[DriverParams.PEER_CN.value] = cn

        # noinspection PyUnresolvedReferences
        connection = SocketConnection(self.request, self.server.connector, self.server.ssl_context)
        # noinspection PyUnresolvedReferences
        driver = self.server.driver
        driver.add_connection(connection)

        connection.read_loop()

        driver.close_connection(connection)
