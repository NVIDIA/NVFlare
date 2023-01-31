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
import threading
from socketserver import BaseRequestHandler
from typing import Any, Union

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection, ConnState, BytesAlike
from nvflare.fuel.f3.drivers.driver import Driver, Connector
from nvflare.fuel.f3.drivers.prefix import PREFIX_LEN, Prefix

log = logging.getLogger(__name__)

MAX_FRAME_SIZE = 1024*1024*1024


class StreamConnection(Connection):

    def __init__(self, stream: Any, connector: Connector, peer_address):
        super().__init__(connector)
        self.stream = stream
        self.closing = False
        self.peer_address = peer_address

    def get_conn_properties(self) -> dict:
        addr = self.peer_address
        if isinstance(addr, tuple):
            return {"peer_host": addr[0], "peer_port": addr[1]}
        elif addr:
            return {"peer_addr": addr}
        else:
            return {}

    def close(self):
        log.debug(f"Connection {self.name} is closing")
        self.closing = True
        if self.stream:
            self.stream.close()

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        try:
            self.stream.sendall(frame)
        except BaseException as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {ex}")

    def read_loop(self):

        while not self.closing:

            frame = self.read_frame()

            if self.frame_receiver:
                self.frame_receiver.process_frame(frame)
            else:
                log.error(f"Frame receiver not registered for connection: {self.name}")

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
            n = self.stream.recv_into(view, remaining)
            if n == 0:
                raise CommError(CommError.CLOSED, f"Connection {self.name} is closed by peer")
            view = view[n:]
            remaining -= n


class ConnectionHandler(BaseRequestHandler):

    def handle(self):

        # ThreadingUnixStreamServer doesn't set client_address
        address = self.client_address if self.client_address else self.server.path

        connection = StreamConnection(self.request, self.server.connector, address)
        self.server.driver.add_connection(connection)

        try:
            connection.read_loop()
        except BaseException as ex:
            log.error(f"Passive connection {connection.name} closed due to error: {ex}")
        finally:
            if connection:
                self.server.driver.close_connection(connection)


class SocketDriver(Driver):
    """Common base class for socket-based drivers"""
    def __init__(self):
        super().__init__()
        self.connections = {}
        self.connector = None
        self.server = None
        self.conn_lock = threading.Lock()

    def shutdown(self):
        with self.conn_lock:
            for _, conn in self.connections.items():
                conn.close()

        if self.server:
            self.server.shutdown()

    def add_connection(self, conn: StreamConnection):
        log.debug(f"New connection created: {self.get_name()}:{conn.name}, peer address: {conn.peer_address}")
        with self.conn_lock:
            self.connections[conn.name] = conn

        if not self.conn_monitor:
            log.error(f"Connection monitor not registered for driver {self.get_name()}")
        else:
            conn.state = ConnState.CONNECTED
            self.conn_monitor.state_change(conn)

    def close_connection(self, conn: StreamConnection):
        log.debug(f"Connection: {self.get_name()}:{conn.name} is disconnected")
        with self.conn_lock:
            self.connections.pop(conn.name)

        if not self.conn_monitor:
            log.error(f"Connection monitor not registered for driver {self.get_name()}")
        else:
            conn.state = ConnState.CLOSED
            self.conn_monitor.state_change(conn)
