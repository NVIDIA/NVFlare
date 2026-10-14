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
import os
import socket
from socketserver import TCPServer, ThreadingTCPServer
from typing import Any, Dict, List

from nvflare.fuel.f3.comm_config_utils import requires_secure_connection
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.driver import ConnectorInfo, Driver
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context, get_tcp_urls
from nvflare.fuel.f3.drivers.socket_conn import ConnectionHandler, SocketConnection
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)


class TcpStreamServer(ThreadingTCPServer):

    TCPServer.allow_reuse_address = True
    daemon_threads = True

    def __init__(self, driver: Driver, connector: ConnectorInfo):
        self.driver = driver
        self.connector = connector

        params = connector.params
        self.ssl_context = get_ssl_context(params, ssl_server=True)

        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))
        self.local_addr = f"{host}:{port}"

        TCPServer.__init__(self, (host, port), ConnectionHandler, False)

        try:
            self.server_bind()
            self.server_activate()
        except Exception as ex:
            log.error(f"{os.getpid()}: Error binding to  {host}:{port}: {secure_format_exception(ex)}")
            self.server_close()
            raise

    def get_request(self):
        sock, address = super().get_request()
        if self.ssl_context:
            try:
                sock = self.ssl_context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
            except Exception:
                sock.close()
                raise
        return sock, address


class TcpDriver(BaseDriver):
    def __init__(self):
        super().__init__()
        self.server = None
        self._pending_sockets = set()

    def register_pending_socket(self, sock):
        with self.conn_lock:
            admitted = not self._connection_admission_closed
            if admitted:
                self._pending_sockets.add(sock)
        if not admitted:
            sock.close()
        return admitted

    def release_pending_socket(self, sock):
        with self.conn_lock:
            self._pending_sockets.discard(sock)

    @staticmethod
    def supported_transports() -> List[str]:
        return ["tcp", "stcp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

    def listen(self, connector: ConnectorInfo):
        self.connector = connector
        server = TcpStreamServer(self, connector)
        # Pair publication with shutdown's snapshot. Once published, serve_forever()
        # must run so that a concurrent server.shutdown() can finish.
        with self.conn_lock:
            stopped = self._connection_admission_closed or connector.stopped.is_set()
            if not stopped:
                self.server = server
        if stopped:
            server.server_close()
        else:
            server.serve_forever()

    def connect(self, connector: ConnectorInfo):
        self.connector = connector
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            timeout = params.get(DriverParams.CONNECT_TIMEOUT)
            sock.settimeout(float(timeout) if timeout is not None else None)
            context = get_ssl_context(params, ssl_server=False)
            if context:
                sock = context.wrap_socket(sock)
            if not self.register_pending_socket(sock):
                return
            try:
                sock.connect((host, port))
                sock.settimeout(None)
            finally:
                self.release_pending_socket(sock)
        except Exception:
            sock.close()
            raise

        connection = SocketConnection(sock, connector, bool(context))
        if not self.add_connection(connection):
            return
        connection.read_loop()
        self.close_connection(connection)

    def shutdown(self):
        self.stop_connection_admission()
        with self.conn_lock:
            server = self.server
            pending = list(self._pending_sockets)
        for sock in pending:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if server:
            server.shutdown()
        self.close_all()
        if server:
            server.server_close()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = requires_secure_connection(resources)
        if secure:
            scheme = "stcp"

        return get_tcp_urls(scheme, resources)
