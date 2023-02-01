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
import os
import socket
from socketserver import ThreadingTCPServer, TCPServer
from typing import List, Dict, Any

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams, Connector, DriverCap
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.fuel.f3.drivers.socket_conn import ConnectionHandler, SocketConnection
from nvflare.fuel.hci.security import get_certificate_common_name

log = logging.getLogger(__name__)


class TcpStreamServer(ThreadingTCPServer):

    TCPServer.allow_reuse_address = True

    def __init__(self, driver: 'Driver', connector: Connector):
        self.driver = driver
        self.connector = connector

        params = connector.params
        self.ssl_context = get_ssl_context(params, ssl_server=True)

        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))
        self.local_addr = f"{host}:{port}"

        TCPServer.__init__(self, (host, port), ConnectionHandler, False)

        if self.ssl_context:
            self.socket = self.ssl_context.wrap_socket(self.socket, server_side=True)

        try:
            self.server_bind()
            self.server_activate()
        except BaseException as ex:
            log.error(f"{os.getpid()}: Error binding to  {host}:{port}: {ex}")
            self.server_close()
            raise


class TcpDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        self.server = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["tcp", "stcp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: False,
            DriverCap.SUPPORT_SSL.value: True
        }

    def listen(self, connector: Connector):
        self.connector = connector
        self.server = TcpStreamServer(self, connector)
        self.server.serve_forever()

    def connect(self, connector: Connector):
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        context = get_ssl_context(params, ssl_server=False)
        if context:
            sock = context.wrap_socket(sock)

        sock.connect((host, port))

        peer = sock.getpeername()
        conn_props = {DriverParams.PEER_ADDR.value: f"{peer[0]}:{peer[1]}"}
        local = sock.getsockname()
        conn_props[DriverParams.LOCAL_ADDR.value] = f"{local[0]}:{local[1]}"

        if context:
            cn = get_certificate_common_name(sock.getpeercert())
            if cn:
                conn_props[DriverParams.PEER_CN.value] = cn

        connection = SocketConnection(sock, connector, conn_props)
        self.add_connection(connection)

        try:
            connection.read_loop()
        except BaseException as ex:
            log.error(f"Active connection {connection.name} closed due to error: {ex}")

        finally:
            if connection:
                self.close_connection(connection)

    def shutdown(self):
        self.close_all()
        if self.server:
            self.server.shutdown()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):

        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "stcp"

        host = resources.get("host") if resources else None
        if not host:
            host = "localhost"

        port = net_utils.get_open_tcp_port(resources)
        if not port:
            raise CommError(CommError.BAD_CONFIG, "Can't find an open port in the specified range")

        # Always listen on all interfaces
        listening_url = f"{scheme}://0:{port}"
        connect_url = f"{scheme}://{host}:{port}"

        return connect_url, listening_url
