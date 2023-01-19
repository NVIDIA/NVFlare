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
from socketserver import TCPServer, ThreadingTCPServer
from typing import List

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import net_utils
from nvflare.fuel.f3.drivers.driver import Connector, Driver, DriverParams
from nvflare.fuel.f3.drivers.socket_driver import ConnectionHandler, SocketDriver, StreamConnection

log = logging.getLogger(__name__)


class TcpStreamServer(ThreadingTCPServer):

    TCPServer.allow_reuse_address = True

    def __init__(self, driver: "Driver", connector: Connector):
        self.driver = driver
        self.connector = connector

        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))

        TCPServer.__init__(self, (host, port), ConnectionHandler, False)

        # Wrap SSL here

        try:
            self.server_bind()
            self.server_activate()
        except BaseException as ex:
            log.error(f"Error binding to  {host}:{port}: {ex}")
            self.server_close()
            raise


class TcpDriver(SocketDriver):
    @staticmethod
    def supported_transports() -> List[str]:
        return ["tcp", "stcp"]

    def listen(self, connector: Connector):
        self.connector = connector
        self.server = TcpStreamServer(self, connector)
        self.server.serve_forever()

    def connect(self, connector: Connector):
        params = connector.params
        address = params.get(DriverParams.HOST.value), int(params.get(DriverParams.PORT.value))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)

        connection = StreamConnection(sock, connector, address)
        self.add_connection(connection)
        connection.read_loop()

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
