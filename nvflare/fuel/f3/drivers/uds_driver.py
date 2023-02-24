# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
import random
import socket
from socketserver import ThreadingUnixStreamServer, UnixStreamServer
from typing import Any, Dict, List

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.base_driver import BaseDriver
from nvflare.fuel.f3.drivers.driver import ConnectorInfo, Driver
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.socket_conn import ConnectionHandler, SocketConnection

log = logging.getLogger(__name__)


class SocketStreamServer(ThreadingUnixStreamServer):
    def __init__(self, path: str, driver: Driver, connector: ConnectorInfo):
        self.path = path
        self.driver = driver
        self.connector = connector
        self.ssl_context = None  # SSL is not supported
        self.local_addr = path

        UnixStreamServer.__init__(self, path, ConnectionHandler, False)

        try:
            self.server_bind()
            self.server_activate()
        except Exception as ex:
            log.error(f"Error binding to the path {path}: {ex}")
            self.server_close()
            raise ex


class UdsDriver(BaseDriver):
    """Transport driver for Unix Domain Socket"""

    def __init__(self):
        super().__init__()
        self.server = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["uds"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.HEARTBEAT.value: False, DriverCap.SUPPORT_SSL.value: False}

    def listen(self, connector: ConnectorInfo):
        self.connector = connector

        socket_path = self.get_socket_path(connector.params)
        if not self.remove_socket_file(socket_path):
            raise CommError(CommError.ERROR, f"Can't remove existing socket file: {socket_path}")

        try:
            self.server = SocketStreamServer(socket_path, self, connector)
            self.server.serve_forever()
        finally:
            self.remove_socket_file(socket_path)

    def connect(self, connector: ConnectorInfo):
        self.connector = connector

        socket_path = self.get_socket_path(connector.params)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)

        connection = SocketConnection(sock, connector)
        self.add_connection(connection)
        connection.read_loop()
        self.close_connection(connection)

    def shutdown(self):
        self.close_all()
        if self.server:
            self.server.shutdown()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):

        socket_path = resources.get("socket")
        if not socket_path:
            prefix = resources.get("socket_prefix")
            if not prefix:
                prefix = "/tmp/nvflare"

            for i in range(10):
                n = random.randint(1000, 999999)
                socket_path = f"{prefix}_{n}"
                if not os.path.exists(socket_path):
                    break
                log.debug(f"Socket path {socket_path} exists, retrying another one...")

        url = f"{scheme}://{socket_path}"
        return url, url

    @staticmethod
    def get_socket_path(params: dict) -> str:
        socket_path = params.get(DriverParams.SOCKET.value)
        if socket_path:
            return socket_path

        host = params.get(DriverParams.HOST.value)
        path = params.get(DriverParams.PATH.value)

        socket_path = host + path
        if not socket_path.startswith("/"):
            socket_path = "/" + socket_path

        return socket_path

    @staticmethod
    def remove_socket_file(socket_path: str) -> bool:
        try:
            os.unlink(socket_path)
        except OSError as error:
            log.debug(f"Removing {socket_path} : {error}")

        # return True if file is removed
        return not os.path.exists(socket_path)
