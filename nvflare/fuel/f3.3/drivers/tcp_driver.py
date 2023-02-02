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
from socketserver import ThreadingTCPServer, TCPServer, BaseRequestHandler
from typing import List, Any, Union, Optional

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection
from nvflare.fuel.f3.drivers.driver import Driver, DriverParams, Connector

log = logging.getLogger(__name__)

QUEUE_SIZE = 16
THREAD_POOL_SIZE = 8
LO_PORT = 1025
HI_PORT = 65535


class StreamConnection(Connection):

    def __init__(self, stream: Any, connector: Connector, peer_address):
        super().__init__(connector)
        self.stream = stream
        self.closing = False
        self.peer_address = peer_address

    def get_conn_properties(self) -> dict:
        addr = self.websocket.remote_address
        if addr:
            return {"peer_host": addr[0], "peer_port": addr[1]}
        else:
            return {}

    def close(self):
        self.closing = True

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        try:
            self.stream.sendall(frame)
        except BaseException as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {ex}")


class ConnectionHandler(BaseRequestHandler):
    pass


class StreamServer(ThreadingTCPServer):

    TCPServer.allow_reuse_address = True

    def __init__(self, connector: Connector):
        self.connector = connector

        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        TCPServer.__init__(self, (host, port), ConnectionHandler, False)

        # Wrap SSL here

        try:
            self.server_bind()
            self.server_activate()
        except BaseException:
            self.server_close()
            raise


class TcpDriver(Driver):

    def __init__(self):
        super().__init__()
        self.connections = {}
        self.connector = None

    @staticmethod
    def supported_transports() -> List[str]:
        return ["tcp", "stcp"]

    def listen(self, connector: Connector):
        pass

    def connect(self, connector: Connector):
        pass

    def shutdown(self):
        pass
