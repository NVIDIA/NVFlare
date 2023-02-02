#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from urllib.parse import urlparse, urlencode, parse_qsl

from nvflare.fuel.f3.drivers.connection import ConnState, Connection
from nvflare.fuel.f3.drivers.connnector import Connector


class DriverParams(str, Enum):

    # URL components. Those parameters are part of the URL, no need to be included in query string
    # URL = SCHEME://HOST:PORT/PATH;PARAMS?QUERY#FRAG
    URL = "url"
    SCHEME = "scheme"
    HOST = "host"
    PORT = "port"
    PATH = "path"
    PARAMS = "params"
    FRAG = "frag"
    QUERY = "query"

    # Other parameters
    CA_CERT = "ca_cert"
    SERVER_CERT = "server_cert"
    SERVER_KEY = "server_key"
    CLIENT_CERT = "client_cert"
    CLIENT_KEY = "client_key"
    SECURE = "secure"
    PORTS = "ports"


class ConnMonitor(ABC):

    @abstractmethod
    def state_change(self, connection: Connection):
        """Driver state change notification, including new connections

         Args:
             connection: The connection that state has changed

         Raises:
             CommError: If any error happens while processing the frame
        """
        pass


class Driver(ABC):
    """Transport driver spec
    A transport driver is responsible for establishing connections. The connections are used
    to transport frames to remote endpoint.

    The frame is opaque to the driver, except the length (first 4-bytes), which may be needed to
    determine the frame boundary on stream-based transports like TCP or Pipe.

    """

    def __init__(self):
        self.state = ConnState.IDLE
        self.conn_monitor = None

    def get_name(self) -> str:
        """Return the name of the driver, used for logging
        By default, it returns class name
        """
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def supported_transports() -> List[str]:
        """Return a list of transports supported by this driver, for example
           ["http", "https", "ws", "wss"]
        """
        pass

    @abstractmethod
    def listen(self, connector: Connector):
        """Start the driver in passive mode

        Args:
            connector: Connector with parameters

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def connect(self, connector: Connector):
        """Start the driver in active mode

        Args:
            connector: Connector with parameters

        Raises:
            CommError: If any errors
        """
        pass

    @staticmethod
    @abstractmethod
    def get_connect_url(scheme: str, resources: dict):
        """Get the URL that can be used to connect to this endpoint

        Args:
            scheme: A scheme supported by the driver, like http or https
            resources: User specified resources like host and port ranges.

        Raises:
            CommError: If no free port can be found
        """
        pass

    @staticmethod
    @abstractmethod
    def get_listening_url(scheme: str, resources: dict):
        """Get the URL that can be used to listen for connections to this endpoint

        Args:
            scheme: A scheme supported by the driver, like http or https
            resources: User specified resources like host and port ranges.

        Raises:
            CommError: If no free port can be found
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Stop driver and disconnect all the connections created by it

        Raises:
            CommError: If any errors
        """
        pass

    def register_conn_monitor(self, monitor: ConnMonitor):
        """Register a monitor for connection state change, including new connections
        """
        self.conn_monitor = monitor

    @staticmethod
    def parse_url(url: str) -> dict:
        if not url:
            return {}

        params = {DriverParams.URL.value: url}
        parsed_url = urlparse(url)
        params[DriverParams.SCHEME.value] = parsed_url.scheme
        parts = parsed_url.netloc.split(":")
        if len(parts) >= 1:
            host = parts[0]
            # Host is required in URL. 0 is used as the placeholder for empty host
            if host == "0":
                host = ""
            params[DriverParams.HOST.value] = host
        if len(parts) >= 2:
            params[DriverParams.PORT.value] = parts[1]

        params[DriverParams.PATH.value] = parsed_url.path
        params[DriverParams.PARAMS.value] = parsed_url.params
        params[DriverParams.QUERY.value] = parsed_url.query
        params[DriverParams.FRAG.value] = parsed_url.fragment

        if parsed_url.query:
            for k, v in parse_qsl(parsed_url.query):
                # Only last one is saved if duplicate keys
                params[k] = v

        return params

    @staticmethod
    def encode_url(params: dict) -> str:

        # Original URL is not needed
        params.pop(DriverParams.URL.value, None)

        scheme = params.pop(DriverParams.SCHEME.value, None)
        host = params.pop(DriverParams.HOST.value, None)
        if not host:
            host = "0"
        port = params.pop(DriverParams.PORT.value, None)
        path = params.pop(DriverParams.PATH.value, None)
        parameters = params.pop(DriverParams.PARAMS.value, None)
        # Encoded query is not needed
        params.pop(DriverParams.QUERY.value, None)
        frag = params.pop(DriverParams.FRAG.value, None)

        url = f"{scheme}://{host}"
        if port:
            url += ":" + str(port)

        if path:
            url += path

        if parameters:
            url += ";" + parameters

        if params:
            url += urlencode(params)

        if frag:
            url += '#' + frag

        return url
