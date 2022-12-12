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

from nvflare.fuel.f3.drivers.connection import ConnState, Connection


class CommonKeys:
    URL = "url"
    HOST = "host"
    PORT = "port"


class Mode(Enum):
    ACTIVE = 0
    PASSIVE = 1


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

    def __init__(self, conn_props: dict, resource_policy: dict):
        self.state = ConnState.IDLE
        self.conn_props = conn_props if conn_props else {}
        self.resource_policy = resource_policy if resource_policy else {}
        self.conn_monitor = None

    def get_name(self) -> str:
        """Return the name of the driver, used for logging
        By default, it returns class name
        """
        return self.__class__.__name__

    @abstractmethod
    def supported_transports(self) -> List[str]:
        """Return a list of transports supported by this driver, for example
           ["http", "https", "ws", "wss"]
        """
        pass

    @abstractmethod
    def listen(self, properties: dict):
        """Start the driver in passive mode

        Args:
            properties: Properties needed to start the driver

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def connect(self, properties: dict):
        """Start the driver in active mode

        Args:
            properties: Properties needed to start the driver

        Raises:
            CommError: If any errors
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
