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
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from nvflare.fuel.f3.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo


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
    determine the frame boundary on stream-based transports like TCP or sockets.

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

    @staticmethod
    @abstractmethod
    def capabilities() -> Dict[str, Any]:
        """Return a dictionary of capabilities of the driver."""
        pass

    @abstractmethod
    def listen(self, connector: ConnectorInfo):
        """Start the driver in passive mode

        Args:
            connector: Connector with parameters

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def connect(self, connector: ConnectorInfo):
        """Start the driver in active mode

        Args:
            connector: Connector with parameters

        Raises:
            CommError: If any errors
        """
        pass

    @staticmethod
    @abstractmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        """Get active and passive URL pair based on resources

        Args:
            scheme: A scheme supported by the driver, like http or https
            resources: User specified resources like host and port ranges.

        Returns:
            A tuple with active and passive URLs

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
        """Register a monitor for connection state change, including new connections"""
        self.conn_monitor = monitor
