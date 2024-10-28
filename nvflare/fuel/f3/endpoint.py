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
from enum import IntEnum


class EndpointState(IntEnum):
    IDLE = 0  # Initial state
    READY = 1  # Endpoint is ready
    CLOSING = 2  # Endpoint is closing, can't send
    DISCONNECTED = 3  # Endpoint is disconnected
    ERROR = 4  # Endpoint is in error state


class Endpoint:
    """Endpoint represents a logical party in the SFM network. For each communicator,
    there is only one local endpoint. There may be multiple remote endpoints.
    A remote endpoint may be reachable through multiple connections"""

    def __init__(self, name: str, properties: dict = None, conn_props: dict = None):
        """Construct an endpoint

        Args:
            name: The endpoint name
            properties: Public properties exchanged with peer
            conn_props: Connection properties and local credentials like certificates

        Raises:
            CommError: If any error happens while sending the request
        """
        self.name = name
        self.state = EndpointState.IDLE

        # public properties exchanged while handshake
        self.properties = properties if properties else {}

        # Connection properties like peer address, certificate location
        self.conn_props = conn_props if conn_props else {}

    def set_prop(self, key, value):
        self.properties[key] = value

    def get_prop(self, key):
        return self.properties.get(key)


class EndpointMonitor(ABC):
    """Monitor for endpoint lifecycle changes"""

    @abstractmethod
    def state_change(self, endpoint: Endpoint):
        pass
