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
from abc import ABC, abstractmethod

from nvflare.fuel.f3.conn_state import ConnState


class Endpoint:

    CERTIFICATE = "certificate"

    def __init__(self, name: str, url: str, properties: dict = None):
        self.name = name
        self.url = url
        self.state = ConnState.IDLE

        # public properties exchanged while handshake
        self.properties = properties if properties else {}

        # Connection properties like peer address, certificate
        self.conn_props = {}

    def set_prop(self, key, value):
        self.properties[key] = value

    def get_pro(self, key):
        return self.properties.get(key)

    def get_certificate(self) -> dict:
        return self.conn_props(Endpoint.CERTIFICATE)


class EndpointMonitor(ABC):
    """Monitor for endpoint lifecycle changes"""

    @abstractmethod
    def state_change(self, endpoint: Endpoint, state: ConnState):
        pass
