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
from enum import Enum


class State(Enum):
    IDLE = 1,
    READY = 2,
    CONNECTING = 3,
    DISCONNECTING = 4


class Endpoint(ABC):

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.state = State.IDLE
        self.relay = None

    @abstractmethod
    def get_certificate(self) -> dict:
        pass


class TcpEndpoint(Endpoint):

    def get_certificate(self):
        return None

    def get_local_addr(self):
        return ""

    def get_local_port(self):
        return 0

    def get_remote_addr(self):
        return ""

    def get_remote_port(self):
        return 0