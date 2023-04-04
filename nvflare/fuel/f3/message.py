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
from typing import Any

from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.endpoint import Endpoint


class AppIds:
    """Reserved application IDs"""

    ALL = 0
    DEFAULT = 1
    CELL_NET = 2
    PUB_SUB = 3


class Headers(dict):

    # Reserved Keys
    MSG_ID = "_MSG_ID_"
    TOPIC = "_TOPIC_"
    DEST = "_DEST_"
    JOB_ID = "_JOB_ID_"


class Message:
    def __init__(self, headers: Headers, payload: Any):
        """Construct an FCI message"""

        self.headers = headers
        self.payload = payload

    def set_header(self, key: str, value):
        if self.headers is None:
            self.headers = {}

        self.headers[key] = value

    def add_headers(self, headers: dict):
        if self.headers is None:
            self.headers = {}
        self.headers.update(headers)

    def get_header(self, key: str, default=None):
        if self.headers is None:
            return None

        return self.headers.get(key, default)

    def remove_header(self, key: str):
        if self.headers:
            self.headers.pop(key, None)

    def set_prop(self, key: str, value):
        setattr(self, key, value)

    def get_prop(self, key: str, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MessageReceiver(ABC):
    @abstractmethod
    def process_message(self, endpoint: Endpoint, connection: Connection, app_id: int, message: Message):
        pass
