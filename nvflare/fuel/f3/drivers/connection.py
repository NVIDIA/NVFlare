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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

import uuid as uuid

Bytes = Union[bytes, bytearray, memoryview]


class ConnState(Enum):
    IDLE = 1           # Initial state
    CONNECTED = 2      # New connection
    CLOSED = 3         # Connection is closed


class FrameReceiver(ABC):

    @abstractmethod
    def process_frame(self, frame: Bytes):
        """Frame received callback

         Args:
             frame: The frame received

         Raises:
             CommError: If any error happens while processing the frame
        """
        pass


class Connection(ABC):
    """FCI connection spec. A connection is used to transfer opaque frames.
    """

    def __init__(self):
        self.name = str(uuid.uuid4())
        self.state = ConnState.IDLE
        self.frame_receiver = None

    @abstractmethod
    def get_conn_properties(self) -> dict:
        """Get connection specific properties, like peer address, TLS certificate etc

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def close(self):
        """Close connection

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def send_frame(self, frame: Bytes):
        """Send a SFM frame through the connection to the remote endpoint.

        Args:
            frame: The frame to be sent

        Raises:
            CommError: If any error happens while sending the frame
        """
        pass

    def register_frame_receiver(self, receiver: FrameReceiver):
        """Register frame receiver

        Args:
            receiver: The frame receiver

        Raises:
            CommError: If any error happens while processing the frame
        """
        self.frame_receiver = receiver
