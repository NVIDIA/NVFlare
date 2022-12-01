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
from typing import Optional

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.conn_state import ConnState
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.frame import Frame
from nvflare.fuel.f3.headers import Headers


class FrameReceiver(ABC):

    @abstractmethod
    def process_frame(self, endpoint: Endpoint, header: bytes, payload: bytes):
        """Frame received callback

         Args:
             endpoint: The endpoint where the frame is from
             header: Encoded header bytes
             payload: Payload bytes

         Raises:
             CommError: If any error happens while processing the frame
             """

        pass


class DriverSpec(ABC):
    """Transport driver spec
    A transport driver is responsible for sending a frame to the remote endpoint and calling
    FrameReceiver when a frame arrives from remote endpoint.

    A frame is made of header (including prefix) and payload. They are provided separately to reduce
    memory copying.

    When sending, driver should make 2 writes,
        header, payload

    When receiving, driver for framed transport can send header and payload separately.

    For stream transport like pipe/socket/tcp, drivers needs to handle framing. It should make 3 reads,
        Read 12-byte prefix and call decode_prefix to get header length
        Read header and prefix+header is header
        Read payload

    """

    def __init__(self):
        self.state = ConnState.IDLE
        self.frame_receiver = None

    def get_state(self) -> ConnState:
        """Get connection state

        Raises:
            CommError: If any errors
        """
        return self.state

    def get_name(self) -> str:
        """Return the name of the driver, something like 'http'.
        By default, it returns class name
        """
        return self.__class__.__name__

    @abstractmethod
    def get_conn_properties(self) -> dict:
        """Returns connection specific properties, like peer address, certificate etc

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def listen(self):
        """Start listening to connection

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def connect(self):
        """Start connecting to remote endpoint

        Raises:
            CommError: If any errors
        """
        pass

    def update_headers(self, headers: Headers) -> Optional[Headers]:
        """Driver can use this call to update headers. Return None if no change

         Args:
             headers: Current headers for the next frame

         Raises:
             CommError: If any error happens while sending the frame
         """

        return None

    @abstractmethod
    def send_frame(self, header: bytes, payload: bytes):
        """Send a SFM frame through the driver to the remote endpoint
        The header and payload are separate to reduce buffer copying

        Args:
            header: The encoded header
            payload: The payload of the frame

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

    @staticmethod
    def get_lengths(prefix: bytes) -> (int, int, int):
        """Get frame, header, payload length

        Args:
            prefix: The buffer for prefix

        Raises:
            CommError: If buffer is too small
        """

        if len(prefix) < Frame.PREFIX_LEN:
            raise CommError(CommError.BAD_DATA, f"Prefix too short: {len(prefix)}")

        frame = Frame()
        frame.decode_prefix(prefix)
        return frame.length, frame.header_len, frame.get_payload_len()
