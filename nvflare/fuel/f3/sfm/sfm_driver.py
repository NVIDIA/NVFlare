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
from nvflare.fuel.f3.sfm.frame import Frame


class SfmDriver(ABC):

    def __init__(self, url: str, parameters: dict):
        self.url = url
        self.parameters = parameters
        self.state = ConnState.IDLE

    def get_state(self) -> ConnState:
        """Get connection state

        Raises:
            CommError: If any errors
            """

        return self.state

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

    @abstractmethod
    def send_frame(self, frame: Frame):
        """Send a SFM frame through the driver to the remote endpoint

        Args:
            frame: The frame to be sent

        Raises:
            CommError: If any error happens while sending the frame
            """
        pass

    def frame_received(self, frame: Frame):
        """Driver calls this method when a frame arrives

        Args:
            frame: The frame received from remote endpoint

        Raises:
            CommError: If any error happens while processing the frame
            """
        pass

