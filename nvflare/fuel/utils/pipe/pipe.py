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


class Pipe(ABC):
    @abstractmethod
    def open(self, name: str, me: str):
        """Open the pipe

        Args:
            name: name of the pipe
            me: my endpoint name. A pipe has two endpoints. Each endpoint must have a unique name.

        Returns: None

        """
        pass

    @abstractmethod
    def clear(self):
        """Clear the pipe"""
        pass

    @abstractmethod
    def send(self, topic: str, data: bytes, timeout=None) -> bool:
        """Send message with the specified topic and data to the peer.

        Args:
            topic: topic of the message
            data: data of the message
            timeout: if specified, number of secs to wait for the peer to read the message.

        Returns: whether the message is read by the peer.
        If timeout is not specified, always return False.

        """
        pass

    @abstractmethod
    def receive(self, timeout=None) -> (str, bytes):
        """Try to receive data from peer.

        Args:
            timeout: how long (number of seconds) to try

        Returns: topic and data, if data is received; (None, None) if not

        """
        pass

    @abstractmethod
    def close(self):
        """Close the pipe

        Returns: None

        """
        pass
