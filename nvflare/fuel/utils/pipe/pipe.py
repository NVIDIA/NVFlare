# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Union

from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.validation_utils import check_str


class Message:

    REQUEST = "REQ"
    REPLY = "REP"

    def __init__(self, msg_type: str, topic: str, data: Any, msg_id=None, req_id=None):
        check_str("msg_type", msg_type)
        if msg_type not in [Message.REPLY, Message.REQUEST]:
            raise ValueError(f"invalid note_type '{msg_type}': must be one of {[Message.REPLY, Message.REQUEST]}")
        self.msg_type = msg_type

        check_str("topic", topic)
        if not topic:
            raise ValueError("topic must not be empty")

        if not re.match("[a-zA-Z0-9_]+$", topic):
            raise ValueError("topic contains invalid char - only alphanumeric and underscore are allowed")

        self.topic = topic

        if not msg_id:
            msg_id = str(uuid.uuid4())

        self.data = data
        self.msg_id = msg_id
        self.req_id = req_id
        self.sent_time = None
        self.received_time = None

    @staticmethod
    def new_request(topic: str, data: Any, msg_id=None):
        return Message(Message.REQUEST, topic, data, msg_id)

    @staticmethod
    def new_reply(topic: str, data: Any, req_msg_id, msg_id=None):
        return Message(Message.REPLY, topic, data, msg_id, req_id=req_msg_id)

    def __str__(self):
        return f"Message(topic={self.topic}, msg_id={self.msg_id}, req_id={self.req_id}, msg_type={self.msg_type})"


class Pipe(ABC):
    def __init__(self, mode: Mode):
        """Creates the pipe.

        Args:
            mode (Mode): Mode of the endpoint. A pipe has two endpoints.
                An endpoint can be either the one that initiates communication or the one listening.
        """
        if mode != Mode.ACTIVE and mode != Mode.PASSIVE:
            raise ValueError(f"mode must be '{Mode.ACTIVE}' or '{Mode.PASSIVE}' but got {mode}")
        self.mode = mode

    @abstractmethod
    def open(self, name: str):
        """Open the pipe

        Args:
            name: name of the pipe
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear the pipe"""
        pass

    @abstractmethod
    def send(self, msg: Message, timeout=None) -> bool:
        """Send the specified message to the peer.

        Args:
            msg: the message to be sent
            timeout: if specified, number of secs to wait for the peer to read the message.

        Returns: whether the message is read by the peer.
        If timeout is not specified, always return False.

        """
        pass

    @abstractmethod
    def receive(self, timeout=None) -> Union[None, Message]:
        """Try to receive message from peer.

        Args:
            timeout: how long (number of seconds) to try

        Returns: the message received; or None if no message

        """
        pass

    @abstractmethod
    def close(self):
        """Close the pipe

        Returns: None

        """
        pass
