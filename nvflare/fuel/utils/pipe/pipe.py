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
from typing import Any, Tuple, Union

from nvflare.fuel.utils.attributes_exportable import AttributesExportable, ExportMode
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.validation_utils import check_str


class Topic(object):

    ABORT = "_ABORT_"
    END = "_END_"
    HEARTBEAT = "_HEARTBEAT_"
    PEER_GONE = "_PEER_GONE_"


class Message:

    REQUEST = "REQ"
    REPLY = "REP"

    def __init__(self, msg_type: str, topic: str, data: Any, msg_id=None, req_id=None):
        check_str("msg_type", msg_type)
        if msg_type not in [Message.REPLY, Message.REQUEST]:
            raise ValueError(f"invalid msg_type '{msg_type}': must be one of {[Message.REPLY, Message.REQUEST]}")
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
        """Creates a new request message.

        This static method creates a new `Message` object representing a request.

        Args:
            topic (str): The topic of the request message.
                This is a string that identifies the subject or category of the request.
            data (Any): The data associated with the request message.
                This can be any type of data that is relevant to the request.
            msg_id (Optional[Any]): An optional identifier for the message.
                If provided, this ID is used to uniquely identify the message.
                If not provided, a UUID will be generated to uniquely identify the message.

        Returns:
            Message: A `Message` object with the type set to `Message.REQUEST`,
                and the provided `topic`, `data`, and `msg_id`.

        """
        return Message(Message.REQUEST, topic, data, msg_id)

    @staticmethod
    def new_reply(topic: str, data: Any, req_msg_id, msg_id=None):
        """Creates a new reply message in response to a request.

        This static method creates a new `Message` object representing a reply to a previous request.

        Args:
            topic (str): The topic of the reply message.
                This is a string that identifies the subject or category of the reply.
            data (Any): The data associated with the reply message. This can be any type of data relevant to the reply.
            req_msg_id (int): The identifier of the request message that this reply is responding to.
                This ID links the reply to the original request.
            msg_id (Optional[int]): An optional identifier for the reply message.
                If provided, this ID is used to uniquely identify the message.
                If not provided, a UUID will be generated to uniquely identify the message.

        Returns:
            Message: A `Message` object with the type set to `Message.REPLY`,
                and the provided `topic`, `data`, `msg_id`, and `req_msg_id`.
        """
        return Message(Message.REPLY, topic, data, msg_id, req_id=req_msg_id)

    def __str__(self):
        return f"Message(topic={self.topic}, msg_id={self.msg_id}, req_id={self.req_id}, msg_type={self.msg_type})"


class Pipe(AttributesExportable, ABC):
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
        """Sends the specified message to the peer.

        Args:
            msg: the message to be sent
            timeout: if specified, number of secs to wait for the peer to read the message.
                If not specified, wait indefinitely.

        Returns:
            Whether the message is read by the peer.

        """
        pass

    @abstractmethod
    def receive(self, timeout=None) -> Union[None, Message]:
        """Try to receive message from peer.

        Args:
            timeout: how long (number of seconds) to try
                If not specified, return right away.

        Returns:
            the message received; or None if no message

        """
        pass

    @abstractmethod
    def close(self):
        """Close the pipe

        Returns: None

        """
        pass

    @abstractmethod
    def can_resend(self) -> bool:
        """Whether the pipe is able to resend a message."""
        pass

    def get_last_peer_active_time(self):
        """Get the last time that the peer is known to be active

        Returns: the last time that the peer is known to be active; or 0 if this info is not available

        """
        return 0

    def export(self, export_mode: str) -> Tuple[str, dict]:
        if export_mode == ExportMode.SELF:
            mode = self.mode
        else:
            mode = Mode.ACTIVE if self.mode == Mode.PASSIVE else Mode.PASSIVE

        return f"{self.__module__}.{self.__class__.__name__}", {"mode": mode}
