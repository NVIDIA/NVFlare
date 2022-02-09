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

import enum

from .fl_context import FLContext
from .shareable import Shareable


class MessageSendStatus(enum.Enum):

    OK = "ok"  # message sent and response received
    TIMEOUT = "timeout"  # message sent but no response received
    FAILURE = "failure"  # failed to send message
    REPLY_ERROR = "reply_error"  # error in reply


def aux_request_handle_func_signature(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    """This is the signature of the message_handle_func.

    The message_handle_func is a callback function that is registered to handle an aux request of a specific topic.
    Any implementation of a message_handle_func must follow this signature.

    Example from the client runner:
        engine.register_aux_message_handler(topic=ReservedTopic.END_RUN, message_handle_func=self._handle_end_run)

    Args:
        topic: topic of the message to be handled
        request: the message data to be handled
        fl_ctx: FL context

    Returns: a Shareable response to the requester

    """
    pass
