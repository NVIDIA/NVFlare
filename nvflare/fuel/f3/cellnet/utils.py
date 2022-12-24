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

from nvflare.fuel.f3.message import Message, Headers
from .defs import MessageHeaderKey


def make_reply(rc: str, error: str = "", body=None) -> Message:
    headers = Headers()
    headers[MessageHeaderKey.RETURN_CODE] = rc
    if error:
        headers[MessageHeaderKey.ERROR] = error
    return Message(headers, payload=body)


def new_message(headers: dict = None, payload=None):
    msg_headers = Headers()
    if headers:
        msg_headers.update(headers)
    return Message(msg_headers, payload)


def format_log_message(fqcn: str, message: Message, log: str) -> str:
    parts = [
        "[ME=" + fqcn,
        "O=" + message.get_header(MessageHeaderKey.ORIGIN, "?"),
        "D=" + message.get_header(MessageHeaderKey.DESTINATION, "?"),
        "F=" + message.get_header(MessageHeaderKey.FROM_CELL, "?"),
        "T=" + message.get_header(MessageHeaderKey.TO_CELL, "?") + "]",
        log
    ]
    return " ".join(parts)
