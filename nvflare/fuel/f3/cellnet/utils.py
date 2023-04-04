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
import nvflare.fuel.utils.fobs as fobs
from nvflare.fuel.f3.cellnet.defs import Encoding, MessageHeaderKey
from nvflare.fuel.f3.message import Headers, Message


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
        "T=" + message.get_header(MessageHeaderKey.TO_CELL, "?"),
        "CH=" + message.get_header(MessageHeaderKey.CHANNEL, "?"),
        "TP=" + message.get_header(MessageHeaderKey.TOPIC, "?") + "]",
        log,
    ]
    return " ".join(parts)


def encode_payload(message: Message):
    encoding = message.get_header(MessageHeaderKey.PAYLOAD_ENCODING)
    if not encoding:
        if message.payload is None:
            encoding = Encoding.NONE
        elif isinstance(message.payload, bytes) or isinstance(message.payload, bytearray):
            encoding = Encoding.BYTES
        else:
            encoding = Encoding.FOBS
            message.payload = fobs.dumps(message.payload)
        message.set_header(MessageHeaderKey.PAYLOAD_ENCODING, encoding)


def decode_payload(message: Message):
    encoding = message.get_header(MessageHeaderKey.PAYLOAD_ENCODING)
    if not encoding:
        return

    if encoding == Encoding.FOBS:
        message.payload = fobs.loads(message.payload)
    elif encoding == Encoding.NONE:
        message.payload = None
    else:
        # assume to be bytes
        pass
    message.remove_header(MessageHeaderKey.PAYLOAD_ENCODING)
