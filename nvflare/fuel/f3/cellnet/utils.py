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
import copy
import time
from typing import Any

import nvflare.fuel.utils.fobs as fobs
from nvflare.fuel.f3.cellnet.defs import Encoding, MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils.buffer_list import BufferList
from nvflare.fuel.utils.time_utils import time_to_string

cell_mapping = {
    "O": MessageHeaderKey.ORIGIN,
    "D": MessageHeaderKey.DESTINATION,
    "F": MessageHeaderKey.FROM_CELL,
    "T": MessageHeaderKey.TO_CELL,
}

msg_mapping = {
    "CH": MessageHeaderKey.CHANNEL,
    "TP": MessageHeaderKey.TOPIC,
    "SCH": StreamHeaderKey.CHANNEL,
    "STP": StreamHeaderKey.TOPIC,
    "SEQ": StreamHeaderKey.SEQUENCE,
}


_MSG_SIZE_HW_THRESHOLD = 1024 * 1024 * 10


class MsgHighWaterInfo:

    def __init__(self, hw_type: str):
        self.hw_type = hw_type
        self.size = 0
        self.timestamp = None
        self.topic = None
        self.origin = None
        self.destination = None
        self.headers = None

    def update(self, size, msg: Message):
        if size <= self.size:
            return

        self.size = size
        self.timestamp = time.time()
        self.topic = msg.get_header(MessageHeaderKey.TOPIC)
        self.origin = msg.get_header(MessageHeaderKey.ORIGIN)
        self.destination = msg.get_header(MessageHeaderKey.DESTINATION)
        self.headers = copy.copy(msg.headers)

        if size > _MSG_SIZE_HW_THRESHOLD:
            info = f"from {self.origin} to {self.destination}: {size=} topic={self.topic} headers={self.headers}"
            print(f"{time_to_string(self.timestamp)}: {self.hw_type} Msg Size High Water: {info}")


# Some stats of msg size high water
_sent_hw_info = MsgHighWaterInfo("Sent")
_received_hw_info = MsgHighWaterInfo("Received")


def new_cell_message(headers: dict, payload=None):
    msg_headers = {}
    if headers:
        msg_headers.update(headers)
    return Message(msg_headers, payload)


def make_reply(rc: str, error: str = "", body=None) -> Message:
    headers = {MessageHeaderKey.RETURN_CODE: rc}
    if error:
        headers[MessageHeaderKey.ERROR] = error
    return Message(headers, payload=body)


def shorten_string(string):
    if len(string) > 8:
        ss = ":" + string[-7:]
    else:
        ss = string
    return ss


def buffer_len(buffer: Any):

    if not buffer:
        buf_len = 0
    elif isinstance(buffer, list):
        buf_len = BufferList(buffer).get_size()
    else:
        buf_len = len(buffer)

    return buf_len


def shorten_fqcn(fqcn):
    parts = fqcn.split(".")
    s_fqcn = ".".join([shorten_string(p) for p in parts])
    return s_fqcn


def get_msg_header_value(m, k):
    return m.get_header(k, "?")


def format_log_message(fqcn: str, message: Message, log: str) -> str:
    context = [f"[ME={shorten_fqcn(fqcn)}"]
    for k, v in cell_mapping.items():
        string = f"{k}={shorten_fqcn(get_msg_header_value(message, v))}"
        context.append(string)
    for k, v in msg_mapping.items():
        string = f"{k}={get_msg_header_value(message, v)}"
        context.append(string)
    return " ".join(context) + f"] {log}"


def encode_payload(message: Message, encoding_key=MessageHeaderKey.PAYLOAD_ENCODING, fobs_ctx: dict = None) -> int:
    """Encode the payload of the specified message.

    Args:
        message: the message to be encoded
        encoding_key: the key name of the encoding property in the message header. If the encoding property is not
        set in the message header, then it means that the message payload has not been encoded. If the property is
        already set, then the message payload is already encoded, and no processing is done.
        If encoding is needed, we will determine the encoding scheme based on the data type of the payload:
        - If the payload is None, encoding scheme is NONE
        - If the payload data type is like bytes, encoding scheme is BYTES
        - Otherwise, encoding scheme is FOBS, and the payload is serialized with FOBS.
        fobs_ctx: contextual info for decomposers

    Returns: the encoded payload size.

    """
    if isinstance(fobs_ctx, dict):
        fobs_ctx[fobs.FOBSContextKey.MESSAGE] = message

    encoding = message.get_header(encoding_key)
    if not encoding:
        if message.payload is None:
            encoding = Encoding.NONE
        elif isinstance(message.payload, (bytes, bytearray, memoryview)):
            encoding = Encoding.BYTES
        else:
            encoding = Encoding.FOBS
            message.payload = fobs.dumps(message.payload, buffer_list=True, fobs_ctx=fobs_ctx)
        message.set_header(encoding_key, encoding)

    size = buffer_len(message.payload)
    message.set_header(MessageHeaderKey.PAYLOAD_LEN, size)

    global _sent_hw_info
    _sent_hw_info.update(size, message)

    return size


def decode_payload(message: Message, encoding_key=MessageHeaderKey.PAYLOAD_ENCODING, fobs_ctx: dict = None):
    if isinstance(fobs_ctx, dict):
        fobs_ctx[fobs.FOBSContextKey.MESSAGE] = message

    size = buffer_len(message.payload)
    message.set_header(MessageHeaderKey.PAYLOAD_LEN, size)

    global _received_hw_info
    _received_hw_info.update(size, message)

    encoding = message.get_header(encoding_key)
    if not encoding:
        return

    if encoding == Encoding.FOBS:
        message.payload = fobs.loads(message.payload, fobs_ctx=fobs_ctx)
    elif encoding == Encoding.NONE:
        message.payload = None
    else:
        # assume to be bytes
        pass
    message.remove_header(encoding_key)


def format_size(size, binary=False):
    """Format size in human-readable formats like  KB, MB, KiB, MiB

    Args:
        size: Size in bytes
        binary: If binary, one K is 1024 bytes, otherwise 1000 bytes.

    Returns: Size in human-readable format (like 10MB, 100.2GiB etc)

    """

    if binary:
        kilo = 1024.0
        suffix = "iB"
    else:
        kilo = 1000.0
        suffix = "B"

    num = int(size)
    unit_found = False
    for unit in ("", "K", "M", "G", "T"):
        if abs(num) < kilo:
            unit_found = True
            break
        num /= kilo

    if not unit_found:
        unit = "P"

    if not unit:
        suffix = "B"

    num_str = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{num_str}{unit}{suffix}"
