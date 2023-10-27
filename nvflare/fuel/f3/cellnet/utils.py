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
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey

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
}


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


def shorten_fqcn(fqcn):
    parts = fqcn.split(".")
    s_fqcn = ".".join([shorten_string(p) for p in parts])
    return s_fqcn


def get_msg_header_value(m, k):
    return m.get_header(k, "?")


def format_log_message(fqcn: str, message: Message, log: str) -> str:
    context = [f"[ME={shorten_fqcn(fqcn)}"]
    for k, v in cell_mapping.items():
        string = f"{k}=" + shorten_fqcn(get_msg_header_value(message, v))
        context.append(string)
    for k, v in msg_mapping.items():
        string = f"{k}=" + get_msg_header_value(message, v)
        context.append(string)
    return " ".join(context) + f"] {log}"


def encode_payload(message: Message, encoding_key=MessageHeaderKey.PAYLOAD_ENCODING):
    encoding = message.get_header(encoding_key)
    if not encoding:
        if message.payload is None:
            encoding = Encoding.NONE
        elif isinstance(message.payload, (bytes, bytearray, memoryview)):
            encoding = Encoding.BYTES
        else:
            encoding = Encoding.FOBS
            message.payload = fobs.dumps(message.payload, buffer_list=True)
        message.set_header(encoding_key, encoding)


def decode_payload(message: Message, encoding_key=MessageHeaderKey.PAYLOAD_ENCODING):
    encoding = message.get_header(encoding_key)
    if not encoding:
        return

    if encoding == Encoding.FOBS:
        message.payload = fobs.loads(message.payload)
    elif encoding == Encoding.NONE:
        message.payload = None
    else:
        # assume to be bytes
        pass
    message.remove_header(encoding_key)
