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

from nvflare.private.admin_defs import Message
from nvflare.private.fed.protos.admin_pb2 import Message as Proto_Message


def message_to_proto(message: Message) -> Proto_Message:
    proto_message = Proto_Message()
    proto_message.id = message.id
    proto_message.topic = message.topic
    if isinstance(message.body, str):
        proto_message.body_type = "str"
        proto_message.body = bytes(message.body, "utf-8")
    elif isinstance(message.body, bytes):
        proto_message.body_type = "bytes"
        proto_message.body = message.body
    else:
        proto_message.body_type = "unknown"
        proto_message.body = message.body

    for k, v in message.headers.items():
        proto_message.headers[k] = v
    return proto_message


def proto_to_message(proto: Proto_Message) -> Message:
    if proto.body_type == "str":
        message = Message(topic=proto.topic, body=proto.body.decode("utf-8"))
    elif proto.body_type == "bytes":
        message = Message(topic=proto.topic, body=proto.body)
    else:
        message = Message(topic=proto.topic, body=proto.body)

    message.id = proto.id
    for k, v in proto.headers.items():
        message.headers[k] = v
    return message
