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

from nvflare.fuel.utils.pipe.pipe import Message


def message_to_file_name(msg: Message) -> str:
    """Produce the file name that encodes the meta info of the message

    Args:
        msg: message for which the file name is to be produced

    Returns:

    """
    if msg.msg_type == Message.REQUEST:
        return f"{msg.msg_type}.{msg.topic}.{msg.msg_id}"
    elif msg.msg_type == Message.REPLY:
        return f"{msg.msg_type}.{msg.topic}.{msg.req_id}.{msg.msg_id}"
    else:
        raise ValueError(f"invalid message type '{msg.msg_type}'")


def file_name_to_message(file_name: str) -> Message:
    """Decode the file name to produce the meta info of the message.

    Args:
        file_name: the file name to be decoded.

    Returns: a Message object that contains meta info.

    """
    parts = file_name.split(".")
    num_parts = len(parts)
    if num_parts < 3 or num_parts > 4:
        raise ValueError(f"bad file name: {file_name} - wrong number of parts {num_parts}")
    msg_type = parts[0]
    topic = parts[1]
    msg_id = parts[-1]
    data = None
    if msg_type == Message.REQUEST:
        if num_parts != 3:
            raise ValueError(f"bad file name for {msg_type}: {file_name} - must be 3 parts but got {num_parts}")
        return Message.new_request(topic, data, msg_id)
    elif msg_type == Message.REPLY:
        if num_parts != 4:
            raise ValueError(f"bad file name for {msg_type}: {file_name} - must be 4 parts but got {num_parts}")
        req_id = parts[2]
        return Message.new_reply(topic, data, req_id, msg_id)
    else:
        raise ValueError(f"bad file name: {file_name} - invalid msg type '{msg_type}'")
