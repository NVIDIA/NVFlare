# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import flwr.proto.grpcadapter_pb2 as pb2

from nvflare.apis.shareable import Shareable

from .defs import Constant


def msg_container_to_shareable(msg: pb2.MessageContainer) -> Shareable:
    """Convert Flower-defined MessageContainer object to a Shareable object.
    This function is typically used to in two cases:
    1. Convert Flower-client generated request to Shareable before sending it to FLARE Server via RM.
    2. Convert Flower-server generated response to Shareable before sending it back to FLARE client.

    Args:
        msg: MessageContainer object to be converted

    Returns: a Shareable object

    """
    s = Shareable()
    headers = msg.metadata
    if headers is not None:
        # must convert msg.metadata to dict; otherwise it is not serializable.
        headers = dict(msg.metadata)
    s[Constant.PARAM_KEY_CONTENT] = msg.grpc_message_content
    s[Constant.PARAM_KEY_HEADERS] = headers
    s[Constant.PARAM_KEY_MSG_NAME] = msg.grpc_message_name
    return s


def shareable_to_msg_container(s: Shareable) -> pb2.MessageContainer:
    """Convert Shareable object to Flower-defined MessageContainer
    This function is typically used to in two cases:
    1. Convert a Shareable object received from FLARE client to MessageContainer before sending it to Flower server.
    2. Convert a Shareable object received from FLARE server to MessageContainer before sending it to Flower client.

    Args:
        s: the Shareable object to be converted

    Returns: a MessageContainer object

    """
    m = pb2.MessageContainer(
        grpc_message_name=s.get(Constant.PARAM_KEY_MSG_NAME),
        grpc_message_content=s.get(Constant.PARAM_KEY_CONTENT),
    )
    headers = s.get(Constant.PARAM_KEY_HEADERS)
    if headers:
        # Note: headers is a dict, but m.metadata is Google defined MapContainer, which is subclass of dict.
        m.metadata.update(headers)
    return m


def reply_should_exit() -> pb2.MessageContainer:
    return pb2.MessageContainer(metadata={"should-exit": "true"})
