# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.cellnet.utils import encode_payload
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey


def test_stream_dispatch_preserves_receiver_owned_peer_and_ignores_wire_imitation():
    peer = Endpoint("site-1.job")
    message = Message(
        {
            MessageHeaderKey.ORIGIN: "server",
            StreamHeaderKey.CHANNEL: "server_command",
            StreamHeaderKey.TOPIC: "abort",
            MessagePropKey.ENDPOINT: "server",  # sender-supplied header cannot set a local property
        },
        {"data": "harmless"},
    )
    message.set_prop(MessagePropKey.ENDPOINT, peer)
    encode_payload(message, StreamHeaderKey.PAYLOAD_ENCODING)
    cell = MagicMock()
    cell.decode_pass_through_channels = set()
    cell.decode_pass_through_topics = set()
    cell.get_fobs_context.return_value = {}
    task = RxTask(17, "server", cell)
    task._handle_new_stream(message)
    task.stream_future.set_result(message.payload)
    received = []

    Adapter(received.append, SimpleNamespace(fqcn="server.victim"), cell).call(task.stream_future)

    assert len(received) == 1
    assert received[0].get_prop(MessagePropKey.ENDPOINT) is peer
    assert received[0].payload == {"data": "harmless"}
