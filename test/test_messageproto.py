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
from nvflare.private.fed.utils.messageproto import message_to_proto, proto_to_message


class TestMessageProto:
    def test_message_proto_convert(self):
        message = Message(topic="topic", body="{'id': 100}")
        message.set_header("Content-Type", "application/json")
        message_proto = message_to_proto(message)
        new_message = proto_to_message(message_proto)
        assert new_message.__dict__ == message.__dict__
