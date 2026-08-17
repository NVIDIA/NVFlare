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

from unittest.mock import MagicMock

from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.fed.client.command_agent import CommandAgent


def test_stopped_client_command_agent_rejects_requests_without_engine_access():
    agent = CommandAgent(MagicMock())
    agent.engine = MagicMock()
    agent.shutdown()
    request = CellMessage()

    execute_reply = agent.execute_command(request)
    aux_reply = agent.aux_communication(request)

    assert execute_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    assert aux_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    agent.engine.new_context.assert_not_called()
