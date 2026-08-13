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
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.fed.server.server_command_agent import ServerCommandAgent


class TestAuxCommunicateAuthCheck:
    """Verify that aux_communicate returns early when authentication fails."""

    def test_dispatch_not_called_on_auth_failure(self):
        mock_engine = MagicMock()
        mock_fl_ctx = MagicMock()
        mock_engine.new_context.return_value.__enter__ = MagicMock(return_value=mock_fl_ctx)
        mock_engine.new_context.return_value.__exit__ = MagicMock(return_value=False)
        mock_fl_ctx.get_engine.return_value = mock_engine

        # authentication_check returns an error string
        mock_engine.server.server_state.aux_communicate.return_value = "state_ok"
        mock_engine.server.authentication_check.return_value = "auth_failed"

        agent = ServerCommandAgent.__new__(ServerCommandAgent)
        agent.engine = mock_engine

        request = CellMessage()
        request.payload = {"test": "data"}
        request.set_header(MessageHeaderKey.TOPIC, "test_topic")
        result = agent.aux_communicate(request)

        assert not mock_engine.dispatch.called, "engine.dispatch should NOT be called when authentication fails"
        assert result is not None, "aux_communicate should return an error reply, not None"
        assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR


def test_server_job_rejects_internal_command_from_untrusted_endpoint():
    cell = MagicMock()
    cell.get_fqcn.return_value = "server.job-1"
    agent = ServerCommandAgent(MagicMock(), cell)
    request = CellMessage(
        {
            MessageHeaderKey.ORIGIN: "server",
            MessageHeaderKey.DESTINATION: "server.job-1",
            MessageHeaderKey.TOPIC: AdminCommandNames.CONFIGURE_JOB_LOG,
        },
        "INFO",
    )
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="site-1"))
    command = MagicMock()

    with patch("nvflare.private.fed.server.server_command_agent.ServerCommands.get_command", return_value=command):
        reply = agent.execute_command(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR
    command.process.assert_not_called()
