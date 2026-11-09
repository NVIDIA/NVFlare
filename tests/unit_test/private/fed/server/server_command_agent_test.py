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

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.fed.server.server_command_agent import ServerCommandAgent
from nvflare.private.fed.server.server_commands import ServerCommands


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

        agent = ServerCommandAgent(engine=mock_engine, cell=MagicMock())

        request = CellMessage()
        request.payload = {"test": "data"}
        request.set_header(MessageHeaderKey.TOPIC, "test_topic")
        result = agent.aux_communicate(request)

        assert not mock_engine.dispatch.called, "engine.dispatch should NOT be called when authentication fails"
        assert result is not None, "aux_communicate should return an error reply, not None"
        assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR


def test_stopped_server_command_agent_rejects_requests_without_engine_access():
    engine = MagicMock()
    agent = ServerCommandAgent(engine=engine, cell=MagicMock())
    agent.shutdown()
    request = CellMessage()

    execute_reply = agent.execute_command(request)
    aux_reply = agent.aux_communicate(request)

    assert execute_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    assert aux_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    engine.new_context.assert_not_called()


@pytest.mark.parametrize(
    "origin,peer",
    [
        ("site-1.attacker", "server"),  # authenticated client traffic relayed by the server
        ("server", "site-1.attacker"),  # claimed server origin on a direct client connection
        ("server", None),
    ],
)
def test_control_commands_require_parent_origin_and_transport_peer(monkeypatch, origin, peer):
    engine = MagicMock()
    command = MagicMock()
    monkeypatch.setattr(ServerCommands, "get_command", lambda name: command)
    request = CellMessage({MessageHeaderKey.TOPIC: "abort", MessageHeaderKey.ORIGIN: origin}, Shareable())
    # A wire header with a property key is not receiver-owned transport provenance.
    request.set_header(MessagePropKey.ENDPOINT, SimpleNamespace(name="server"))
    if peer is not None:
        request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=peer))

    reply = ServerCommandAgent(engine, MagicMock()).execute_command(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR
    command.process.assert_not_called()
    engine.new_context.assert_not_called()


def test_parent_can_issue_status_command(monkeypatch):
    command = MagicMock()
    command.process.return_value = Shareable({"result": "parent command processed"})
    monkeypatch.setattr(ServerCommands, "get_command", lambda name: command)
    request = CellMessage({MessageHeaderKey.TOPIC: "show_stats", MessageHeaderKey.ORIGIN: "server"}, Shareable())
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="server"))

    reply = ServerCommandAgent(MagicMock(), MagicMock()).execute_command(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload["result"] == "parent command processed"
