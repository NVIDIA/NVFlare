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

import pytest

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.defs import MessagePropKey
from nvflare.fuel.f3.endpoint import Endpoint
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


@pytest.mark.parametrize(
    "origin,endpoint,allowed",
    [
        ("server", "relay.site-2", True),
        ("server", "server", True),
        ("relay.site-2", "server", False),
        ("relay.site-2", "relay.site-2", True),
        ("relay.site-1", "relay.site-2", False),
        ("relay.site-2.other-job", "relay.site-2", False),
        ("server", "relay.site-1", False),
        ("server", None, False),
    ],
)
def test_job_commands_require_server_or_own_parent(origin, endpoint, allowed):
    client = MagicMock()
    client.cell.get_fqcn.return_value = "relay.site-2.job"
    agent = CommandAgent(client)
    engine = MagicMock()
    engine.new_context.return_value.__enter__.return_value.get_engine.return_value = engine
    agent.engine = engine
    request = CellMessage(
        headers={
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.TOPIC: AdminCommandNames.RESET_ERRORS,
            # A wire header must never substitute for trusted transport metadata.
            MessagePropKey.ENDPOINT: "relay.site-2",
        },
        payload={},
    )
    if endpoint:
        request.set_prop(MessagePropKey.ENDPOINT, Endpoint(endpoint))

    reply = agent.execute_command(request)

    if allowed:
        engine.reset_errors.assert_called_once()
    else:
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
        engine.new_context.assert_not_called()
        engine.reset_errors.assert_not_called()
