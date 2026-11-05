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

from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.core_cell import CoreCell, TargetMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message, ok_reply
from nvflare.private.defs import CellChannel, RequestHeader, SysCommandTopic, TrainingTopic, new_cell_message
from nvflare.private.fed.client.admin import FedAdminAgent, RequestProcessor


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(AuditService, "get_auditor", lambda: Mock(spec=Auditor))
    cell = Mock()
    cell.get_fqcn.return_value = "relay.site-2"
    engine = MagicMock()
    engine.new_context.return_value.__enter__.return_value = FLContext()
    agent = FedAdminAgent("site-2", cell, engine)
    processor = Mock(spec=RequestProcessor)
    processor.get_topics.return_value = [SysCommandTopic.SHELL, TrainingTopic.NOTIFY_JOB_STATUS]
    processor.process.return_value = ok_reply(body="processed")
    agent.register_processor(processor)
    return agent, processor


@pytest.mark.parametrize(
    "origin,endpoint,topic,job_id,allowed",
    [
        ("server", "relay.site-1", SysCommandTopic.SHELL, None, False),
        ("server", None, SysCommandTopic.SHELL, None, False),
        ("server.job", "relay", SysCommandTopic.SHELL, None, False),
        ("relay.site-1", "relay", SysCommandTopic.SHELL, None, False),
        ("relay.site-2.job", "relay.site-2.job", SysCommandTopic.SHELL, "job", False),
        ("relay.site-2.job", "relay.site-2.job", TrainingTopic.NOTIFY_JOB_STATUS, "job", True),
        ("relay.site-2.job", "relay", TrainingTopic.NOTIFY_JOB_STATUS, "job", False),
        ("relay.site-1.job", "relay.site-1.job", TrainingTopic.NOTIFY_JOB_STATUS, "job", False),
        ("relay.site-2.job", "relay.site-2.job", TrainingTopic.NOTIFY_JOB_STATUS, "other", False),
    ],
)
def test_only_server_commands_and_own_child_status_are_dispatched(agent, origin, endpoint, topic, job_id, allowed):
    agent, processor = agent
    payload = Message(topic, "command")
    payload.set_header(RequestHeader.JOB_ID, job_id)
    request = new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload)
    if endpoint:
        request.set_prop(MessagePropKey.ENDPOINT, Endpoint(endpoint))

    reply = agent._dispatch_request(request)

    if allowed:
        assert reply.payload.body == "processed"
        processor.process.assert_called_once()
    else:
        assert reply.payload.body == "request must originate from server"
        processor.process.assert_not_called()
        agent.app_ctx.new_context.assert_not_called()


@pytest.mark.parametrize("origin,wire_endpoint", [("relay.site-1", None), ("server", None), ("server", "server")])
def test_bulk_admin_headers_cannot_supply_trusted_endpoint(agent, origin, wire_endpoint):
    agent, processor = agent
    payload = new_cell_message({}, Message(SysCommandTopic.SHELL, "command"))
    item = TargetMessage("relay.site-2", CellChannel.CLIENT_MAIN, SysCommandTopic.SHELL, payload)
    headers = {MessageHeaderKey.ORIGIN: origin, MessageHeaderKey.FROM_CELL: "server"}
    if wire_endpoint:
        headers[MessagePropKey.ENDPOINT] = wire_endpoint
    bulk = new_cell_message(headers, [item.to_dict()])
    bulk.set_prop(MessagePropKey.ENDPOINT, Endpoint("relay"))
    cell = Mock()

    CoreCell._process_one_bulk(cell, bulk)

    cell._process_request.assert_called_once()
    request = cell._process_request.call_args.kwargs["message"]
    assert request.get_prop(MessagePropKey.ENDPOINT) is None
    reply = agent._dispatch_request(request)
    assert reply.payload.body == "request must originate from server"
    processor.process.assert_not_called()
    agent.app_ctx.new_context.assert_not_called()
