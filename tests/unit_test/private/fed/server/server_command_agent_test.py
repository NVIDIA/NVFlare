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

from nvflare.apis.fl_constant import ServerCommandKey, ServerCommandNames
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.defs import CellMessageHeaderKeys, SpecialTaskName
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
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


def _run_get_task_command(task_name, monkeypatch, *, accounting=True):
    engine = MagicMock()
    engine.server.authentication_check.return_value = None
    command = MagicMock()
    command.get_state_check.return_value = {}
    task = Shareable()
    task.set_header(ServerCommandKey.TASK_NAME, task_name)
    command.process.return_value = task

    counter = F3Counter()
    cell = MagicMock()
    cell.get_fqcn.return_value = "server/job-1"
    agent = ServerCommandAgent(engine=engine, cell=cell)
    monkeypatch.setattr(agent, "_get_client", lambda _token: SimpleNamespace(name="site-1"))
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.ServerCommands.get_command",
        lambda _command_name: command,
    )
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.get_job_f3_counter",
        lambda: counter if accounting else None,
    )

    request = CellMessage(
        headers={
            MessageHeaderKey.TOPIC: ServerCommandNames.GET_TASK,
            MessageHeaderKey.ORIGIN: "site-1/job-1",
            CellMessageHeaderKeys.TOKEN: "token-1",
        },
        payload=Shareable(),
    )
    return agent.execute_command(request), counter


def test_real_task_response_is_classified_at_server_job_origin(monkeypatch):
    response, counter = _run_get_task_command("train", monkeypatch)

    context = response.get_logical_send_context()
    assert context is not None
    assert context._accounting is counter
    assert context._traffic_class is F3TrafficClass.TASK_RESPONSE
    assert context._recipient_name == "site-1"
    assert counter.pending_count == 1

    attempt = context.try_begin("server/job-1", "site-1/job-1")
    assert attempt is not None
    attempt.accepted(10)
    assert counter.freeze()["sent_to"] == [{"participant_name": "site-1", "payload_bytes": "10", "messages": "1"}]


def test_task_binding_receives_existing_request_and_registered_server_client(monkeypatch):
    observed = {}

    def resolver_factory(destination, participant_name, **kwargs):
        observed.update(destination=destination, participant_name=participant_name, **kwargs)
        return lambda _consumer, _request: None

    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.make_authenticated_consumer_resolver", resolver_factory
    )
    response, _counter = _run_get_task_command("train", monkeypatch)

    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert observed["destination"] == "site-1/job-1"
    assert observed["participant_name"] == observed["client"].name == "site-1"
    assert observed["server_job_fqcn"] == "server/job-1"
    assert observed["task_request"].get_header(CellMessageHeaderKeys.TOKEN) == "token-1"
    assert observed["server"] is not None


def test_inactive_accounting_does_not_verify_task_credentials(monkeypatch):
    factory = MagicMock(side_effect=AssertionError("inactive accounting must not inspect credentials"))
    monkeypatch.setattr("nvflare.private.fed.server.server_command_agent.make_authenticated_consumer_resolver", factory)
    response, _counter = _run_get_task_command("train", monkeypatch, accounting=False)

    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert response.get_logical_send_context() is None
    factory.assert_not_called()


def test_real_task_response_pre_admits_while_callback_is_active(monkeypatch):
    observed = {}

    def observe_attach(_message, _counter, _traffic_class, *, recipient_name, pre_admit=None, consumer_resolver=None):
        observed["callback_drained"] = agent.wait_for_callbacks(0.0)
        observed["recipient_name"] = recipient_name
        observed["pre_admit"] = pre_admit
        return True

    engine = MagicMock()
    engine.server.authentication_check.return_value = None
    command = MagicMock()
    command.get_state_check.return_value = {}
    task = Shareable()
    task.set_header(ServerCommandKey.TASK_NAME, "train")
    command.process.return_value = task
    cell = MagicMock()
    cell.get_fqcn.return_value = "server/job-1"
    agent = ServerCommandAgent(engine=engine, cell=cell)
    monkeypatch.setattr(agent, "_get_client", lambda _token: SimpleNamespace(name="site-1"))
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.ServerCommands.get_command",
        lambda _command_name: command,
    )
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.get_job_f3_counter",
        lambda: F3Counter(),
    )
    monkeypatch.setattr("nvflare.private.fed.server.server_command_agent.attach_f3_context", observe_attach)
    request = CellMessage(
        headers={
            MessageHeaderKey.TOPIC: ServerCommandNames.GET_TASK,
            MessageHeaderKey.ORIGIN: "site-1/job-1",
            CellMessageHeaderKeys.TOKEN: "token-1",
        },
        payload=Shareable(),
    )

    agent.execute_command(request)

    assert observed == {
        "callback_drained": False,
        "recipient_name": "site-1",
        "pre_admit": ("server/job-1", "site-1/job-1"),
    }


def test_task_response_accounting_identity_failure_does_not_change_reply(monkeypatch):
    engine = MagicMock()
    engine.server.authentication_check.return_value = None
    command = MagicMock()
    command.get_state_check.return_value = {}
    task = Shareable()
    task.set_header(ServerCommandKey.TASK_NAME, "train")
    command.process.return_value = task
    counter = F3Counter()
    cell = MagicMock()
    cell.get_fqcn.side_effect = RuntimeError("identity unavailable")
    agent = ServerCommandAgent(engine=engine, cell=cell)
    monkeypatch.setattr(agent, "_get_client", lambda _token: MagicMock())
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.ServerCommands.get_command",
        lambda _command_name: command,
    )
    monkeypatch.setattr(
        "nvflare.private.fed.server.server_command_agent.get_job_f3_counter",
        lambda: counter,
    )
    request = CellMessage(
        headers={
            MessageHeaderKey.TOPIC: ServerCommandNames.GET_TASK,
            MessageHeaderKey.ORIGIN: "site-1/job-1",
            CellMessageHeaderKeys.TOKEN: "token-1",
        },
        payload=Shareable(),
    )

    response = agent.execute_command(request)

    assert response.payload is task
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert counter.freeze()["issues"] == ["counter_gap"]


@pytest.mark.parametrize("task_name", [SpecialTaskName.TRY_AGAIN, SpecialTaskName.END_RUN, ""])
def test_non_task_get_task_responses_are_not_classified(task_name, monkeypatch):
    response, _counter = _run_get_task_command(task_name, monkeypatch)

    assert response.get_logical_send_context() is None
