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

import pytest

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic, MessageHeaderKey, MessagePropKey, ReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.admin_defs import Message, ok_reply
from nvflare.private.defs import CellMessageHeaderKeys, RequestHeader, TrainingTopic
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_commands import AdminCommands
from nvflare.private.fed.client.admin_commands import ConfigureJobLogCommand as ClientConfigureJobLogCommand
from nvflare.private.fed.client.command_agent import CommandAgent
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_command_agent import ServerCommandAgent
from nvflare.private.fed.server.server_commands import ConfigureJobLogCommand as ServerConfigureJobLogCommand


def test_cp_admin_dispatch_rejects_untrusted_origin_before_processor_runs():
    agent = FedAdminAgent.__new__(FedAdminAgent)
    agent.cell = MagicMock()
    agent.cell.get_fqcn.return_value = "site-1"
    processor = MagicMock()
    agent.processors = {"admin-topic": processor}
    inner = Message("admin-topic", "payload")
    inner.set_header(RequestHeader.ADMIN_COMMAND, "admin-command")
    inner.set_header(RequestHeader.REQUIRE_AUTHZ, "true")
    request = CellMessage(
        {
            MessageHeaderKey.ORIGIN: "attacker",
            MessageHeaderKey.DESTINATION: "site-1",
            MessageHeaderKey.TOPIC: "admin-topic",
        },
        inner,
    )
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="attacker"))

    reply = agent._dispatch_request(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
    processor.process.assert_not_called()


def test_cp_admin_dispatch_requires_server_created_envelope():
    agent = FedAdminAgent.__new__(FedAdminAgent)
    agent.cell = MagicMock()
    agent.cell.get_fqcn.return_value = "site-1"
    agent.processors = {"admin-topic": MagicMock()}
    request = CellMessage(
        {
            MessageHeaderKey.ORIGIN: "server",
            MessageHeaderKey.DESTINATION: "site-1",
            MessageHeaderKey.TOPIC: "admin-topic",
        },
        Message("admin-topic", "payload"),
    )
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="server"))

    reply = agent._dispatch_request(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST


def _cp_job_status_agent():
    agent = FedAdminAgent.__new__(FedAdminAgent)
    agent.cell = MagicMock()
    agent.cell.get_fqcn.return_value = "site-1"
    agent.auditor = None
    agent.app_ctx = MagicMock()
    processor = MagicMock()
    processor.process.return_value = ok_reply()
    agent.processors = {TrainingTopic.NOTIFY_JOB_STATUS: processor}
    return agent, processor


def _job_status_request(origin="site-1.job-1", destination="site-1", endpoint="site-1.job-1", job_id="job-1"):
    inner = Message(TrainingTopic.NOTIFY_JOB_STATUS, "")
    inner.set_header(RequestHeader.JOB_ID, job_id)
    inner.set_header(RequestHeader.JOB_STATUS, "started")
    request = CellMessage(
        {
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.DESTINATION: destination,
            MessageHeaderKey.TOPIC: TrainingTopic.NOTIFY_JOB_STATUS,
        },
        inner,
    )
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=endpoint))
    return request


def test_cp_dispatch_accepts_status_from_direct_child_job():
    agent, processor = _cp_job_status_agent()
    request = _job_status_request()

    agent._dispatch_request(request)

    processor.process.assert_called_once_with(request.payload, agent.app_ctx)


@pytest.mark.parametrize(
    "origin,destination,endpoint,job_id",
    [
        ("site-1.job-1", "site-1", "attacker", "job-1"),
        ("site-1.job-1", "site-1", "site-1.job-1", "job-2"),
        ("site-2.job-1", "site-1", "site-2.job-1", "job-1"),
        ("site-1.job-1", "site-2", "site-1.job-1", "job-1"),
    ],
)
def test_cp_dispatch_rejects_spoofed_job_status(origin, destination, endpoint, job_id):
    agent, processor = _cp_job_status_agent()

    reply = agent._dispatch_request(_job_status_request(origin, destination, endpoint, job_id))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
    processor.process.assert_not_called()


def test_server_job_rejects_parent_command_from_untrusted_connection():
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
    request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name="attacker"))

    reply = agent.execute_command(request)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR


def test_challenge_reply_does_not_disclose_server_bearer_headers():
    server = FederatedServer.__new__(FederatedServer)
    server.logger = MagicMock()
    server.my_own_token_signature = ""
    server.my_own_auth_client_name = "server"
    server.my_own_token = "secret-token"
    server.sign_auth_token = MagicMock(return_value="secret-signature")
    reply = CellMessage(
        {
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_MAIN,
            MessageHeaderKey.TOPIC: CellChannelTopic.Challenge,
        }
    )

    server._add_auth_headers(reply)

    assert reply.get_header(CellMessageHeaderKeys.TOKEN) is None
    assert reply.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE) is None
    server.sign_auth_token.assert_not_called()


def test_cellnet_bye_reply_does_not_disclose_server_bearer_headers():
    # CELLNET/Bye is exempt from incoming auth (a direct-neighbor teardown ack),
    # so its reply must not leak the server's reusable bearer material either;
    # otherwise an uncredentialed peer can elicit it and replay to authenticate
    # as the server.
    server = FederatedServer.__new__(FederatedServer)
    server.logger = MagicMock()
    server.my_own_token_signature = ""
    server.my_own_auth_client_name = "server"
    server.my_own_token = "secret-token"
    server.sign_auth_token = MagicMock(return_value="secret-signature")
    reply = CellMessage(
        {
            MessageHeaderKey.CHANNEL: CellChannel.CELLNET,
            MessageHeaderKey.TOPIC: CellChannelTopic.Bye,
        }
    )

    server._add_auth_headers(reply)

    assert reply.get_header(CellMessageHeaderKeys.TOKEN) is None
    assert reply.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE) is None
    server.sign_auth_token.assert_not_called()


def test_authenticated_reply_still_receives_bearer_headers():
    # Guard against over-suppression: a normal (non-pre-auth) reply must still
    # carry the server's auth headers.
    server = FederatedServer.__new__(FederatedServer)
    server.logger = MagicMock()
    server.my_own_token_signature = "secret-signature"
    server.my_own_auth_client_name = "server"
    server.my_own_token = "secret-token"
    server.sign_auth_token = MagicMock(return_value="secret-signature")
    reply = CellMessage(
        {
            MessageHeaderKey.CHANNEL: CellChannel.SERVER_COMMAND,
            MessageHeaderKey.TOPIC: "some_topic",
        }
    )

    server._add_auth_headers(reply)

    assert reply.get_header(CellMessageHeaderKeys.TOKEN) == "secret-token"
    assert reply.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE) == "secret-signature"


def _make_command_agent(local_fqcn):
    fed_client = MagicMock()
    fed_client.cell.get_fqcn.return_value = local_fqcn
    return CommandAgent(fed_client)


def _client_command(origin, destination, topic=AdminCommandNames.ABORT):
    return CellMessage(
        {
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.DESTINATION: destination,
            MessageHeaderKey.TOPIC: topic,
        }
    )


def test_client_command_accepts_server_and_own_parent_origin():
    # Server-originated (relayed) and the job cell's own parent CP (self-management)
    # are the only legitimate CLIENT_COMMAND sources.
    agent = _make_command_agent("site-1.job-1")
    assert agent._is_authorized_command_sender(_client_command("server", "site-1.job-1")) is True
    assert agent._is_authorized_command_sender(_client_command("site-1", "site-1.job-1")) is True


def test_client_command_rejects_enrolled_peer_and_misrouted_origin():
    agent = _make_command_agent("site-1.job-1")
    # a different enrolled site addressing this job cell
    assert agent._is_authorized_command_sender(_client_command("site-2", "site-1.job-1")) is False
    # a message not actually addressed to this cell
    assert agent._is_authorized_command_sender(_client_command("server", "site-2.job-1")) is False


def test_execute_command_rejects_unauthorized_origin_without_dispatch():
    agent = _make_command_agent("site-1.job-1")
    with patch.object(AdminCommands, "get_command") as get_command:
        reply = agent.execute_command(_client_command("site-2", "site-1.job-1", topic=AdminCommandNames.ABORT))

    # the enrolled-peer command is rejected before any AdminCommand is dispatched
    get_command.assert_not_called()
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR


@pytest.mark.parametrize("origin", ["server", "site-1"])
def test_execute_command_still_dispatches_for_authorized_origin(origin):
    # Guard against over-blocking: legitimate server-relayed and own-parent commands
    # must still reach the AdminCommand handler.
    agent = _make_command_agent("site-1.job-1")
    agent.engine = MagicMock()
    command = MagicMock()  # process() returns a non-None MagicMock -> RC OK path
    with patch.object(AdminCommands, "get_command", return_value=command) as get_command:
        reply = agent.execute_command(_client_command(origin, "site-1.job-1", topic=AdminCommandNames.ABORT))

    get_command.assert_called_once_with(AdminCommandNames.ABORT)
    command.process.assert_called_once()
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK


def _log_command_context():
    workspace = MagicMock()
    workspace.get_run_dir.return_value = "/tmp/run"
    workspace.get_log_config_file_path.return_value = "/tmp/log.json"
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = "job-1"
    return fl_ctx


def test_server_job_log_command_rejects_executable_dict_config():
    with patch("nvflare.private.fed.server.server_commands.dynamic_log_config") as dynamic:
        error = ServerConfigureJobLogCommand().process({"()": "os.mkdir"}, _log_command_context())

    assert "configure_job_log only supports log levels and built-in log modes" in error
    dynamic.assert_not_called()


def test_client_job_log_command_rejects_executable_dict_config():
    with patch("nvflare.private.fed.client.admin_commands.dynamic_log_config") as dynamic:
        error = ClientConfigureJobLogCommand().process({"()": "os.mkdir"}, _log_command_context())

    assert "configure_job_log only supports log levels and built-in log modes" in error
    dynamic.assert_not_called()
