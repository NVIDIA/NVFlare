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
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic, MessageHeaderKey, MessagePropKey, ReturnCode
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.admin_defs import Message
from nvflare.private.defs import CellMessageHeaderKeys, RequestHeader
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_commands import ConfigureJobLogCommand as ClientConfigureJobLogCommand
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


def test_harvested_server_bearer_cannot_reach_server_job_command_processor():
    root = FederatedServer.__new__(FederatedServer)
    root.logger = MagicMock()
    root.my_own_auth_client_name = "server"
    root.my_own_token = "secret-token"
    root.cell = MagicMock()
    root.cell.get_fqcn.return_value = "server"
    root._get_id_asserter = MagicMock(return_value=MagicMock(cert=MagicMock()))

    replay = CellMessage(
        {
            MessageHeaderKey.ORIGIN: "server",
            MessageHeaderKey.DESTINATION: "server.job-1",
            MessageHeaderKey.TOPIC: AdminCommandNames.ABORT,
            CellMessageHeaderKeys.CLIENT_NAME: "server",
            CellMessageHeaderKeys.TOKEN: "secret-token",
            CellMessageHeaderKeys.TOKEN_SIGNATURE: "harvested-signature",
        },
        Shareable(),
    )
    replay.set_prop(MessagePropKey.ENDPOINT, Endpoint("site-1"))

    command = MagicMock()
    command.process.return_value = Shareable()
    job_cell = MagicMock()
    job_cell.get_fqcn.return_value = "server.job-1"
    job_agent = ServerCommandAgent(MagicMock(), job_cell)

    with (
        patch("nvflare.private.fed.server.fed_server.validate_auth_headers", return_value=None),
        patch("nvflare.private.fed.server.server_command_agent.ServerCommands.get_command", return_value=command),
    ):
        root_reply = root._validate_auth_headers(replay)
        if root_reply is None:
            # Model the root's forwarding hop.  This is the condition that made
            # the vulnerable server-job check trust the replayed logical origin.
            replay.set_prop(MessagePropKey.ENDPOINT, Endpoint("server"))
            job_agent.execute_command(replay)

    assert root_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED
    command.process.assert_not_called()


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
