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

from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.hci.client.api import AdminAPI, ResultKey, _LoginReplyProcessor, _ServerReplyJsonProcessor
from nvflare.fuel.hci.client.api_spec import AdminConfigKey, CommandContext
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import ProtoKey
from nvflare.tool import cli_output


@pytest.mark.parametrize("output_format", ["txt", "json", "jsonl"])
def test_admin_api_passes_server_identity_without_logging_registration_token(capsys, monkeypatch, output_format):
    monkeypatch.setattr(cli_output, "_output_format", output_format)
    cell_kwargs = {}
    authenticator_kwargs = {}

    class _FakeCell:
        def __init__(self, **kwargs):
            cell_kwargs.update(kwargs)
            self.core_cell = Mock()

        def register_request_cb(self, **kwargs):
            pass

        def get_fqcn(self):
            return "admin@nvidia.com"

        def start(self):
            pass

    class _FakeTokenVerifier:
        pass

    class _FakeAuthenticator:
        def __init__(self, **kwargs):
            authenticator_kwargs.update(kwargs)

        def authenticate(self, shared_fl_ctx, abort_signal):
            return "registration-secret", "registration-signature", "ssid", _FakeTokenVerifier()

    admin_config = {
        AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
        AdminConfigKey.PROJECT_NAME: "project",
        AdminConfigKey.SERVER_IDENTITY: "gcp-server",
        AdminConfigKey.CONNECTION_SCHEME: "tcp",
        AdminConfigKey.CA_CERT: "rootCA.pem",
        AdminConfigKey.CLIENT_CERT: "client.crt",
        AdminConfigKey.CLIENT_KEY: "client.key",
        AdminConfigKey.HOST: "35.192.226.99",
        AdminConfigKey.PORT: 8003,
    }
    api = AdminAPI(user_name="admin@nvidia.com", admin_config=admin_config, cmd_modules=[], debug=True)

    with (
        patch("nvflare.fuel.hci.client.api.Authenticator", _FakeAuthenticator),
        patch("nvflare.fuel.hci.client.api.AuxRunner", Mock()),
        patch("nvflare.fuel.hci.client.api.Cell", _FakeCell),
        patch("nvflare.fuel.hci.client.api.NetAgent", Mock()),
        patch("nvflare.fuel.hci.client.api.ObjectStreamer", Mock()),
        patch("nvflare.fuel.hci.client.api.TokenVerifier", _FakeTokenVerifier),
        patch("nvflare.fuel.hci.client.api.set_add_auth_headers_filters") as add_filters,
    ):
        api.connect()

    assert cell_kwargs["auth_identity_map"] == {FQCN.ROOT_SERVER: "gcp-server"}
    assert authenticator_kwargs["expected_sp_identity"] == "gcp-server"
    add_filters.assert_called_once_with(
        api.cell, api.user_name, "registration-secret", "registration-signature", "ssid"
    )
    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    assert "Successfully authenticated to gcp-server" in diagnostic
    assert "registration-secret" not in diagnostic
    assert "registration-signature" not in diagnostic


@pytest.mark.parametrize("output_format", ["txt", "json", "jsonl"])
@pytest.mark.parametrize("debug", [False, True])
def test_login_reply_redacts_diagnostics_without_changing_session_token(capsys, monkeypatch, output_format, debug):
    monkeypatch.setattr(cli_output, "_output_format", output_format)
    api = AdminAPI(
        user_name="admin@nvidia.com",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CLIENT_CERT: "client.crt",
            AdminConfigKey.CLIENT_KEY: "client.key",
        },
        cmd_modules=[],
        debug=debug,
    )
    ctx = CommandContext()
    ctx.set_api(api)
    ctx.set_reply_processor(_LoginReplyProcessor())
    response = {
        ProtoKey.DATA: [
            {ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "OK"},
            {ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "signed-session-secret"},
        ],
        ProtoKey.META: {
            "registration_token": "registration-secret",
            "diagnostic": "-----BEGIN PRIVATE KEY-----\nprivate-key-secret\n-----END PRIVATE KEY-----",
        },
    }

    _ServerReplyJsonProcessor(ctx).process_server_reply(response)

    assert api.token == "signed-session-secret"
    assert api.login_result == "OK"
    assert ctx.get_command_result() is response
    assert response[ProtoKey.DATA][1][ProtoKey.DATA] == "signed-session-secret"
    assert response[ProtoKey.META]["registration_token"] == "registration-secret"
    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    for secret in ("signed-session-secret", "registration-secret", "private-key-secret", "BEGIN PRIVATE KEY"):
        assert secret not in diagnostic
    if debug:
        assert "Server Reply:" in diagnostic
        assert "OK" in diagnostic
        assert "<redacted>" in diagnostic
    else:
        assert diagnostic == ""


@pytest.mark.parametrize("status", [APIStatus.SUCCESS, APIStatus.ERROR_AUTHENTICATION])
def test_login_logs_status_without_dumping_result(capsys, monkeypatch, status):
    api = AdminAPI(
        user_name="admin@nvidia.com",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CLIENT_CERT: "client.crt",
            AdminConfigKey.CLIENT_KEY: "client.key",
        },
        cmd_modules=[],
        debug=True,
    )
    result = {
        ResultKey.STATUS: status,
        ResultKey.DETAILS: [{ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "signed-session-secret"}],
    }
    monkeypatch.setattr(api, "_try_login", lambda: result)

    assert api.login() is result

    captured = capsys.readouterr()
    diagnostic = captured.out + captured.err
    assert f"login status: {status}" in diagnostic
    assert "signed-session-secret" not in diagnostic
