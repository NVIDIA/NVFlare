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

import threading
from types import SimpleNamespace

import pytest

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.client.api import AdminAPI, ResultKey, _AdminCellReconnectMonitor
from nvflare.fuel.hci.client.api_spec import CommandContext
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.reg import CommandRegister
from nvflare.fuel.hci.table import Table


@pytest.mark.parametrize("reconnected", [False, True])
@pytest.mark.parametrize("login_ok", [False, True])
def test_command_recovers_reconnected_admin_before_sending(reconnected, login_ok):
    api = AdminAPI.__new__(AdminAPI)
    api.in_logout = False
    api._cell_reconnect_lock = threading.RLock()
    api._cell_recovery_needed = False
    api._cell_recovery_in_progress = False
    api._cell_auth_monitor = _AdminCellReconnectMonitor()
    if reconnected:
        api._cell_auth_monitor.reconnected.set()
    api.debug = lambda message: None
    commands = []
    logins = []

    def new_context(command, args, entry):
        return CommandContext()

    def send(ctx):
        commands.append(ctx.get_command())
        ctx.set_command_result({ResultKey.STATUS: APIStatus.SUCCESS})

    def login():
        logins.append(True)
        # Model _user_login replacing the retired Cell before sending CERT_LOGIN.
        api._cell_auth_monitor = _AdminCellReconnectMonitor()
        if login_ok:
            return api.server_execute("_cert_login admin")
        return {ResultKey.STATUS: APIStatus.ERROR_AUTHENTICATION}

    api._new_command_context = new_context
    api._try_command = send
    api._user_login = login
    result = api.server_execute("check_status server")

    assert logins == ([True] if reconnected else [])
    if reconnected and not login_ok:
        assert result[ResultKey.STATUS] == APIStatus.ERROR_AUTHENTICATION
        assert commands == []
    else:
        assert result[ResultKey.STATUS] == APIStatus.SUCCESS
        assert commands == (["_cert_login admin"] if reconnected else []) + ["check_status server"]


def test_do_client_command_preserves_command_props():
    api = AdminAPI.__new__(AdminAPI)
    captured = {}

    def _new_command_context(command, args, ent):
        ctx = CommandContext()
        ctx.set_command(command)
        ctx.set_command_args(args)
        ctx.set_command_entry(ent)
        return ctx

    def _handler(args, ctx):
        captured["props"] = ctx.get_command_props()
        ctx.set_command_result({"status": "ok"})

    api._new_command_context = _new_command_context
    ent = SimpleNamespace(handler=_handler)

    result = api._do_client_command("submit_job hello", ["submit_job", "hello"], ent, props={"study": "study-a"})

    assert result == {"status": "ok"}
    assert captured["props"] == {"study": "study-a"}


def test_user_login_sends_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api._cell_reconnect_lock = threading.RLock()
    api._cell_recovery_in_progress = False
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = "cancer-research"
    api.login_result = None
    api.cell = object()
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            assert private_key_file == "client.key"
            assert cert_file == "client.crt"

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["command"] = command
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    result = api._user_login()

    assert result == {"status": "ok"}
    assert captured["command"] == "_cert_login admin@nvidia.com"
    assert captured["headers"]["study"] == "cancer-research"


def test_user_login_defaults_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api._cell_reconnect_lock = threading.RLock()
    api._cell_recovery_in_progress = False
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = DEFAULT_STUDY
    api.login_result = None
    api.cell = object()
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            pass

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    api._user_login()

    assert captured["headers"]["study"] == DEFAULT_STUDY


def test_user_login_parses_structured_reject_code(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api._cell_reconnect_lock = threading.RLock()
    api._cell_recovery_in_progress = False
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = "cancer-research"
    api.login_result = None
    api.cell = object()

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            pass

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        api.login_result = (
            "REJECT: AUTH_STUDY_USER_NOT_MAPPED: user 'admin@nvidia.com' is not mapped to study 'cancer-research'"
        )

    api.server_execute = _fake_server_execute

    result = api._user_login()

    assert result["status"] == APIStatus.ERROR_AUTHENTICATION
    assert result["details"] == "user 'admin@nvidia.com' is not mapped to study 'cancer-research'"
    assert result[ResultKey.AUTH_CODE] == "AUTH_STUDY_USER_NOT_MAPPED"


@pytest.mark.parametrize("failure", ["cert-login", "command-list", "exception"])
def test_ordinary_command_retries_failed_recovery_with_a_new_unset_monitor(monkeypatch, failure):
    api = AdminAPI.__new__(AdminAPI)
    api.in_logout = False
    api.server_sess_active = True
    api._cell_reconnect_lock = threading.RLock()
    api._cell_recovery_needed = False
    api._cell_recovery_in_progress = False
    api._cell_auth_monitor = _AdminCellReconnectMonitor()
    api._cell_auth_monitor.reconnected.set()
    api.cell = object()
    api.ensure_client_cert_valid = lambda: None
    api.client_key, api.client_cert, api.user_name, api.study = "key", "cert", "admin", DEFAULT_STUDY
    api.debug = lambda message: None
    api.server_cmd_reg = initial_registry = CommandRegister(app_ctx=api)
    initial_registry.add_command("server", "check_status", "", "", None, None, True, "")
    initial_registry.finalize()
    api.client_cmd_reg = CommandRegister(app_ctx=api)
    api.cmd_modules = []
    api.all_cmds = []
    attempts, commands = [], []

    def reset():
        api.cell = None
        api._cell_auth_monitor = None
        api.server_sess_active = False
        api.token = None

    def connect():
        api.cell = object()
        api._cell_auth_monitor = _AdminCellReconnectMonitor()

    def send(ctx):
        command = ctx.get_command()
        if command.startswith("_cert_login"):
            attempts.append(command)
            if len(attempts) == 1 and failure == "exception":
                raise ConnectionError("transport unavailable during login")
            if len(attempts) == 1 and failure == "cert-login":
                return  # CERT_LOGIN timed out after Cell registration succeeded.
            api.login_result = "OK"
        elif command == InternalCommands.GET_CMD_LIST:
            if len(attempts) == 1 and failure == "command-list":
                return
            table = Table(["scope", "name", "description", "usage", "confirm"])
            table.add_row(["server", "check_status", "", "", ""])
            ctx.get_reply_processor().process_table(ctx, table)
        else:
            commands.append(command)
        ctx.set_command_result({ResultKey.STATUS: APIStatus.SUCCESS})

    api._reset_cell, api.connect, api._try_command = reset, connect, send
    monkeypatch.setattr(
        "nvflare.fuel.hci.client.api.IdentityAsserter",
        lambda **kwargs: SimpleNamespace(cert_data="cert", sign_common_name=lambda nonce: "signature"),
    )
    if failure == "exception":
        with pytest.raises(ConnectionError):
            api.do_command("check_status server")
    else:
        assert api.do_command("check_status server")[ResultKey.STATUS] == APIStatus.ERROR_RUNTIME
    assert not api.server_sess_active
    assert not api._cell_auth_monitor.reconnected.is_set()
    assert api._cell_recovery_needed
    assert not commands
    assert api.server_cmd_reg is initial_registry

    assert api.do_command("check_status server")[ResultKey.STATUS] == APIStatus.SUCCESS
    assert api.server_cmd_reg is not initial_registry
    assert len(attempts) == 2
    assert commands == ["check_status server"]
    assert api.server_sess_active
    assert not api._cell_recovery_needed
