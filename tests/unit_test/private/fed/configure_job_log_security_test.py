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

import logging
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.fuel.utils.log_utils import LogMode, dynamic_log_config
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.client.admin_commands import ConfigureJobLogCommand as ClientConfigureJobLogCommand
from nvflare.private.fed.client.sys_cmd import ConfigureSiteLogProcessor
from nvflare.private.fed.server.server_commands import ConfigureJobLogCommand as ServerConfigureJobLogCommand
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.sys_cmd import SystemCommandModule

COMMANDS = [
    (ServerConfigureJobLogCommand, "nvflare.private.fed.server.server_commands.dynamic_log_config"),
    (ClientConfigureJobLogCommand, "nvflare.private.fed.client.admin_commands.dynamic_log_config"),
]

SAFE_CONFIGS = ["DEBUG", "20", LogMode.CONCISE, LogMode.RELOAD]


def _log_command_context(tmp_path):
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(tmp_path / "run")
    workspace.get_log_config_file_path.return_value = str(tmp_path / "log_config.json")
    engine = MagicMock()
    engine.get_workspace.return_value = workspace
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = "job-1"
    return fl_ctx


def _site_log_context(tmp_path):
    workspace = MagicMock()
    workspace.get_root_dir.return_value = str(tmp_path)
    workspace.get_log_config_file_path.return_value = str(tmp_path / "log_config.json")
    fl_ctx = MagicMock()
    fl_ctx.get_identity_name.return_value = "site-1"
    fl_ctx.get_prop.return_value = workspace
    engine = MagicMock()
    engine.new_context.return_value = fl_ctx
    return engine


def test_remote_log_level_does_not_load_same_named_file(tmp_path, monkeypatch):
    (tmp_path / "DEBUG").write_text('{"version": 1}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    root_logger = MagicMock()
    root_logger.hasHandlers.return_value = True

    with patch("nvflare.fuel.utils.log_utils.logging.getLogger", return_value=root_logger):
        dynamic_log_config("DEBUG", str(tmp_path), "unused", allow_file_config=False)

    root_logger.setLevel.assert_called_once_with(logging.DEBUG)


def test_remote_reload_still_loads_workspace_config(tmp_path):
    reload_path = tmp_path / "log_config.json"
    reload_path.write_text('{"version": 1}', encoding="utf-8")

    with patch("nvflare.fuel.utils.log_utils.apply_log_config") as apply_config:
        dynamic_log_config(LogMode.RELOAD, str(tmp_path), str(reload_path), allow_file_config=False)

    apply_config.assert_called_once_with({"version": 1}, str(tmp_path))


@pytest.mark.parametrize(("command_class", "dynamic_path"), COMMANDS)
@pytest.mark.parametrize("config", SAFE_CONFIGS)
def test_remote_job_log_accepts_safe_controls(tmp_path, command_class, dynamic_path, config):
    with patch(dynamic_path) as dynamic:
        error = command_class().process(config, _log_command_context(tmp_path))

    assert error is None
    dynamic.assert_called_once_with(
        config=config,
        dir_path=str(tmp_path / "run"),
        reload_path=str(tmp_path / "log_config.json"),
        allow_file_config=False,
    )


@pytest.mark.parametrize(("command_class", "dynamic_path"), COMMANDS)
@pytest.mark.parametrize("config_type", ["inline-json", "file-path", "dict-factory"])
def test_remote_job_log_rejects_unsafe_config(tmp_path, command_class, dynamic_path, config_type):
    factory_calls = []

    def harmless_factory():
        factory_calls.append(True)
        return logging.NullHandler()

    if config_type == "inline-json":
        config = '{"version": 1}'
    elif config_type == "file-path":
        config_path = tmp_path / "attacker-log-config.json"
        config_path.write_text('{"version": 1}', encoding="utf-8")
        config = str(config_path)
    else:
        config = {
            "version": 1,
            "handlers": {"factory": {"()": harmless_factory}},
            "root": {"level": "INFO", "handlers": ["factory"]},
        }

    with patch(dynamic_path) as dynamic:
        error = command_class().process(config, _log_command_context(tmp_path))

    assert "only supports log levels and built-in log modes" in error
    dynamic.assert_not_called()
    assert not factory_calls


@pytest.mark.parametrize("config", SAFE_CONFIGS)
def test_server_site_log_accepts_safe_controls(tmp_path, config):
    conn = MagicMock()
    conn.app_ctx = MagicMock(spec=ServerEngine)
    workspace = conn.app_ctx.get_workspace.return_value
    workspace.get_root_dir.return_value = str(tmp_path)
    workspace.get_log_config_file_path.return_value = str(tmp_path / "log_config.json")
    message = MagicMock()
    command = SystemCommandModule()
    command.send_request_to_clients = MagicMock(return_value=[])
    command.process_replies_to_table = MagicMock()

    with (
        patch("nvflare.private.fed.server.sys_cmd.dynamic_log_config") as dynamic,
        patch("nvflare.private.fed.server.sys_cmd.new_message", return_value=message) as new_message,
    ):
        command.configure_site_log(conn, ["configure_site_log", "all", config])

    new_message.assert_called_once_with(conn, topic=SysCommandTopic.CONFIGURE_SITE_LOG, body=config, require_authz=True)
    dynamic.assert_called_once_with(
        config=config,
        dir_path=str(tmp_path),
        reload_path=str(tmp_path / "log_config.json"),
        allow_file_config=False,
    )


def test_server_site_log_rejects_existing_file_path(tmp_path):
    config_path = tmp_path / "attacker-log-config.json"
    config_path.write_text('{"version": 1}', encoding="utf-8")
    conn = MagicMock()

    with (
        patch("nvflare.private.fed.server.sys_cmd.dynamic_log_config") as dynamic,
        patch("nvflare.private.fed.server.sys_cmd.new_message") as new_message,
    ):
        SystemCommandModule().configure_site_log(conn, ["configure_site_log", "all", str(config_path)])

    assert "only supports log levels and built-in log modes" in conn.append_error.call_args.args[0]
    dynamic.assert_not_called()
    new_message.assert_not_called()


def test_server_site_log_rejects_non_level_logging_attribute():
    conn = MagicMock()

    SystemCommandModule().configure_site_log(conn, ["configure_site_log", "all", "basic_format"])

    assert "only supports log levels and built-in log modes" in conn.append_error.call_args.args[0]
    assert conn.append_error.call_args.kwargs["meta"][MetaKey.STATUS] == MetaStatusValue.SYNTAX_ERROR


@pytest.mark.parametrize("config", SAFE_CONFIGS)
def test_client_site_log_accepts_safe_controls(tmp_path, config):
    request = Message(topic=SysCommandTopic.CONFIGURE_SITE_LOG, body=config)

    with patch("nvflare.private.fed.client.sys_cmd.dynamic_log_config") as dynamic:
        reply = ConfigureSiteLogProcessor().process(request, _site_log_context(tmp_path))

    assert reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.OK
    dynamic.assert_called_once_with(
        config=config,
        dir_path=str(tmp_path),
        reload_path=str(tmp_path / "log_config.json"),
        allow_file_config=False,
    )


def test_client_site_log_rejects_existing_file_path(tmp_path):
    config_path = tmp_path / "attacker-log-config.json"
    config_path.write_text('{"version": 1}', encoding="utf-8")
    request = Message(topic=SysCommandTopic.CONFIGURE_SITE_LOG, body=str(config_path))

    with patch("nvflare.private.fed.client.sys_cmd.dynamic_log_config") as dynamic:
        reply = ConfigureSiteLogProcessor().process(request, _site_log_context(tmp_path))

    assert reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR
    assert "only supports log levels and built-in log modes" in reply.body
    dynamic.assert_not_called()
