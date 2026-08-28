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

from nvflare.fuel.utils.log_utils import LogMode
from nvflare.private.fed.client.admin_commands import ConfigureJobLogCommand as ClientConfigureJobLogCommand
from nvflare.private.fed.server.server_commands import ConfigureJobLogCommand as ServerConfigureJobLogCommand

COMMANDS = [
    (ServerConfigureJobLogCommand, "nvflare.private.fed.server.server_commands.dynamic_log_config"),
    (ClientConfigureJobLogCommand, "nvflare.private.fed.client.admin_commands.dynamic_log_config"),
]


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


@pytest.mark.parametrize(("command_class", "dynamic_path"), COMMANDS)
@pytest.mark.parametrize("config", ["DEBUG", "20", LogMode.MSG_ONLY, LogMode.RELOAD])
def test_remote_job_log_accepts_safe_controls(tmp_path, command_class, dynamic_path, config):
    with patch(dynamic_path) as dynamic:
        error = command_class().process(config, _log_command_context(tmp_path))

    assert error is None
    dynamic.assert_called_once_with(
        config=config,
        dir_path=str(tmp_path / "run"),
        reload_path=str(tmp_path / "log_config.json"),
    )


@pytest.mark.parametrize(("command_class", "dynamic_path"), COMMANDS)
def test_remote_job_log_rejects_inline_json(tmp_path, command_class, dynamic_path):
    with patch(dynamic_path) as dynamic:
        error = command_class().process('{"version": 1}', _log_command_context(tmp_path))

    assert "only supports log levels and built-in log modes" in error
    dynamic.assert_not_called()


@pytest.mark.parametrize(("command_class", "dynamic_path"), COMMANDS)
def test_remote_job_log_rejects_existing_file_path(tmp_path, command_class, dynamic_path):
    config_path = tmp_path / "attacker-log-config.json"
    config_path.write_text('{"version": 1}', encoding="utf-8")

    with patch(dynamic_path) as dynamic:
        error = command_class().process(str(config_path), _log_command_context(tmp_path))

    assert "only supports log levels and built-in log modes" in error
    dynamic.assert_not_called()


@pytest.mark.parametrize("command_class", [ServerConfigureJobLogCommand, ClientConfigureJobLogCommand])
def test_remote_job_log_rejects_dict_config_factory_before_execution(tmp_path, command_class):
    factory_calls = []

    def harmless_factory():
        factory_calls.append(True)
        return logging.NullHandler()

    config = {
        "version": 1,
        "handlers": {"factory": {"()": harmless_factory}},
        "root": {"level": "INFO", "handlers": ["factory"]},
    }

    error = command_class().process(config, _log_command_context(tmp_path))

    assert "only supports log levels and built-in log modes" in error
    assert factory_calls == []
