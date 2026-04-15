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

from unittest.mock import MagicMock, patch


def test_cli_sets_connect_timeout(monkeypatch):
    from nvflare import cli as cli_mod

    def _fake_parse_args(_):
        args = MagicMock()
        args.out_format = "txt"
        args.connect_timeout = 7.5
        args.sub_command = None
        args.version = False
        return MagicMock(), args, {}

    with patch.object(cli_mod, "parse_args", side_effect=_fake_parse_args):
        with patch("nvflare.tool.cli_output.set_output_format"):
            with patch("nvflare.tool.cli_output.set_connect_timeout") as mock_set_timeout:
                with patch.object(cli_mod, "print_nvflare_version"):
                    cli_mod.run("nvflare")
    mock_set_timeout.assert_called_once_with(7.5)


def test_job_get_session_uses_connect_timeout_env(monkeypatch):
    from nvflare.tool.job import job_cli

    with patch.object(job_cli, "find_admin_user_and_dir", return_value=("admin", "/tmp/startup")):
        with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=3.25):
            with patch("nvflare.tool.job.job_cli.new_cli_session") as mock_session:
                job_cli._get_session()

    _, kwargs = mock_session.call_args
    assert kwargs["timeout"] == 3.25


def test_system_get_session_uses_connect_timeout_env(monkeypatch):
    from nvflare.tool.system import system_cli

    with patch("nvflare.utils.cli_utils.get_hidden_config", return_value=(None, {})):
        with patch("nvflare.utils.cli_utils.get_startup_kit_dir_for_target", return_value="/tmp/startup"):
            with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=4.0):
                with patch("nvflare.tool.cli_session.new_cli_session") as mock_session:
                    system_cli._get_system_session()

    _, kwargs = mock_session.call_args
    assert kwargs["timeout"] == 4.0
