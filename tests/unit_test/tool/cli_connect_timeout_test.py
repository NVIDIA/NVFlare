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


def _called_active_session_mock(*mocks):
    called = [mock for mock in mocks if mock.called]
    assert len(called) == 1
    return called[0]


def _call_arg(call, name, position):
    args, kwargs = call
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def test_cli_sets_connect_timeout(monkeypatch):
    from nvflare import cli as cli_mod

    def _fake_parse_args(_):
        args = MagicMock()
        args.format = "txt"
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


def test_job_get_session_uses_active_session_with_connect_timeout(monkeypatch):
    from nvflare.tool.job import job_cli

    with (
        patch("nvflare.tool.cli_output.get_connect_timeout", return_value=3.25),
        patch(
            "nvflare.tool.cli_session.new_cli_session_for_args",
            return_value=MagicMock(),
        ) as shared_active,
        patch(
            "nvflare.tool.job.job_cli.new_cli_session_for_args",
            return_value=MagicMock(),
        ) as job_active,
    ):
        job_cli._get_session(study="default")

    active_session = _called_active_session_mock(shared_active, job_active)
    assert _call_arg(active_session.call_args, "timeout", 1) == 3.25
    assert _call_arg(active_session.call_args, "study", 2) in (None, "default")


def test_system_get_session_uses_active_session_with_connect_timeout(monkeypatch):
    from nvflare.tool.system import system_cli

    with (
        patch("nvflare.tool.cli_output.get_connect_timeout", return_value=4.0),
        patch(
            "nvflare.tool.cli_session.new_cli_session_for_args",
            return_value=MagicMock(),
        ) as shared_active,
    ):
        system_cli._get_system_session()

    assert shared_active.called
    assert _call_arg(shared_active.call_args, "timeout", 1) == 4.0
