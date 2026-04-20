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

import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import NoConnection
from nvflare.tool import cli_output


class TestSystemResources:
    """Tests for nvflare system resources command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target=None, client_names=None, output="json"):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.output = output
        return args

    def test_resources_json_output_shape(self, capsys):
        """JSON output has schema_version, status, and data keys."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        resource_data = {"server": {"cpu": "20%", "memory": "4GB"}}
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = resource_data

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_resources(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"] == resource_data

    def test_resources_calls_report_resources(self, capsys):
        """report_resources is called with correct target."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args(target="client", client_names=["site-1"])
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_resources(args)

        mock_sess.report_resources.assert_called_once_with("client", ["site-1"])

    def test_resources_connection_failed_exits_2(self):
        """Connection failure exits with code 2."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_resources(args)
        assert exc_info.value.code == 2

    def test_resources_no_connection_propagates_to_top_level_handler(self):
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.report_resources.side_effect = NoConnection("conn error")

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with pytest.raises(NoConnection):
                cmd_system_resources(args)

    def test_resources_connection_failed_does_not_emit_success_when_error_output_mocked(self):
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        mocked_output = MagicMock()
        mocked_ok = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with patch("nvflare.tool.system.system_cli.output_error", mocked_output):
                with patch("nvflare.tool.system.system_cli.output_ok", mocked_ok):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_system_resources(args)

        assert exc_info.value.code == 2
        mocked_output.assert_called_once()
        mocked_ok.assert_not_called()

    def test_resources_default_target_is_all(self, capsys):
        """When target is None, defaults to 'all'."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args(target=None)
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_resources(args)

        mock_sess.report_resources.assert_called_once_with("all", None)

    def test_resources_human_output_for_empty_result(self, capsys):
        """Human-readable empty result message is specific to missing resource selection."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args(output="default")
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch("nvflare.tool.cli_output._output_format", "default"):
                cmd_system_resources(args)

        captured = capsys.readouterr()
        assert "No resources specified." in captured.out

    def test_resources_human_empty_result_does_not_fall_through_to_output_ok(self):
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args(output="default")
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}
        mocked_ok = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch("nvflare.tool.cli_output._output_format", "default"):
                with patch("nvflare.tool.system.system_cli.output_ok", mocked_ok):
                    cmd_system_resources(args)

        mocked_ok.assert_not_called()
