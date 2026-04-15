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

from nvflare.tool import cli_output


class TestSystemStatus:
    """Tests for nvflare system status command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target=None, client_names=None, output="json"):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.output = output
        return args

    def test_status_json_output_shape(self, capsys):
        """JSON output has schema_version, status, and data keys."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"server_status": "running", "clients": []}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "data" in data

    def test_status_server_only(self, capsys):
        """Target=server calls check_status with 'server'."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="server")
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"server_status": "running"}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("server", None)

    def test_status_with_client_names(self, capsys):
        """Client names are passed to check_status."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="client", client_names=["site-1", "site-2"])
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"client_status": {}}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("client", ["site-1", "site-2"])

    def test_status_connection_failed_exits_2(self):
        """Connection failure exits with code 2."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("connection error")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_status(args)
        assert exc_info.value.code == 2

    def test_status_connection_failed_uses_custom_hint(self, capsys):
        """system status uses a non-recursive hint on connection failure."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("connection error")):
            with pytest.raises(SystemExit):
                cmd_system_status(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["hint"] == "Start the server or verify the admin startup kit endpoint."

    def test_status_default_target_is_all(self, capsys):
        """When target is None, defaults to 'all'."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target=None)
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("all", None)


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

    def test_resources_json_empty_output(self, capsys):
        """Empty resources still return a JSON envelope."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_resources(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"] == {}


class TestSystemStatusHuman:
    @pytest.fixture(autouse=True)
    def text_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

    def _make_args(self, target=None, client_names=None):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        return args

    def test_status_human_output_is_formatted(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {
            "server_status": "started",
            "server_start_time": 1775860407.993352,
            "jobs": [],
            "clients": [
                {"client_name": "site-1", "client_last_conn_time": 1775860421.002409, "fqcn": "site-1"},
                {"client_name": "site-2", "client_last_conn_time": 1775860421.75365, "fqcn": "site-2"},
            ],
            "client_status": [
                {"client_name": "site-1", "status": "no_jobs"},
                {"client_name": "site-2", "status": "no_jobs"},
            ],
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        assert "Engine status: started" in captured.out
        assert "Registered clients: 2" in captured.out
        assert "site-1" in captured.out
        assert "site-2" in captured.out
        assert "clients: [{'client_name'" not in captured.out

    def test_status_client_human_uses_client_status_count_when_inventory_missing(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="client")
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {
            "client_status": [
                {"client_name": "site-1", "status": "no_jobs"},
                {"client_name": "site-2", "status": "no_jobs"},
            ]
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        assert "Clients: 2" in captured.out
        assert "site-1" in captured.out
        assert "site-2" in captured.out
        assert "Connected: 0" not in captured.out
