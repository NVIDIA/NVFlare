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

import nvflare as _nvflare_mod
from nvflare.tool import cli_output


def _make_args(site="all"):
    args = MagicMock()
    args.site = site
    return args


def _make_session(client_names=None, raw_versions=None):
    """Build a mock session with get_system_info() and report_version() pre-configured."""
    mock_sess = MagicMock()

    mock_sys_info = MagicMock()
    clients = []
    for name in client_names or []:
        c = MagicMock()
        c.name = name
        clients.append(c)
    mock_sys_info.client_info = clients
    mock_sess.get_system_info.return_value = mock_sys_info

    if raw_versions is None:
        raw_versions = {"server": {"version": "2.8.0"}}
        for name in client_names or []:
            raw_versions[name] = {"version": "2.8.0"}
    mock_sess.report_version.return_value = raw_versions

    return mock_sess


class TestSystemVersion:
    """Tests for nvflare system version command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def test_version_json_output_shape(self, capsys):
        """JSON envelope has sites, compatible, mismatched_sites, and admin_version keys."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        payload = data["data"]
        assert "sites" in payload
        assert "compatible" in payload
        assert "mismatched_sites" in payload
        assert "admin_version" in payload

    def test_version_all_sites_same_version_is_compatible(self, capsys):
        """All sites on the same version → compatible=True, mismatched_sites=[]."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1", "site-2"],
            raw_versions={
                "server": {"version": "2.8.0"},
                "site-1": {"version": "2.8.0"},
                "site-2": {"version": "2.8.0"},
            },
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        payload = json.loads(capsys.readouterr().out)["data"]
        assert payload["compatible"] is True
        assert payload["mismatched_sites"] == []

    def test_version_mismatch_detected(self, capsys):
        """One client on a different version → compatible=False, mismatched_sites=[that client]."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1", "site-2"],
            raw_versions={
                "server": {"version": "2.8.0"},
                "site-1": {"version": "2.8.0"},
                "site-2": {"version": "2.7.2"},
            },
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        payload = json.loads(capsys.readouterr().out)["data"]
        assert payload["compatible"] is False
        assert payload["mismatched_sites"] == ["site-2"]

    def test_version_multiple_sites_listed(self, capsys):
        """Server and two clients all appear in sites list with correct versions."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(client_names=["site-1", "site-2"])

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        payload = json.loads(capsys.readouterr().out)["data"]
        site_names = [s["site"] for s in payload["sites"]]
        assert "server" in site_names
        assert "site-1" in site_names
        assert "site-2" in site_names
        assert all(s["version"] == "2.8.0" for s in payload["sites"])

    def test_version_admin_version_is_local(self, capsys):
        """admin_version reflects the local CLI nvflare.__version__."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        payload = json.loads(capsys.readouterr().out)["data"]
        assert payload["admin_version"] == "2.8.0"

    def test_version_site_filter(self, capsys):
        """--site server queries only server and returns one entry."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1"],
            raw_versions={"server": {"version": "2.8.0"}},
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args(site="server"))

        payload = json.loads(capsys.readouterr().out)["data"]
        assert len(payload["sites"]) == 1
        assert payload["sites"][0]["site"] == "server"
        assert payload["sites"][0]["version"] == "2.8.0"

    def test_version_client_only_omits_compatibility_without_server(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1"],
            raw_versions={"site-1": {"version": "2.8.0"}},
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "9.9.9"):
                cmd_system_version(_make_args(site="site-1"))

        payload = json.loads(capsys.readouterr().out)["data"]
        assert "compatible" not in payload
        assert "mismatched_sites" not in payload

    def test_version_site_not_found_exits_1(self):
        """--site for an unknown name → SITE_NOT_FOUND, exits 1."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(client_names=["site-1"])

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_system_version(_make_args(site="nonexistent"))
        assert exc_info.value.code == 1

    def test_version_site_not_found_error_code(self, capsys):
        """Error envelope for unknown --site contains SITE_NOT_FOUND."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(client_names=["site-1"])

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                with pytest.raises(SystemExit):
                    cmd_system_version(_make_args(site="nonexistent"))

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "SITE_NOT_FOUND"
        assert "{site}" not in envelope["message"]
        assert "nonexistent" in envelope["message"]

    def test_version_connection_failed_exits_2(self):
        """Session failure → CONNECTION_FAILED, exits 2."""
        from nvflare.tool.system.system_cli import cmd_system_version

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("timeout")):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_system_version(_make_args())
        assert exc_info.value.code == 2

    def test_version_report_version_called_with_all(self):
        """report_version is called with target_type='all' when --site all."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args(site="all"))

        mock_sess.report_version.assert_called_once_with("all", None)

    def test_version_report_version_called_with_server(self):
        """report_version is called with target_type='server' when --site server."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(raw_versions={"server": {"version": "2.8.0"}})

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args(site="server"))

        mock_sess.report_version.assert_called_once_with("server", None)

    def test_version_unknown_version_falls_back(self, capsys):
        """Sites not present in raw_versions dict get version='unknown'."""
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1"],
            raw_versions={"server": {"version": "2.8.0"}},
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())


class TestSystemVersionHuman:
    @pytest.fixture(autouse=True)
    def text_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

    def test_version_human_output_is_formatted(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1", "site-2"],
            raw_versions={
                "server": {"version": "2.8.0"},
                "site-1": {"version": "2.8.0"},
                "site-2": {"version": "2.7.2"},
            },
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "2.8.0"):
                cmd_system_version(_make_args())

        captured = capsys.readouterr()
        assert "Versions" in captured.out
        assert "server" in captured.out
        assert "site-1" in captured.out
        assert "site-2" in captured.out
        assert "Compatible: no" in captured.out
        assert "Mismatched sites: site-2" in captured.out
        assert "sites: [{'site':" not in captured.out

    def test_version_human_output_omits_compatibility_without_server(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_version

        mock_sess = _make_session(
            client_names=["site-1"],
            raw_versions={"site-1": {"version": "2.8.0"}},
        )

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch.object(_nvflare_mod, "__version__", "9.9.9"):
                cmd_system_version(_make_args(site="site-1"))

        captured = capsys.readouterr()
        assert "Compatible:" not in captured.out
        assert "Mismatched sites:" not in captured.out
