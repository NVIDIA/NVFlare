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


class TestConfigOutput:
    """Tests for nvflare config output format."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.startup_kit_dir = kwargs.get("startup_kit_dir", None)
        args.poc_startup_kit_dir = kwargs.get("poc_startup_kit_dir", None)
        args.prod_startup_kit_dir = kwargs.get("prod_startup_kit_dir", None)
        args.poc_workspace_dir = kwargs.get("poc_workspace_dir", None)
        args.poc_workspace_dir_v2 = kwargs.get("poc_workspace_dir_v2", None)
        args.job_templates_dir = kwargs.get("job_templates_dir", None)
        args.debug = False
        return args

    def test_config_json_envelope_shape(self, capsys):
        """config command returns JSON with expected keys."""
        from nvflare.cli import handle_config_cmd

        args = self._make_args()

        mock_config = MagicMock()
        mock_config.get.return_value = None

        with patch("nvflare.cli.get_hidden_config", return_value=("/fake/config.conf", mock_config)):
            with patch("nvflare.tool.cli_schema.handle_schema_flag"):
                handle_config_cmd(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "config_file" in data["data"]
        assert "startup_kit_dir" in data["data"]
        assert "poc_startup_kit_dir" in data["data"]
        assert "prod_startup_kit_dir" in data["data"]
        assert "poc_workspace_dir" in data["data"]
        assert "job_templates_dir" in data["data"]

    def test_config_parser_has_schema_flag(self):
        """config parser should have --schema flag."""
        import argparse

        from nvflare.cli import def_config_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_config_parser(subs)
        args = root.parse_args(["config", "--schema"])
        assert args.schema is True

    def test_config_json_with_dirs_set(self, capsys):
        """When dirs are set, config output reflects them."""
        from nvflare.cli import handle_config_cmd

        args = self._make_args(startup_kit_dir="/path/to/startup")

        mock_config = MagicMock()
        mock_config.get.return_value = "/path/to/startup"

        with patch("nvflare.cli.get_hidden_config", return_value=("/fake/config.conf", mock_config)):
            with patch("nvflare.cli.create_startup_kit_config", return_value=mock_config):
                with patch("nvflare.cli.create_poc_workspace_config", return_value=mock_config):
                    with patch("nvflare.cli.create_job_template_config", return_value=mock_config):
                        with patch("nvflare.cli.save_config"):
                            with patch("nvflare.tool.cli_schema.handle_schema_flag"):
                                handle_config_cmd(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["config_file"] == "/fake/config.conf"

    def test_legacy_startup_kit_alias_only_updates_poc(self, capsys):
        from nvflare.cli import handle_config_cmd

        args = self._make_args(startup_kit_dir="/path/to/poc-startup")
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "poc.startup_kit": "/path/to/poc-startup",
            "prod.startup_kit": None,
            "poc.workspace": None,
            "job_template.path": None,
        }.get(key, default)

        with patch("nvflare.cli.get_hidden_config", return_value=("/fake/config.conf", mock_config)):
            with patch("nvflare.cli.create_startup_kit_config", return_value=mock_config) as mock_create_startup:
                with patch("nvflare.cli.create_poc_workspace_config", return_value=mock_config):
                    with patch("nvflare.cli.create_job_template_config", return_value=mock_config):
                        with patch("nvflare.cli.save_config"):
                            with patch("nvflare.tool.cli_schema.handle_schema_flag"):
                                handle_config_cmd(args)

        assert mock_create_startup.call_args_list[0].kwargs["target"] == "poc"
        assert mock_create_startup.call_args_list[0].args[1] == "/path/to/poc-startup"
        assert mock_create_startup.call_args_list[1].kwargs["target"] == "prod"
        assert mock_create_startup.call_args_list[1].args[1] is None

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["poc_startup_kit_dir"] == "/path/to/poc-startup"
        assert data["data"]["prod_startup_kit_dir"] is None

    def test_invalid_startup_kit_path_returns_invalid_args(self, capsys):
        from nvflare.cli import handle_config_cmd

        args = self._make_args(startup_kit_dir="/bad/startup")
        mock_config = MagicMock()
        mock_config.get.return_value = None

        with patch("nvflare.cli.get_hidden_config", return_value=("/fake/config.conf", mock_config)):
            with patch(
                "nvflare.cli.create_startup_kit_config",
                side_effect=ValueError("invalid startup kit location '/bad/startup'"),
            ):
                with patch("nvflare.tool.cli_schema.handle_schema_flag"):
                    with pytest.raises(SystemExit) as exc_info:
                        handle_config_cmd(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"
        assert "invalid startup kit location" in data["message"]
