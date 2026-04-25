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

import argparse
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pyhocon import ConfigFactory as CF

from nvflare.tool import cli_output


def _config_getter(values):
    def _get(key, default=None):
        return values.get(key, default)

    return _get


class TestConfigOutput:
    """Tests for nvflare config after startup-kit registry ownership moved to nvflare kit."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, **kwargs):
        return SimpleNamespace(
            poc_workspace_dir=kwargs.get("poc_workspace_dir"),
            job_templates_dir=kwargs.get("job_templates_dir"),
            debug=False,
            schema=False,
        )

    def test_config_json_envelope_shape_excludes_startup_kit_fields(self, capsys):
        from nvflare.cli import handle_config_cmd

        args = self._make_args()
        mock_config = MagicMock()
        mock_config.get.side_effect = _config_getter(
            {
                "poc.startup_kit": "/legacy/poc/startup",
                "prod.startup_kit": "/legacy/prod/startup",
                "poc.workspace": "/path/to/poc",
                "job_template.path": "/path/to/templates",
            }
        )

        with (
            patch("nvflare.cli.load_hidden_config_state", return_value=("/fake/config.conf", mock_config, False)),
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            handle_config_cmd(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["config_file"] == "/fake/config.conf"
        assert data["data"]["poc_workspace_dir"] == "/path/to/poc"
        assert data["data"]["job_templates_dir"] == "/path/to/templates"
        assert "startup_kit_dir" not in data["data"]
        assert "poc_startup_kit_dir" not in data["data"]
        assert "prod_startup_kit_dir" not in data["data"]

    def test_config_parser_has_schema_flag_and_keeps_non_startup_settings(self):
        from nvflare.cli import def_config_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_config_parser(subs)

        args = root.parse_args(["config", "--schema"])
        assert args.schema is True

        args = root.parse_args(["config", "-pw", "/path/to/poc", "-jt", "/path/to/templates"])
        assert args.poc_workspace_dir == "/path/to/poc"
        assert args.job_templates_dir == "/path/to/templates"

    @pytest.mark.parametrize(
        "old_args",
        [
            ["--poc.startup_kit", "/path/to/startup"],
            ["--prod.startup_kit", "/path/to/startup"],
            ["--startup_kit_dir", "/path/to/startup"],
            ["-d", "/path/to/startup"],
        ],
    )
    def test_config_parser_rejects_old_startup_kit_arguments(self, old_args):
        from nvflare.cli import def_config_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_config_parser(subs)

        with pytest.raises(SystemExit):
            root.parse_args(["config", *old_args])

    def test_config_write_updates_only_non_startup_settings(self, capsys):
        from nvflare.cli import handle_config_cmd

        args = self._make_args(poc_workspace_dir="/path/to/poc", job_templates_dir="/path/to/templates")
        mock_config = MagicMock()
        mock_config.get.side_effect = _config_getter(
            {
                "poc.workspace": "/path/to/poc",
                "job_template.path": "/path/to/templates",
            }
        )

        with (
            patch("nvflare.cli.load_hidden_config_state", return_value=("/fake/config.conf", mock_config, False)),
            patch("nvflare.cli.create_poc_workspace_config", return_value=mock_config) as create_poc,
            patch("nvflare.cli.create_job_template_config", return_value=mock_config) as create_job_template,
            patch("nvflare.cli.save_config") as save_config,
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            handle_config_cmd(args)

        create_poc.assert_called_once_with(mock_config, "/path/to/poc")
        create_job_template.assert_called_once_with(mock_config, "/path/to/templates")
        save_config.assert_called_once_with(mock_config, "/fake/config.conf")

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["config_file"] == "/fake/config.conf"
        assert data["data"]["poc_workspace_dir"] == "/path/to/poc"
        assert data["data"]["job_templates_dir"] == "/path/to/templates"
        assert "startup_kit_dir" not in data["data"]
        assert "poc_startup_kit_dir" not in data["data"]
        assert "prod_startup_kit_dir" not in data["data"]

    def test_config_write_warns_when_startup_kit_keys_are_removed(self, capsys):
        from nvflare.cli import handle_config_cmd

        args = self._make_args(poc_workspace_dir="/new/poc")
        config = CF.parse_string(
            """
                poc {
                    workspace = "/old/poc"
                    startup_kit = "/old/poc/admin@nvidia.com"
                }
                prod {
                    startup_kit = "/old/prod/admin@nvidia.com"
                }
            """
        )

        with (
            patch("nvflare.cli.load_hidden_config_state", return_value=("/fake/config.conf", config, False)),
            patch("nvflare.cli.backup_hidden_config_file", return_value="/fake/config.conf.bak") as backup_config,
            patch("nvflare.cli.save_config") as save_config,
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            handle_config_cmd(args)

        backup_config.assert_called_once_with("/fake/config.conf")
        saved_config = save_config.call_args.args[0]
        assert saved_config.get("poc.workspace") == "/new/poc"
        assert saved_config.get("poc.startup_kit", None) is None
        assert saved_config.get("prod.startup_kit", None) is None

        captured = capsys.readouterr()
        assert "removed startup kit config keys" in captured.err
        assert "poc.startup_kit" in captured.err
        assert "prod.startup_kit" in captured.err
        assert "backup saved to /fake/config.conf.bak" in captured.err
        assert json.loads(captured.out)["status"] == "ok"
