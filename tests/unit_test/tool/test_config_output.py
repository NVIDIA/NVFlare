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


class TestConfigOutput:
    """Tests for nvflare config output format."""

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.startup_kit_dir = kwargs.get("startup_kit_dir", None)
        args.poc_workspace_dir = kwargs.get("poc_workspace_dir", None)
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
        assert "config_file" in data["data"]
        assert "startup_kit_dir" in data["data"]
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
        assert data["data"]["config_file"] == "/fake/config.conf"
