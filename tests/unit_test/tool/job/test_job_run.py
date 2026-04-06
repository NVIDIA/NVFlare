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


def _make_args(recipe_folder=".", env="poc", entry=None, workspace="/tmp/sim_ws"):
    args = MagicMock()
    args.recipe_folder = recipe_folder
    args.env = env
    args.entry = entry
    args.workspace = workspace
    return args


class TestJobRun:
    """Tests for nvflare job run command."""

    def _patch_recipe(self, entry="mymod:MyRecipe"):
        """Return a context manager that patches importlib so a mock Recipe is found."""
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        import importlib

        original_import = importlib.import_module

        def fake_import(name, *a, **kw):
            if name == "mymod":
                mod = MagicMock()
                mod.MyRecipe = mock_cls
                return mod
            return original_import(name, *a, **kw)

        return patch("importlib.import_module", side_effect=fake_import), mock_inst

    def test_run_poc_success_returns_job_id(self, capsys, tmp_path):
        """--env poc: submit_job returns job_id; stdout is one JSON envelope."""
        from nvflare.tool.job.job_cli import cmd_job_run

        args = _make_args(recipe_folder=str(tmp_path), env="poc", entry="mymod:MyRecipe")
        mock_sess = MagicMock()
        mock_sess.submit_job.return_value = "job-xyz"

        patch_import, mock_inst = self._patch_recipe()
        with patch_import:
            with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                cmd_job_run(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["job_id"] == "job-xyz"
        assert data["data"]["env"] == "poc"

    def test_run_sim_success(self, capsys, tmp_path):
        """--env sim: runs simulator, output has status=FINISHED_OK."""
        from nvflare.tool.job.job_cli import cmd_job_run

        args = _make_args(recipe_folder=str(tmp_path), env="sim", entry="mymod:MyRecipe")

        mock_runner = MagicMock()
        patch_import, mock_inst = self._patch_recipe()
        with patch_import:
            with patch("nvflare.private.fed.app.simulator.simulator_runner.SimulatorRunner", return_value=mock_runner):
                cmd_job_run(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "FINISHED_OK"
        assert data["data"]["env"] == "sim"

    def test_run_recipe_not_found_exits_1(self, capsys, tmp_path):
        """Export failure (no recipe) → RECIPE_ENTRY_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_run

        # Empty folder — no .py files so auto-discovery finds nothing
        recipe_dir = tmp_path / "empty"
        recipe_dir.mkdir()
        args = _make_args(recipe_folder=str(recipe_dir), env="poc")

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_run(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "RECIPE_ENTRY_NOT_FOUND"

    def test_run_connection_failed_exits_2(self, capsys, tmp_path):
        """poc submit connection failure → CONNECTION_FAILED, exit 2."""
        from nvflare.tool.job.job_cli import cmd_job_run

        args = _make_args(recipe_folder=str(tmp_path), env="poc", entry="mymod:MyRecipe")
        mock_sess = MagicMock()
        mock_sess.submit_job.side_effect = Exception("connection refused")

        patch_import, _ = self._patch_recipe()
        with patch_import:
            with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_job_run(args)
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "CONNECTION_FAILED"

    def test_run_export_failure_exits_1(self, capsys, tmp_path):
        """recipe.export() raises → RECIPE_EXPORT_FAILED, exit 1; temp dir cleaned up."""
        from nvflare.tool.job.job_cli import cmd_job_run

        args = _make_args(recipe_folder=str(tmp_path), env="poc", entry="mymod:MyRecipe")

        import importlib

        original_import = importlib.import_module

        def fake_import(name, *a, **kw):
            if name == "mymod":
                mock_cls = MagicMock()
                mock_inst = MagicMock()
                mock_inst.export.side_effect = RuntimeError("export broke")
                mock_cls.return_value = mock_inst
                mod = MagicMock()
                mod.MyRecipe = mock_cls
                return mod
            return original_import(name, *a, **kw)

        with patch("importlib.import_module", side_effect=fake_import):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_run(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "RECIPE_EXPORT_FAILED"

    def test_run_stdout_is_json_only(self, capsys, tmp_path):
        """No human-readable text on stdout."""
        from nvflare.tool.job.job_cli import cmd_job_run

        args = _make_args(recipe_folder=str(tmp_path), env="poc", entry="mymod:MyRecipe")
        mock_sess = MagicMock()
        mock_sess.submit_job.return_value = "job-abc"

        patch_import, _ = self._patch_recipe()
        with patch_import:
            with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                cmd_job_run(args)

        captured = capsys.readouterr()
        # stdout must parse as JSON
        json.loads(captured.out)
        assert not captured.err.strip().startswith("{")

    def test_run_parser_args(self):
        """run parser: --recipe-folder, --env choices, --entry."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["run"]
        assert parser is not None
        args = parser.parse_args(["--env", "sim", "--entry", "recipe:MyRecipe"])
        assert args.env == "sim"
        assert args.entry == "recipe:MyRecipe"
