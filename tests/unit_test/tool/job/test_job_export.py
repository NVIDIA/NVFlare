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
import textwrap
from unittest.mock import MagicMock, patch

import pytest


def _make_args(recipe_dir=".", out="/tmp/fl_job", entry=None):
    args = MagicMock()
    args.recipe_dir = recipe_dir
    args.out = out
    args.entry = entry
    return args


class TestJobExport:
    """Tests for nvflare job export command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")

    def test_export_success_json_envelope(self, capsys, tmp_path):
        """Successful export: stdout is one JSON envelope with job_folder."""
        from nvflare.tool.job.job_cli import cmd_job_export

        recipe_dir = tmp_path / "recipe"
        recipe_dir.mkdir()
        out_dir = tmp_path / "fl_job"

        mock_recipe_cls = MagicMock()
        mock_recipe_inst = MagicMock()
        mock_recipe_cls.return_value = mock_recipe_inst

        args = _make_args(recipe_dir=str(recipe_dir), out=str(out_dir), entry="mymod:MyRecipe")

        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod
            mock_mod.MyRecipe = mock_recipe_cls

            cmd_job_export(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert "job_folder" in data["data"]
        assert "recipe_dir" in data["data"]

    def test_export_entry_not_found_exits_4(self, capsys, tmp_path):
        """ImportError on explicit --entry → RECIPE_ENTRY_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_export

        args = _make_args(
            recipe_dir=str(tmp_path),
            out=str(tmp_path / "out"),
            entry="nonexistent_mod:Foo",
        )

        with patch("importlib.import_module", side_effect=ImportError("no module")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_export(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "RECIPE_ENTRY_NOT_FOUND"

    def test_export_no_recipe_found_exits_1(self, capsys, tmp_path):
        """Auto-discovery with no Recipe subclasses → RECIPE_ENTRY_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_export

        # Empty recipe folder — no .py files
        recipe_dir = tmp_path / "empty_recipe"
        recipe_dir.mkdir()

        args = _make_args(recipe_dir=str(recipe_dir), out=str(tmp_path / "out"))

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_export(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "RECIPE_ENTRY_NOT_FOUND"

    def test_export_ambiguous_recipes_exits_1(self, capsys, tmp_path):
        """Auto-discovery with multiple Recipe subclasses → RECIPE_ENTRY_AMBIGUOUS, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_export

        recipe_dir = tmp_path / "multi_recipe"
        recipe_dir.mkdir()
        # Write a .py with two Recipe subclasses
        (recipe_dir / "recipe.py").write_text(
            textwrap.dedent(
                """
                from nvflare.recipe.spec import Recipe
                class RecipeA(Recipe):
                    def export(self, out): pass
                class RecipeB(Recipe):
                    def export(self, out): pass
            """
            )
        )

        args = _make_args(recipe_dir=str(recipe_dir), out=str(tmp_path / "out"))

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_export(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "RECIPE_ENTRY_AMBIGUOUS"

    def test_export_export_failure_exits_1(self, capsys, tmp_path):
        """recipe.export() raises → RECIPE_EXPORT_FAILED, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_export

        args = _make_args(
            recipe_dir=str(tmp_path),
            out=str(tmp_path / "out"),
            entry="mymod:MyRecipe",
        )

        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        mock_inst.export.side_effect = RuntimeError("export broke")

        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod
            mock_mod.MyRecipe = mock_cls

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_export(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "RECIPE_EXPORT_FAILED"

    def test_export_no_json_on_stderr(self, capsys, tmp_path):
        """No JSON envelope leaks to stderr on success."""
        from nvflare.tool.job.job_cli import cmd_job_export

        args = _make_args(
            recipe_dir=str(tmp_path),
            out=str(tmp_path / "out"),
            entry="mymod:MyRecipe",
        )
        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod
            mock_mod.MyRecipe = mock_cls
            cmd_job_export(args)

        captured = capsys.readouterr()
        assert not captured.err.strip().startswith("{")

    def test_export_parser_args(self):
        """export parser: --recipe-dir, --out required, optional --entry."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["export"]
        assert parser is not None
        args = parser.parse_args(["--out", "./fl_job", "--entry", "recipe:MyRecipe"])
        assert args.out == "./fl_job"
        assert args.entry == "recipe:MyRecipe"
