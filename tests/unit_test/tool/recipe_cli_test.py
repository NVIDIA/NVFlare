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

from argparse import ArgumentParser, Namespace

import pytest


def test_recipe_missing_subcommand_prints_help_then_error(capsys):
    from nvflare.tool.recipe.recipe_cli import def_recipe_parser, handle_recipe_cmd

    parser = ArgumentParser(prog="nvflare")
    subparsers = parser.add_subparsers(dest="sub_command")
    def_recipe_parser(subparsers)

    with pytest.raises(SystemExit) as exc_info:
        handle_recipe_cmd(Namespace(recipe_sub_cmd=None))
    assert exc_info.value.code == 4

    captured = capsys.readouterr()
    assert "usage: nvflare recipe" in captured.err
    assert "\n\nInvalid arguments. — recipe subcommand required\n" in captured.err
    assert "Hint: Run with -h for usage." in captured.err
    assert "Code: INVALID_ARGS (exit 4)" in captured.err


def test_recipe_list_human_output_not_duplicated(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda framework=None: [{"name": "fedavg", "framework": "any", "description": "demo"}],
    )

    cmd_recipe_list(Namespace(framework=None))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("fedavg") == 1
    assert "description: demo" not in captured.out


def test_recipe_list_framework_with_no_matches_errors_in_json(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda framework=None: [])

    with pytest.raises(SystemExit) as exc_info:
        cmd_recipe_list(Namespace(framework="pytorch"))
    assert exc_info.value.code == 4

    captured = capsys.readouterr()
    assert '"error_code": "INVALID_ARGS"' in captured.out
    assert "no installed recipes found for framework 'pytorch'" in captured.out
    assert "pip install nvflare[PT]" in captured.out
    assert "pip install torch" in captured.out


def test_recipe_list_human_empty_catalog_explains_why(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda framework=None: [])

    cmd_recipe_list(Namespace(framework=None))

    captured = capsys.readouterr()
    assert "No recipes are currently available." in captured.out
    assert "Install optional framework dependencies" in captured.out
    assert "pip install nvflare[PT,SKLEARN]" in captured.out
    assert "pip install tensorflow xgboost" in captured.out


def test_recipe_list_human_empty_framework_catalog_suggests_framework_install(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda framework=None: [])

    with pytest.raises(SystemExit):
        cmd_recipe_list(Namespace(framework="pytorch"))

    captured = capsys.readouterr()
    assert "no installed recipes found for framework 'pytorch'" in captured.err
    assert "pip install nvflare[PT]" in captured.err
    assert "pip install torch" in captured.err
