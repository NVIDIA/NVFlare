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
from types import ModuleType

import pytest


def test_recipe_missing_subcommand_prints_help_then_error(capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import def_recipe_parser, handle_recipe_cmd

    cli_output._output_format = "txt"
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
        lambda framework=None: [{"name": "fedavg", "framework": "core", "description": "demo"}],
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


def test_recipe_catalog_is_discovered_from_package_modules(monkeypatch):
    from nvflare.recipe.spec import Recipe
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    class FakeRecipe(Recipe):
        """Demo discovered recipe."""

        def __init__(self):
            pass

    fake_package = ModuleType("fake.recipes")
    fake_package.__path__ = ["fake/recipes"]

    fake_module = ModuleType("fake.recipes.fedavg")
    FakeRecipe.__module__ = "fake.recipes.fedavg"
    setattr(fake_module, "FakeRecipe", FakeRecipe)

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [{"package": "fake.recipes", "framework": "pytorch"}],
    )

    def fake_import_module(name):
        if name == "fake.recipes":
            return fake_package
        if name == "fake.recipes.fedavg":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.recipes.fedavg", False)],
    )

    catalog = _load_catalog(framework="pytorch")

    assert catalog == [
        {
            "name": "fedavg-pt",
            "description": "Demo discovered recipe.",
            "framework": "pytorch",
            "module": "fake.recipes.fedavg",
            "class": "FakeRecipe",
        }
    ]


def test_recipe_catalog_core_framework_is_not_special_catch_all(monkeypatch):
    from nvflare.recipe.spec import Recipe
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    class CoreRecipe(Recipe):
        """Core recipe."""

        def __init__(self):
            pass

    core_package = ModuleType("fake.core")
    core_package.__path__ = ["fake/core"]

    core_module = ModuleType("fake.core.fedavg")
    CoreRecipe.__module__ = "fake.core.fedavg"
    setattr(core_module, "CoreRecipe", CoreRecipe)

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [
            {"package": "fake.core", "framework": "core"},
            {"package": "fake.pt", "framework": "pytorch"},
        ],
    )

    def fake_import_module(name):
        if name == "fake.core":
            return core_package
        if name == "fake.core.fedavg":
            return core_module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.core.fedavg", False)],
    )

    assert _load_catalog(framework="pytorch") == []
    assert _load_catalog(framework="core") == [
        {
            "name": "fedavg",
            "description": "Core recipe.",
            "framework": "core",
            "module": "fake.core.fedavg",
            "class": "CoreRecipe",
        }
    ]


def test_recipe_catalog_skips_plain_import_errors_from_optional_recipes(monkeypatch):
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    fake_package = ModuleType("fake.recipes")
    fake_package.__path__ = ["fake/recipes"]

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [{"package": "fake.recipes", "framework": "pytorch"}],
    )

    def fake_import_module(name):
        if name == "fake.recipes":
            return fake_package
        if name == "fake.recipes.broken":
            raise ImportError("broken recipe import")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.recipes.broken", False)],
    )

    assert _load_catalog(framework="pytorch") == []


def test_recipe_catalog_skips_syntax_errors_from_optional_recipes(monkeypatch):
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    fake_package = ModuleType("fake.recipes")
    fake_package.__path__ = ["fake/recipes"]

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [{"package": "fake.recipes", "framework": "pytorch"}],
    )

    def fake_import_module(name):
        if name == "fake.recipes":
            return fake_package
        if name == "fake.recipes.broken":
            raise SyntaxError("invalid syntax")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.recipes.broken", False)],
    )

    assert _load_catalog(framework="pytorch") == []


def test_recipe_catalog_prefers_leaf_recipe_class_when_module_has_base_and_subclass(monkeypatch):
    from nvflare.recipe.spec import Recipe
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    class BaseRecipe(Recipe):
        """Base helper recipe."""

        def __init__(self):
            pass

    class FinalRecipe(BaseRecipe):
        """Concrete exported recipe."""

        def __init__(self):
            pass

    fake_package = ModuleType("fake.recipes")
    fake_package.__path__ = ["fake/recipes"]

    fake_module = ModuleType("fake.recipes.swarm")
    BaseRecipe.__module__ = "fake.recipes.swarm"
    FinalRecipe.__module__ = "fake.recipes.swarm"
    setattr(fake_module, "BaseRecipe", BaseRecipe)
    setattr(fake_module, "FinalRecipe", FinalRecipe)

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [{"package": "fake.recipes", "framework": "pytorch"}],
    )

    def fake_import_module(name):
        if name == "fake.recipes":
            return fake_package
        if name == "fake.recipes.swarm":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.recipes.swarm", False)],
    )

    catalog = _load_catalog(framework="pytorch")

    assert catalog == [
        {
            "name": "swarm-pt",
            "description": "Concrete exported recipe.",
            "framework": "pytorch",
            "module": "fake.recipes.swarm",
            "class": "FinalRecipe",
        }
    ]
