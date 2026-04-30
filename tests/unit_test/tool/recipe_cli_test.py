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


def test_recipe_missing_subcommand_still_exits_when_usage_error_is_mocked():
    from unittest.mock import patch

    from nvflare.tool.recipe.recipe_cli import handle_recipe_cmd

    with patch("nvflare.tool.recipe.recipe_cli.output_usage_error") as mocked_usage_error:
        with pytest.raises(SystemExit) as exc_info:
            handle_recipe_cmd(Namespace(recipe_sub_cmd=None))

    assert exc_info.value.code == 4
    mocked_usage_error.assert_called_once()


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


def test_recipe_list_filters_catalog_with_repeated_filter_args(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda framework=None: [
            {
                "name": "fedavg-pt",
                "framework": "pytorch",
                "description": "FedAvg",
                "algorithm": "fedavg",
                "aggregation": "weighted_average",
                "state_exchange": "full_model",
                "privacy": [],
            },
            {
                "name": "fedavg-he-pt",
                "framework": "pytorch",
                "description": "FedAvg HE",
                "algorithm": "fedavg",
                "aggregation": "weighted_average",
                "state_exchange": "full_model",
                "privacy": ["homomorphic_encryption"],
            },
            {
                "name": "fedopt-pt",
                "framework": "pytorch",
                "description": "FedOpt",
                "algorithm": "fedopt",
                "aggregation": "server_optimizer",
                "state_exchange": "weight_diff",
                "privacy": [],
            },
        ],
    )

    cmd_recipe_list(Namespace(framework=None, filters=["framework=pytorch", "privacy=homomorphic-encryption"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert [entry["name"] for entry in payload["data"]] == ["fedavg-he-pt"]


def test_recipe_list_framework_flag_combines_with_filters(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    calls = []

    def fake_load_catalog(framework=None):
        calls.append(framework)
        return [
            {
                "name": "fedavg-pt",
                "framework": "pytorch",
                "description": "FedAvg",
                "algorithm": "fedavg",
                "aggregation": "weighted_average",
                "state_exchange": "full_model",
                "privacy": [],
            },
            {
                "name": "fedopt-pt",
                "framework": "pytorch",
                "description": "FedOpt",
                "algorithm": "fedopt",
                "aggregation": "server_optimizer",
                "state_exchange": "weight_diff",
                "privacy": [],
            },
        ]

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", fake_load_catalog)

    cmd_recipe_list(Namespace(framework="pytorch", filters=["algorithm=fedopt"]))

    payload = json.loads(capsys.readouterr().out)
    assert calls == ["pytorch"]
    assert [entry["name"] for entry in payload["data"]] == ["fedopt-pt"]


def test_recipe_list_valid_filter_with_no_matches_returns_empty_result(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda framework=None: [
            {
                "name": "fedavg-pt",
                "framework": "pytorch",
                "description": "FedAvg",
                "algorithm": "fedavg",
                "aggregation": "weighted_average",
                "state_exchange": "full_model",
                "privacy": [],
            }
        ],
    )

    cmd_recipe_list(Namespace(framework=None, filters=["privacy=differential_privacy"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["data"] == []


def test_recipe_list_rejects_conflicting_framework_filters(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")

    with pytest.raises(SystemExit) as exc_info:
        cmd_recipe_list(Namespace(framework="pytorch", filters=["framework=tensorflow"]))

    assert exc_info.value.code == 4
    captured = capsys.readouterr()
    assert '"error_code": "INVALID_ARGS"' in captured.out
    assert "conflicts with --filter framework=tensorflow" in captured.out


def test_recipe_list_rejects_invalid_filter(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")

    with pytest.raises(SystemExit) as exc_info:
        cmd_recipe_list(Namespace(framework=None, filters=["unknown=value"]))

    assert exc_info.value.code == 4
    captured = capsys.readouterr()
    assert '"error_code": "INVALID_ARGS"' in captured.out
    assert "unsupported filter key" in captured.out


def test_recipe_list_empty_framework_catalog_still_exits_when_output_error_is_mocked(monkeypatch):
    from unittest.mock import patch

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda framework=None: [])

    with patch("nvflare.tool.cli_output.output_error_message") as output_error_message:
        with patch("nvflare.tool.cli_output.output_ok") as output_ok:
            with pytest.raises(SystemExit) as exc_info:
                cmd_recipe_list(Namespace(framework="pytorch"))

    assert exc_info.value.code == 4
    output_error_message.assert_called_once()
    output_ok.assert_not_called()


def test_recipe_show_returns_queryable_metadata(monkeypatch, capsys):
    import json

    from nvflare.recipe.spec import Recipe
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import _CATALOG_RECIPE_CLASS_KEY, cmd_recipe_show

    class FakeRecipe(Recipe):
        """Fake detailed recipe."""

        optional_dependencies = ["pip install fake-framework"]
        template_references = ["nvflare/agent/templates/fake"]

        def __init__(
            self,
            *,
            min_clients: int,
            num_rounds: int = 2,
            train_script: str = "client.py",
            secure: bool = False,
        ):
            pass

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda include_recipe_class=False, framework=None: [
            {
                "name": "fake-pt",
                "description": "Fake detailed recipe.",
                "framework": "pytorch",
                "module": "fake.recipes.fake",
                "class": "FakeRecipe",
                "algorithm": "fedavg",
                "aggregation": "weighted_average",
                "state_exchange": "full_model",
                "privacy": [],
                _CATALOG_RECIPE_CLASS_KEY: FakeRecipe,
            }
        ],
    )

    cmd_recipe_show(Namespace(name="fake-pt"))

    payload = json.loads(capsys.readouterr().out)
    data = payload["data"]
    assert payload["status"] == "ok"
    assert data["name"] == "fake-pt"
    assert data["framework"] == "pytorch"
    assert data["privacy"] == []
    assert data["framework_support"] == ["pytorch"]
    assert data["privacy_compatible"] == ["homomorphic_encryption"]
    assert data["optional_dependencies"] == ["pip install fake-framework"]
    assert data["template_references"] == ["nvflare/agent/templates/fake"]
    assert data["client_requirements"]["min_clients"] == {"required": True, "default": None}
    assert data["client_requirements"]["requires_training_script"] is True
    assert {p["name"]: p for p in data["parameters"]}["num_rounds"]["default"] == 2


def test_recipe_show_unknown_recipe_errors(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda include_recipe_class=False: [])

    with pytest.raises(SystemExit) as exc_info:
        cmd_recipe_show(Namespace(name="missing"))

    assert exc_info.value.code == 4
    captured = capsys.readouterr()
    assert '"error_code": "INVALID_ARGS"' in captured.out
    assert "unknown recipe 'missing'" in captured.out
    assert "nvflare recipe list --format json" in captured.out


def test_recipe_show_schema_succeeds_without_name(capsys):
    from unittest.mock import patch

    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show, def_recipe_parser

    parser = ArgumentParser(prog="nvflare")
    subparsers = parser.add_subparsers(dest="sub_command")
    def_recipe_parser(subparsers)

    with patch("sys.argv", ["nvflare", "recipe", "show", "--schema"]):
        with pytest.raises(SystemExit) as exc_info:
            cmd_recipe_show(Namespace())

    assert exc_info.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["output_modes"] == ["json"]
    assert schema["streaming"] is False
    assert schema["mutating"] is False
    assert schema["idempotent"] is True
    assert schema["retry_token"] == {"supported": False}


def test_recipe_list_schema_includes_command_contract_metadata(capsys):
    from unittest.mock import patch

    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list, def_recipe_parser

    parser = ArgumentParser(prog="nvflare")
    subparsers = parser.add_subparsers(dest="sub_command")
    def_recipe_parser(subparsers)

    with patch("sys.argv", ["nvflare", "recipe", "list", "--schema"]):
        with pytest.raises(SystemExit) as exc_info:
            cmd_recipe_list(Namespace())

    assert exc_info.value.code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["output_modes"] == ["json"]
    assert schema["streaming"] is False
    assert schema["mutating"] is False
    assert schema["idempotent"] is True
    assert schema["retry_token"] == {"supported": False}


def test_recipe_detail_can_be_built_for_each_discovered_recipe():
    from nvflare.tool.recipe.recipe_cli import _load_catalog, _recipe_detail

    catalog = _load_catalog(include_recipe_class=True)
    assert catalog

    for entry in catalog:
        detail = _recipe_detail(entry)
        assert detail["name"] == entry["name"]
        assert "parameters" in detail
        assert "framework_support" in detail
        assert "privacy_compatible" in detail


def test_recipe_catalog_includes_all_documented_recipe_variants():
    from nvflare.tool.recipe.recipe_cli import _DOCUMENTED_RECIPE_SPECS, _load_catalog

    catalog = _load_catalog()
    names = {entry["name"] for entry in catalog}

    assert len(_DOCUMENTED_RECIPE_SPECS) == 21
    assert set(_DOCUMENTED_RECIPE_SPECS).issubset(names)


def test_recipe_list_filters_documented_recipe_variants_without_optional_dependencies(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")

    cmd_recipe_list(Namespace(framework="tensorflow", filters=["algorithm=fedprox"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert [entry["name"] for entry in payload["data"]] == ["fedprox-tf"]


def test_recipe_show_fedprox_documents_fedavg_constructor(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")

    cmd_recipe_show(Namespace(name="fedprox-pt"))

    payload = json.loads(capsys.readouterr().out)
    data = payload["data"]
    assert payload["status"] == "ok"
    assert data["algorithm"] == "fedprox"
    assert data["class"] == "FedAvgRecipe"
    assert data["notes"]
    assert "same recipe constructor as fedavg-pt" in data["notes"][0]


def test_recipe_show_uses_static_metadata_when_optional_dependency_is_missing(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")

    cmd_recipe_show(Namespace(name="xgb-horizontal"))

    payload = json.loads(capsys.readouterr().out)
    data = payload["data"]
    assert payload["status"] == "ok"
    assert data["name"] == "xgb-horizontal"
    assert data["framework"] == "xgboost"
    assert data["algorithm"] == "xgboost_horizontal"
    assert data["framework_support"] == ["xgboost"]
    assert data["privacy_compatible"] == ["homomorphic_encryption"]
    assert data["optional_dependencies"] == ["pip install xgboost"]
    assert data["parameters"]


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
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

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
            "algorithm": "fedavg",
            "aggregation": "weighted_average",
            "state_exchange": "full_model",
            "privacy": [],
        }
    ]


def test_recipe_catalog_prefers_specific_algorithm_marker_over_fedavg_class_name(monkeypatch):
    from nvflare.recipe.spec import Recipe
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    class KMeansFedAvgRecipe(Recipe):
        """KMeans recipe."""

        def __init__(self):
            pass

    fake_package = ModuleType("fake.recipes")
    fake_package.__path__ = ["fake/recipes"]

    fake_module = ModuleType("fake.recipes.kmeans")
    KMeansFedAvgRecipe.__module__ = "fake.recipes.kmeans"
    setattr(fake_module, "KMeansFedAvgRecipe", KMeansFedAvgRecipe)

    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._RECIPE_PACKAGE_ROOTS",
        [{"package": "fake.recipes", "framework": "sklearn"}],
    )
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

    def fake_import_module(name):
        if name == "fake.recipes":
            return fake_package
        if name == "fake.recipes.kmeans":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli.pkgutil.iter_modules",
        lambda path, prefix="": [(None, "fake.recipes.kmeans", False)],
    )

    catalog = _load_catalog(framework="sklearn")

    assert catalog[0]["algorithm"] == "kmeans"
    assert catalog[0]["aggregation"] == "cluster_centers"
    assert catalog[0]["state_exchange"] == "cluster_centers"


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
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

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
            "algorithm": "fedavg",
            "aggregation": "weighted_average",
            "state_exchange": "full_model",
            "privacy": [],
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
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

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
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

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
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._DOCUMENTED_RECIPE_SPECS", {})

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
            "algorithm": "swarm",
            "aggregation": None,
            "state_exchange": "full_model",
            "privacy": [],
        }
    ]
