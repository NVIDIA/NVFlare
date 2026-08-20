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

import importlib
import json
import sys
from argparse import ArgumentParser, Namespace

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
    assert "Loading installed recipe catalog..." in captured.out
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
                "parameters": [
                    {
                        "name": "params_transfer_type",
                        "type": "TransferType",
                        "required": False,
                        "default": "FULL",
                        "kind": "keyword_only",
                    }
                ],
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

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")
    detail = {
        "name": "fake-pt",
        "description": "Fake detailed recipe.",
        "framework": "pytorch",
        "module": "fake.recipes.fake",
        "class": "FakeRecipe",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "privacy": [],
        "client_requirements": {
            "min_clients": {"required": True, "default": None},
            "requires_training_script": True,
            "requires_per_site_config": False,
        },
        "framework_support": ["pytorch"],
        "privacy_compatible": ["homomorphic_encryption"],
        "parameters": [{"name": "num_rounds", "type": "int", "required": False, "default": 2, "kind": "keyword_only"}],
        "optional_dependencies": ["pip install fake-framework"],
        "template_references": ["nvflare/agent/templates/fake"],
    }
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda include_recipe_detail=False, framework=None: [detail],
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
    assert data["client_requirements"]["requires_per_site_config"] is False
    assert {p["name"]: p for p in data["parameters"]}["num_rounds"]["default"] == 2


def test_recipe_show_human_output_reports_loading_status(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    detail = {
        "name": "fedavg",
        "description": "FedAvg recipe.",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "framework_support": ["pytorch", "tensorflow", "sklearn"],
        "privacy": [],
        "privacy_compatible": [],
        "parameters": [
            {
                "name": "params_transfer_type",
                "type": "str",
                "required": False,
                "default": "FULL",
                "kind": "keyword_only",
            }
        ],
    }
    monkeypatch.setattr(
        "nvflare.tool.recipe.recipe_cli._load_catalog",
        lambda include_recipe_detail=False, framework=None: [detail],
    )

    cmd_recipe_show(Namespace(name="fedavg"))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Loading installed recipe metadata for 'fedavg'..." in captured.out
    assert "recipe: fedavg" in captured.out
    assert "state_exchange: full_model (default; params_transfer_type=FULL, supports FULL or DIFF)" in captured.out
    assert "privacy: none enabled by default" in captured.out
    assert "privacy compatibility: not declared in recipe metadata" in captured.out
    assert "parameters: 1 available; run 'nvflare recipe show fedavg --format json'" in captured.out


def test_recipe_show_unknown_recipe_errors(monkeypatch, capsys):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr("nvflare.tool.recipe.recipe_cli._load_catalog", lambda include_recipe_detail=False: [])

    with pytest.raises(SystemExit) as exc_info:
        cmd_recipe_show(Namespace(name="missing"))

    assert exc_info.value.code == 4
    captured = capsys.readouterr()
    assert '"error_code": "INVALID_ARGS"' in captured.out
    assert "unknown recipe 'missing'" in captured.out
    assert "nvflare recipe list --format json" in captured.out


def test_recipe_list_reports_missing_generated_catalog(monkeypatch, capsys, tmp_path):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe import recipe_cli

    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(recipe_cli, "_RECIPE_CATALOG_PATH", tmp_path / "missing.json")

    with pytest.raises(SystemExit) as exc_info:
        recipe_cli.cmd_recipe_list(Namespace(framework=None, filters=[]))

    assert exc_info.value.code == 5
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert payload["exit_code"] == 5
    assert "Unable to load recipe metadata" in payload["message"]
    assert "regenerate the catalog" in payload["hint"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 999, "recipes": []},
        {"schema_version": 1, "recipes": [{}]},
        {"schema_version": 1, "recipes": [{"summary": {}, "detail": {}}]},
    ],
)
def test_recipe_show_reports_invalid_generated_catalog(payload, monkeypatch, capsys, tmp_path):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe import recipe_cli

    invalid_catalog = tmp_path / "invalid.json"
    invalid_catalog.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(recipe_cli, "_RECIPE_CATALOG_PATH", invalid_catalog)

    with pytest.raises(SystemExit) as exc_info:
        recipe_cli.cmd_recipe_show(Namespace(name="fedavg-pt"))

    assert exc_info.value.code == 5
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert payload["exit_code"] == 5
    assert "invalid generated recipe catalog" in payload["message"]


def test_recipe_show_reports_invalid_parameter_metadata(monkeypatch, capsys, tmp_path):
    from nvflare.tool import cli_output
    from nvflare.tool.recipe import recipe_cli

    catalog = json.loads(recipe_cli._RECIPE_CATALOG_PATH.read_text(encoding="utf-8"))
    catalog["recipes"][0]["detail"]["parameters"] = [{}]
    invalid_catalog = tmp_path / "invalid-parameter.json"
    invalid_catalog.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(recipe_cli, "_RECIPE_CATALOG_PATH", invalid_catalog)

    with pytest.raises(SystemExit) as exc_info:
        recipe_cli.cmd_recipe_show(Namespace(name=catalog["recipes"][0]["detail"]["name"]))

    assert exc_info.value.code == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert "invalid generated recipe catalog" in payload["message"]


def test_recipe_catalog_entry_validation_requires_fields_and_types():
    from nvflare.tool.recipe import recipe_cli

    catalog = json.loads(recipe_cli._RECIPE_CATALOG_PATH.read_text(encoding="utf-8"))
    assert all(recipe_cli._is_valid_catalog_entry(entry) for entry in catalog["recipes"])
    valid_entry = catalog["recipes"][0]

    for section in ("summary", "detail"):
        invalid_entry = json.loads(json.dumps(valid_entry))
        invalid_entry[section] = {}
        assert not recipe_cli._is_valid_catalog_entry(invalid_entry)

    invalid_entry = json.loads(json.dumps(valid_entry))
    invalid_entry["summary"]["framework"] = []
    assert not recipe_cli._is_valid_catalog_entry(invalid_entry)

    invalid_entry = json.loads(json.dumps(valid_entry))
    invalid_entry["detail"]["framework_support"] = "pytorch"
    assert not recipe_cli._is_valid_catalog_entry(invalid_entry)

    parameter = valid_entry["detail"]["parameters"][0]
    for field, invalid_value in (
        ("name", None),
        ("type", []),
        ("required", 1),
        ("kind", None),
    ):
        invalid_entry = json.loads(json.dumps(valid_entry))
        invalid_parameter = dict(parameter)
        invalid_parameter[field] = invalid_value
        invalid_entry["detail"]["parameters"] = [invalid_parameter]
        assert not recipe_cli._is_valid_catalog_entry(invalid_entry)

    invalid_entry = json.loads(json.dumps(valid_entry))
    invalid_parameter = dict(parameter)
    invalid_parameter.pop("default")
    invalid_entry["detail"]["parameters"] = [invalid_parameter]
    assert not recipe_cli._is_valid_catalog_entry(invalid_entry)


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


def test_recipe_detail_is_available_for_each_catalog_recipe():
    from nvflare.tool.recipe.recipe_cli import _load_catalog

    catalog = _load_catalog(include_recipe_detail=True)
    assert catalog

    for entry in catalog:
        assert "parameters" in entry
        assert "framework_support" in entry
        assert "privacy_compatible" in entry


def test_recipe_catalog_includes_all_documented_recipe_variants():
    from nvflare.tool.recipe.recipe_cli import _DOCUMENTED_RECIPE_SPECS, _load_catalog

    catalog = _load_catalog()
    names = {entry["name"] for entry in catalog}

    assert len(_DOCUMENTED_RECIPE_SPECS) == 21
    assert set(_DOCUMENTED_RECIPE_SPECS).issubset(names)
    assert "fedce-pt" in names


def test_recipe_list_omits_tensorflow_fedprox_without_a_concrete_recipe(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")

    cmd_recipe_list(Namespace(framework="tensorflow", filters=["algorithm=fedprox"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["data"] == []
    assert "fedprox-tf" not in str(payload)


def test_recipe_show_fedprox_uses_concrete_recipe(monkeypatch, capsys):
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe import recipe_cli

    monkeypatch.setattr(cli_output, "_output_format", "json")

    recipe_cli.cmd_recipe_show(Namespace(name="fedprox-pt"))

    payload = json.loads(capsys.readouterr().out)
    data = payload["data"]
    assert payload["status"] == "ok"
    assert data["algorithm"] == "fedprox"
    assert data["class"] == "FedProxRecipe"
    assert data["aggregation"] == "weighted_average"
    assert data["state_exchange"] == "full_model"
    params = {parameter["name"]: parameter for parameter in data["parameters"]}
    assert params["fedprox_mu"]["default"] == 0.01
    assert data["notes"]
    assert "Lightning" in data["notes"][0]


def test_recipe_list_uses_generated_catalog_without_importing_recipe_modules(monkeypatch, capsys):
    import importlib
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import _RECIPE_PACKAGE_ROOTS, cmd_recipe_list

    monkeypatch.setattr(cli_output, "_output_format", "json")
    real_import_module = importlib.import_module
    recipe_packages = tuple(root["package"] for root in _RECIPE_PACKAGE_ROOTS)

    def reject_recipe_import(name, *args, **kwargs):
        if name.startswith(recipe_packages):
            raise AssertionError(f"recipe listing unexpectedly imported {name}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", reject_recipe_import)

    cmd_recipe_list(Namespace(framework=None, filters=[]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert len(payload["data"]) == 25


def test_recipe_show_uses_static_metadata_when_optional_dependency_is_missing(monkeypatch, capsys):
    import builtins
    import importlib
    import json

    from nvflare.tool import cli_output
    from nvflare.tool.recipe.recipe_cli import cmd_recipe_show

    monkeypatch.setattr(cli_output, "_output_format", "json")
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def reject_xgboost_import(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("nvflare.app_opt.xgboost"):
            raise AssertionError(f"recipe metadata unexpectedly imported {name}")
        return real_import(name, *args, **kwargs)

    def reject_xgboost_import_module(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("nvflare.app_opt.xgboost"):
            raise AssertionError(f"recipe metadata unexpectedly imported {name}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_xgboost_import)
    monkeypatch.setattr(importlib, "import_module", reject_xgboost_import_module)

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


def test_recipe_catalog_generation_discovers_source_without_importing_optional_dependencies(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "kmeans.py").write_text(
        """import unavailable_optional_dependency
from nvflare.recipe.spec import Recipe

class KMeansFedAvgRecipe(Recipe):
    \"\"\"KMeans recipe.\"\"\"

    def __init__(self, *, num_rounds: int = 2):
        pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "sklearn"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    catalog = recipe_cli._discover_recipe_catalog()

    assert catalog == [
        {
            "name": "kmeans-sklearn",
            "description": "KMeans recipe.",
            "framework": "sklearn",
            "module": "nvflare.fake.recipes.kmeans",
            "class": "KMeansFedAvgRecipe",
            "algorithm": "kmeans",
            "aggregation": "cluster_centers",
            "state_exchange": "cluster_centers",
            "privacy": [],
        }
    ]


def test_recipe_catalog_generation_skips_syntax_errors(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "broken.py").write_text("class BrokenRecipe(Recipe)\n    pass", encoding="utf-8")
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "pytorch"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    assert recipe_cli._discover_recipe_catalog() == []


def test_recipe_catalog_generation_prefers_leaf_recipe_class(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "swarm.py").write_text(
        """from nvflare.recipe.spec import Recipe

class BaseRecipe(Recipe):
    \"\"\"Base helper recipe.\"\"\"

class FinalWorkflow(BaseRecipe):
    \"\"\"Concrete exported recipe.\"\"\"

    def __init__(self):
        pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "pytorch"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    catalog = recipe_cli._discover_recipe_catalog()

    assert catalog[0]["class"] == "FinalWorkflow"
    assert catalog[0]["description"] == "Concrete exported recipe."


def test_recipe_catalog_generation_resolves_imported_recipe_base_alias(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "alias.py").write_text(
        """from nvflare.recipe.spec import Recipe as UnifiedBase

class AliasWorkflow(UnifiedBase):
    \"\"\"Recipe with an imported base alias.\"\"\"

    def __init__(self):
        pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "pytorch"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    catalog = recipe_cli._discover_recipe_catalog()

    assert catalog[0]["class"] == "AliasWorkflow"
    assert catalog[0]["description"] == "Recipe with an imported base alias."


def test_recipe_catalog_generation_resolves_recipe_ancestry_and_inherited_metadata(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "base.py").write_text(
        """from nvflare.recipe.spec import Recipe

class Workflow(Recipe):
    \"\"\"Inherited recipe description.\"\"\"

    recipe_notes = [\"Inherited note.\"]

    def __init__(self, *, num_rounds: int = 2):
        pass
""",
        encoding="utf-8",
    )
    (package_root / "child.py").write_text(
        """from .base import Workflow as Parent

class ConcreteWorkflow(Parent):
    pass
""",
        encoding="utf-8",
    )
    (package_root / "unrelated.py").write_text(
        """class LooksLikeRecipe:
    \"\"\"Not an NVFLARE recipe.\"\"\"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")

    recipe_class, description = recipe_cli._static_recipe_class("nvflare.fake.recipes.child")

    assert recipe_class.name == "ConcreteWorkflow"
    assert description == "Inherited recipe description."
    assert recipe_cli._static_recipe_class("nvflare.fake.recipes.unrelated") is None
    assert recipe_cli._static_recipe_attrs("nvflare.fake.recipes.child", "ConcreteWorkflow") == {
        "recipe_notes": ["Inherited note."]
    }
    assert {
        parameter["name"]
        for parameter in recipe_cli._static_recipe_parameters("nvflare.fake.recipes.child", "ConcreteWorkflow")
    } == {"num_rounds"}


def test_recipe_catalog_generation_prefers_explicit_literal_metadata_after_dynamic_alias(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "explicit.py").write_text(
        """from nvflare.recipe.spec import Recipe

class ExplicitMetadata(Recipe):
    algorithm = compute_algorithm()
    recipe_algorithm = \"kmeans\"
    aggregation = compute_aggregation()
    recipe_aggregation = \"cluster_centers\"
    state_exchange = compute_state_exchange()
    recipe_state_exchange = \"cluster_centers\"
    privacy = compute_privacy()
    recipe_privacy = {\"z_privacy\", \"a_privacy\"}

    def __init__(self):
        pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "pytorch"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    entry = recipe_cli._discover_recipe_catalog()[0]

    assert entry["algorithm"] == "kmeans"
    assert entry["aggregation"] == "cluster_centers"
    assert entry["state_exchange"] == "cluster_centers"
    assert entry["privacy"] == ["a_privacy", "z_privacy"]


def test_documented_recipe_specs_do_not_clear_omitted_discovered_metadata(monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    discovered = [
        {
            "name": "demo-pt",
            "description": "Discovered description.",
            "framework": "pytorch",
            "module": "nvflare.fake.demo",
            "class": "DemoWorkflow",
            "algorithm": "fedavg",
            "aggregation": "weighted_average",
            "state_exchange": "full_model",
            "privacy": ["differential_privacy"],
        }
    ]
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {"demo-pt": {"description": "Documented."}})

    entry = recipe_cli._apply_documented_recipe_specs(discovered)[0]

    assert entry["description"] == "Documented."
    assert entry["framework"] == "pytorch"
    assert entry["algorithm"] == "fedavg"
    assert entry["privacy"] == ["differential_privacy"]


def test_recipe_catalog_generation_preserves_static_recipe_attributes(tmp_path, monkeypatch):
    from nvflare.tool.recipe import recipe_cli

    package_root = tmp_path / "nvflare" / "fake" / "recipes"
    package_root.mkdir(parents=True)
    (package_root / "metadata.py").write_text(
        """from nvflare.recipe.spec import Recipe

class MetadataRecipe(Recipe):
    \"\"\"Recipe with explicit metadata.\"\"\"

    recipe_framework_support = {\"pytorch\", \"numpy\"}
    recipe_optional_dependencies = {\"pip install z-framework\", \"pip install a-framework\"}
    recipe_privacy = {\"z_privacy\", \"a_privacy\"}
    recipe_heterogeneity_support = [\"non_iid\"]
    recipe_privacy_compatible = [\"differential_privacy\"]
    recipe_notes = [\"Custom recipe note.\"]
    recipe_template_references = [\"nvflare/example/template\"]

    def __init__(self):
        pass
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(recipe_cli, "_NVFLARE_PACKAGE_ROOT", tmp_path / "nvflare")
    monkeypatch.setattr(
        recipe_cli,
        "_RECIPE_PACKAGE_ROOTS",
        [{"package": "nvflare.fake.recipes", "framework": "pytorch"}],
    )
    monkeypatch.setattr(recipe_cli, "_DOCUMENTED_RECIPE_SPECS", {})

    from nvflare.tool.recipe.generate_recipe_catalog import _render_catalog

    generated = json.loads(_render_catalog())["recipes"][0]

    assert "_recipe_attrs" not in generated["summary"]
    assert generated["summary"]["privacy"] == ["a_privacy", "z_privacy"]
    assert generated["detail"]["framework_support"] == ["numpy", "pytorch"]
    assert generated["detail"]["optional_dependencies"] == [
        "pip install a-framework",
        "pip install z-framework",
    ]
    assert generated["detail"]["heterogeneity_support"] == ["non_iid"]
    assert generated["detail"]["privacy_compatible"] == ["a_privacy", "differential_privacy", "z_privacy"]
    assert generated["detail"]["notes"] == ["Custom recipe note."]
    assert generated["detail"]["template_references"] == ["nvflare/example/template"]


def test_generated_recipe_catalog_is_current():
    from nvflare.tool.recipe.generate_recipe_catalog import _render_catalog
    from nvflare.tool.recipe.recipe_cli import _RECIPE_CATALOG_PATH

    assert (
        _RECIPE_CATALOG_PATH.read_text(encoding="utf-8") == _render_catalog()
    ), "recipe catalog is stale; run 'python -m nvflare.tool.recipe.generate_recipe_catalog'"


def test_recipe_cli_import_does_not_consume_recipe_export_args(monkeypatch):
    argv = ["nvflare", "job", "list", "--export", "--export-dir"]
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.delitem(sys.modules, "nvflare.tool.recipe.recipe_cli", raising=False)
    monkeypatch.delitem(sys.modules, "nvflare.recipe.spec", raising=False)

    importlib.import_module("nvflare.tool.recipe.recipe_cli")

    assert sys.argv == argv
