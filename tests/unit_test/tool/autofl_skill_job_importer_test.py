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

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from nvflare.app_common.widgets import intime_model_selector as ims

IMPORTER_MODULE_NAME = "nvflare_autofl_skill_job_importer"
_MISSING_MODULE = object()


def _load_importer():
    repo_root = Path(__file__).parents[3]
    importer_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "job_importer.py"
    spec = importlib.util.spec_from_file_location(IMPORTER_MODULE_NAME, importer_path)
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name, _MISSING_MODULE)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is _MISSING_MODULE:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return module


job_importer = _load_importer()
AUTOFL_CONFIG_SCHEMA_VERSION = job_importer.AUTOFL_CONFIG_SCHEMA_VERSION
DeterministicJobImporter = job_importer.DeterministicJobImporter
dump_autofl_yaml = job_importer.dump_autofl_yaml
import_job_to_autofl_config = job_importer.import_job_to_autofl_config


def test_load_importer_removes_temporary_runner_module(monkeypatch):
    monkeypatch.delitem(sys.modules, IMPORTER_MODULE_NAME, raising=False)

    _load_importer()

    assert IMPORTER_MODULE_NAME not in sys.modules


def test_load_importer_restores_cached_runner_module(monkeypatch):
    cached_module = object()
    monkeypatch.setitem(sys.modules, IMPORTER_MODULE_NAME, cached_module)

    loaded_importer = _load_importer()

    assert loaded_importer is not cached_module
    assert sys.modules[IMPORTER_MODULE_NAME] is cached_module


def test_load_importer_restores_cached_runner_module_when_execution_fails(monkeypatch):
    cached_module = object()
    original_spec_from_file_location = importlib.util.spec_from_file_location

    def failing_spec_from_file_location(*args, **kwargs):
        spec = original_spec_from_file_location(*args, **kwargs)

        def fail_execution(_module):
            raise RuntimeError("importer execution failed")

        spec.loader.exec_module = fail_execution
        return spec

    monkeypatch.setitem(sys.modules, IMPORTER_MODULE_NAME, cached_module)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", failing_spec_from_file_location)

    with pytest.raises(RuntimeError, match="importer execution failed"):
        _load_importer()
    assert sys.modules[IMPORTER_MODULE_NAME] is cached_module


def _objective(
    metric,
    source="user_request",
    *,
    mode="max",
    mode_source="core_default",
    job_metric=None,
    job_metric_source=None,
):
    job_metric = job_metric or metric
    job_metric_source = job_metric_source or source
    return {
        "metric": metric,
        "requested_metric": metric,
        "optimization_metric": metric,
        "metric_extraction_order": [metric],
        "mode": mode,
        "mode_contract_source": mode_source,
        "job_key_metric": job_metric,
        "job_key_metric_source": job_metric_source,
        "job_key_metric_mode": mode,
        "job_key_metric_mode_source": mode_source,
        "metric_contract_source": source,
        "metric_invariants": [
            "definition",
            "evaluation_data_and_split",
            "evaluation_timing_and_checkpoint",
            "aggregation_and_population",
            "scale_units_and_direction",
        ],
        "metric_change_policy": "restart_campaign_with_repaired_baseline",
    }


def _write_recipe_job(root):
    (root / "model.py").write_text(
        """
class SimpleNetwork:
    pass
""",
        encoding="utf-8",
    )
    (root / "client.py").write_text(
        """
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    return parser
""",
        encoding="utf-8",
    )
    (root / "job.py").write_text(
        """
import argparse

from model import SimpleNetwork
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument("--key_metric", type=str, default="accuracy")
    return parser.parse_args()


def main():
    args = define_parser()
    recipe = FedAvgRecipe(
        name="demo",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        model=SimpleNetwork(),
        train_script=args.train_script,
        key_metric=args.key_metric,
    )
    env = SimEnv(num_clients=args.n_clients)
    recipe.execute(env)


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    return root / "job.py"


def test_import_recipe_job_extracts_trust_contract_without_executing_code(tmp_path):
    job_path = _write_recipe_job(tmp_path)

    config = import_job_to_autofl_config(
        str(job_path),
        workspace_root=str(tmp_path),
        metric="AUC",
        target_env="prod",
        max_candidates=8,
    )

    assert config["schema_version"] == AUTOFL_CONFIG_SCHEMA_VERSION
    assert config["import"]["support"]["patterns"] == ["recipe:FedAvgRecipe", "env:SimEnv"]
    assert config["import"]["confidence"] == "high"
    assert config["job"]["surface"] == "recipe"
    assert config["job"]["recipe"] == "FedAvgRecipe"
    assert config["job"]["train_script"] == "client.py"
    assert config["objective"] == _objective("AUC", job_metric="accuracy", job_metric_source="arg:key_metric")
    assert config["budget"]["max_candidates"] == 8
    assert config["budget"]["fixed_training_budget"] == {
        "num_rounds": 5,
        "min_clients": 3,
        "num_clients": 3,
    }
    assert config["environment"]["requested"] == "prod"
    assert config["environment"]["profiles"]["sim"] == {"num_clients": 3}
    assert config["environment"]["simulator_env_passthrough"] == []
    assert config["search_space"]["suggested"]["lr"]["default"] == 0.01
    assert config["search_space"]["suggested"]["batch_size"]["type"] == "int"
    assert config["trust_contract"]["allowed_edit_paths"] == ["job.py", "client.py", "model.py"]
    assert config["trust_contract"]["allowed_create_patterns"] == ["**/*.py"]
    assert "allowed_edit_paths" not in config["job"]
    assert "allowed_create_patterns" not in config["job"]
    assert config["trust_contract"]["agent_controls"]["must_not_edit_outside_allowed_paths"] is True
    assert config["unresolved"] == []


def test_import_resolves_explicit_minimization_mode(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,", 'key_metric=args.key_metric,\n        key_metric_mode="min",'
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "min"
    assert config["objective"]["mode_contract_source"] == "job:key_metric_mode"


def test_import_infers_mode_from_same_metric_stop_condition(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,", 'key_metric=args.key_metric,\n        stop_cond="accuracy <= 0.2",'
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "min"
    assert config["objective"]["mode_contract_source"] == "job:stop_cond"


def test_import_resolves_argparse_metric_mode_override(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = (
        job_path.read_text(encoding="utf-8")
        .replace(
            'parser.add_argument("--key_metric", type=str, default="accuracy")',
            'parser.add_argument("--key_metric", type=str, default="accuracy")\n'
            '    parser.add_argument("--key_metric_mode", choices=["min", "max"], default="max")',
        )
        .replace(
            "key_metric=args.key_metric,",
            "key_metric=args.key_metric,\n        key_metric_mode=args.key_metric_mode,",
        )
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(
        str(job_path), workspace_root=str(tmp_path), job_args=["--key_metric_mode", "min"]
    )

    assert config["objective"]["mode"] == "min"
    assert config["objective"]["mode_contract_source"] == "arg:key_metric_mode"


def test_import_marks_dynamic_metric_mode_unresolved(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,",
        "key_metric=args.key_metric,\n        key_metric_mode=get_metric_mode(),",
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "max"
    assert config["objective"]["mode_contract_source"] == "unresolved"
    assert {item["field"] for item in config["unresolved"]} >= {"objective.mode"}


@pytest.mark.parametrize("mode_literal", ['["min"]', '("min",)'])
def test_import_marks_non_string_metric_mode_unresolved(tmp_path, mode_literal):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,",
        f"key_metric=args.key_metric,\n        key_metric_mode={mode_literal},",
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "max"
    assert config["objective"]["mode_contract_source"] == "unresolved"
    assert any(
        item["field"] == "objective.mode" and "invalid key_metric_mode" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_marks_conflicting_metric_direction_declarations_unresolved(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,",
        'key_metric=args.key_metric,\n        key_metric_mode="max",\n        stop_cond="accuracy <= 0.2",',
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "max"
    assert any("conflicts" in item["reason"] for item in config["unresolved"] if item["field"] == "objective.mode")


def test_import_marks_custom_model_selector_direction_unresolved(tmp_path):
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.widgets.widget import Widget


class CustomSelector(Widget):
    pass


job = BaseFedJob(
    name="custom-selector",
    min_clients=2,
    key_metric="loss",
    key_metric_mode="min",
    model_selector=CustomSelector(),
)
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "max"
    assert config["objective"]["mode_contract_source"] == "unresolved"
    assert any(
        item["field"] == "objective.mode" and "custom model_selector" in item["reason"] for item in config["unresolved"]
    )


def test_import_allows_explicitly_absent_model_selector(tmp_path):
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob


job = BaseFedJob(
    name="default-selector",
    min_clients=2,
    key_metric="loss",
    key_metric_mode="min",
    model_selector=None,
)
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "min"
    assert config["objective"]["mode_contract_source"] == "job:key_metric_mode"


def test_import_marks_splatted_key_metric_unresolved(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


parser = argparse.ArgumentParser()
parser.add_argument("--key_metric", default="val_loss")
args = parser.parse_args()
tuning = {"key_metric": args.key_metric}
recipe = FedAvgRecipe(
    name="splatted-metric", min_clients=2, num_rounds=1, train_script="client.py", **tuning
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["metric"] == "accuracy"
    assert config["objective"]["metric_contract_source"] == "default"
    assert any(
        item["field"] == "objective.metric" and "job call passes **kwargs" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_marks_splatted_direction_keywords_unresolved_with_explicit_metric(tmp_path):
    tmp_path.joinpath("train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.widgets.widget import Widget


class CustomSelector(Widget):
    pass


extra = {"model_selector": CustomSelector(), "key_metric_mode": "min"}
job = BaseFedJob(name="splatted-direction", min_clients=2, key_metric="brier", **extra)
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["metric"] == "brier"
    assert config["objective"]["metric_contract_source"] == "literal"
    assert config["objective"]["mode"] == "max"
    assert config["objective"]["mode_contract_source"] == "unresolved"
    assert any(
        item["field"] == "objective.mode" and "job call passes **kwargs" in item["reason"]
        for item in config["unresolved"]
    )
    assert not any(item["field"] == "objective.metric" for item in config["unresolved"])


def test_import_rejects_direction_contract_when_job_call_has_keyword_splat(tmp_path):
    tmp_path.joinpath("train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob


other = {"analytics_receiver": None}
job = BaseFedJob(
    name="explicit-direction",
    min_clients=2,
    key_metric="loss",
    key_metric_mode="min",
    model_selector=None,
    **other,
)
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"]["mode"] == "max"
    assert config["objective"]["mode_contract_source"] == "unresolved"
    assert any(
        item["field"] == "objective.mode" and "remove **kwargs" in item["reason"] for item in config["unresolved"]
    )


@pytest.mark.parametrize(
    "constructor_args",
    [
        '"positional", 2, None, "loss", "min"',
        "*job_args",
    ],
)
def test_import_marks_positional_job_arguments_unresolved(tmp_path, constructor_args):
    tmp_path.joinpath("train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
from nvflare.app_common.executors.script_runner import ScriptRunner
from nvflare.job_config.base_fed_job import BaseFedJob


job_args = ("positional", 2, None, "loss", "min")
job = BaseFedJob({constructor_args})
runner = ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    unresolved_fields = {item["field"] for item in config["unresolved"]}
    assert {"objective.metric", "objective.mode", "budget.fixed_training_budget"} <= unresolved_fields
    positional_reasons = " ".join(item["reason"] for item in config["unresolved"])
    expected_argument_kind = "*args" if constructor_args.startswith("*") else "positional arguments"
    assert expected_argument_kind in positional_reasons


def test_import_marks_splatted_job_budget_unresolved(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


budget = {"num_rounds": 3}
recipe = FedAvgRecipe(
    name="splatted-budget",
    min_clients=2,
    train_script="client.py",
    key_metric="accuracy",
    key_metric_mode="max",
    **budget,
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_rounds" not in config["budget"]["fixed_training_budget"]
    assert any(
        item["field"] == "budget.fixed_training_budget" and "job call passes **kwargs" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_marks_splatted_sim_env_client_budget_unresolved(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


recipe = FedAvgRecipe(name="sim-budget", min_clients=2, num_rounds=1, train_script="client.py")
sim_budget = {"num_clients": 3}
recipe.execute(SimEnv(**sim_budget))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_clients" not in config["budget"]["fixed_training_budget"]
    assert any(
        item["field"] == "budget.fixed_training_budget.num_clients" and "SimEnv call passes **kwargs" in item["reason"]
        for item in config["unresolved"]
    )


@pytest.mark.parametrize(
    "sim_env_args",
    [
        'clients=["site-1", "site-2", "site-3"]',
        'num_clients=0, clients=["site-1", "site-2", "site-3"]',
    ],
)
def test_import_resolves_sim_env_client_budget_from_static_clients(tmp_path, sim_env_args):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


recipe = FedAvgRecipe(name="sim-budget", min_clients=2, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv({sim_env_args}))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["budget"]["fixed_training_budget"]["num_clients"] == 3
    assert config["environment"]["profiles"]["sim"]["num_clients"] == 3
    assert not any(item["field"] == "budget.fixed_training_budget.num_clients" for item in config["unresolved"])


def test_import_selects_conditional_sim_env_from_job_args(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


parser = argparse.ArgumentParser()
parser.add_argument("--environment", choices=["small", "large"], default="small")
args = parser.parse_args()

recipe = FedAvgRecipe(name="sim-budget", min_clients=2, num_rounds=1, train_script="client.py")
if args.environment == "small":
    recipe.execute(SimEnv(num_clients=2))
else:
    recipe.execute(SimEnv(num_clients=5))
""".lstrip(),
        encoding="utf-8",
    )

    default_config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))
    large_config = import_job_to_autofl_config(
        str(job_path), workspace_root=str(tmp_path), job_args=["--environment", "large"]
    )

    assert default_config["budget"]["fixed_training_budget"]["num_clients"] == 2
    assert default_config["environment"]["profiles"]["sim"]["num_clients"] == 2
    assert large_config["budget"]["fixed_training_budget"]["num_clients"] == 5
    assert large_config["environment"]["profiles"]["sim"]["num_clients"] == 5


@pytest.mark.parametrize(("job_args", "expected_num_clients"), [([], 1), (["--use-two-clients"], 2)])
def test_import_resolves_sim_env_clients_from_selected_assignment_branch(tmp_path, job_args, expected_num_clients):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


parser = argparse.ArgumentParser()
parser.add_argument("--use-two-clients", action="store_true")
args = parser.parse_args()

clients = ["site-1"]
if args.use_two_clients:
    clients = ["site-1", "site-2"]

recipe = FedAvgRecipe(name="sim-budget", min_clients=1, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv(clients=clients))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path), job_args=job_args)

    assert config["budget"]["fixed_training_budget"]["num_clients"] == expected_num_clients
    assert config["environment"]["profiles"]["sim"]["num_clients"] == expected_num_clients


def test_import_rejects_sim_env_clients_from_unknown_assignment_branch(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def use_two_clients():
    return False


clients = ["site-1"]
if use_two_clients():
    clients = ["site-1", "site-2"]

recipe = FedAvgRecipe(name="sim-budget", min_clients=1, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv(clients=clients))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_clients" not in config["budget"]["fixed_training_budget"]
    assert "num_clients" not in config["environment"]["profiles"]["sim"]
    assert any(
        item["field"] == "budget.fixed_training_budget.num_clients"
        and "conditional assignment for clients" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_rejects_zero_sim_env_client_count_with_splatted_clients(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


recipe = FedAvgRecipe(name="sim-budget", min_clients=2, num_rounds=1, train_script="client.py")
sim_args = {"clients": ["site-1", "site-2"]}
recipe.execute(SimEnv(num_clients=0, **sim_args))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_clients" not in config["budget"]["fixed_training_budget"]
    assert "num_clients" not in config["environment"]["profiles"]["sim"]
    assert any(
        item["field"] == "budget.fixed_training_budget.num_clients" and "SimEnv call passes **kwargs" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_rejects_dynamic_sim_env_clients(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def get_clients():
    return ["site-1", "site-2"]


recipe = FedAvgRecipe(name="sim-budget", min_clients=2, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv(clients=get_clients()))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_clients" not in config["budget"]["fixed_training_budget"]
    assert any(
        item["field"] == "budget.fixed_training_budget.num_clients" and "clients is dynamic" in item["reason"]
        for item in config["unresolved"]
    )


def test_import_marks_malformed_stop_condition_unresolved(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        "key_metric=args.key_metric,", 'key_metric=args.key_metric,\n        stop_cond="accuracy <= target",'
    )
    job_path.write_text(source, encoding="utf-8")

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert any("invalid stop_cond" in item["reason"] for item in config["unresolved"])


def test_import_is_repeatable_and_yaml_round_trips(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    importer = DeterministicJobImporter(workspace_root=str(tmp_path))

    first = importer.import_job(str(job_path), max_candidates=4)
    second = importer.import_job(str(job_path), max_candidates=4)
    yaml_text = dump_autofl_yaml(first)

    assert first == second
    assert DeterministicJobImporter.dump_yaml is dump_autofl_yaml
    assert importer.dump_yaml(first) == yaml_text
    assert yaml.safe_load(yaml_text) == first
    assert "&id" not in yaml_text
    assert first["trust_contract"]["unresolved"] is not first["unresolved"]


def test_import_marks_dynamic_argparse_defaults_unresolved(tmp_path):
    (tmp_path / "client.py").write_text(
        """
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, default=DEFAULT_MODEL_ARCH)
    return parser
""",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--train_script", type=str, default="client.py")
    return parser.parse_args()


def main():
    args = define_parser()
    recipe = FedAvgRecipe(
        name="demo",
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        train_script=args.train_script,
    )
    recipe.execute(SimEnv(num_clients=args.n_clients))
""",
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    model_arch = config["search_space"]["suggested"]["model_arch"]
    assert model_arch["default"] == "DEFAULT_MODEL_ARCH"
    assert model_arch["confidence"] == "low"
    assert model_arch["unresolved"] is True
    assert config["import"]["confidence"] == "medium"
    assert {
        "field": "search_space.suggested.model_arch.default",
        "reason": "default is dynamic expression: DEFAULT_MODEL_ARCH",
    } in config["unresolved"]


def test_import_marks_dynamic_train_script_unresolved_without_client_fallback(tmp_path):
    (tmp_path / "client.py").write_text(
        """
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    return parser
""",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def get_script():
    return "client.py"


def main():
    recipe = FedAvgRecipe(
        name="demo",
        min_clients=2,
        num_rounds=3,
        train_script=get_script(),
    )
    recipe.execute(SimEnv(num_clients=2))
""",
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "train_script" not in config["job"]
    assert "client.py" not in config["trust_contract"]["allowed_edit_paths"]
    assert {"field": "job.train_script", "reason": "no train_script was found or resolved"} in config["unresolved"]


@pytest.mark.parametrize(
    "expression,expected",
    [
        ('Path("configs/train.py")', "configs/train.py"),
        ('os.path.join("src", args.train_script)', "src/train.py"),
        ('Path("src") / args.train_script', "src/train.py"),
    ],
)
def test_import_resolves_composed_train_script_paths(tmp_path, expression, expected):
    target = tmp_path / expected
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
import argparse
import os
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

parser = argparse.ArgumentParser()
parser.add_argument("--train_script", default="train.py")
args = parser.parse_args()
recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=3, train_script={expression})
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["job"]["train_script"] == expected
    assert expected in config["trust_contract"]["allowed_edit_paths"]


def test_import_surfaces_positional_tunable_as_source_edit_only(tmp_path):
    (tmp_path / "client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

parser = argparse.ArgumentParser()
parser.add_argument("epochs", type=int)
parser.parse_args()
recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=3, train_script="client.py")
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    epochs = config["search_space"]["suggested"]["epochs"]
    assert epochs["mutable_via_run_args"] is False
    assert epochs["unresolved"] is True
    assert {
        "field": "search_space.suggested.epochs.interface",
        "reason": "positional argparse fields require source edits; candidate run_args support long options only",
    } in config["unresolved"]


def test_import_marks_imported_budget_and_metric_constants_unresolved(tmp_path):
    (tmp_path / "client.py").write_text(
        """
def train():
    pass
""",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from config import KEY_METRIC, NUM_ROUNDS
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def main():
    recipe = FedAvgRecipe(
        name="demo",
        min_clients=2,
        num_rounds=NUM_ROUNDS,
        train_script="client.py",
        key_metric=KEY_METRIC,
    )
    recipe.execute(SimEnv(num_clients=2))
""",
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"] == _objective("accuracy", source="default")
    assert config["budget"]["fixed_training_budget"] == {"min_clients": 2, "num_clients": 2}
    assert {
        "field": "budget.fixed_training_budget.num_rounds",
        "reason": "name:NUM_ROUNDS",
    } in config["unresolved"]
    assert {"field": "objective.metric", "reason": "name:KEY_METRIC"} in config["unresolved"]
    assert {"field": "job.FedAvgRecipe.key_metric", "reason": "name:KEY_METRIC"} in config["unresolved"]
    assert {"field": "job.FedAvgRecipe.num_rounds", "reason": "name:NUM_ROUNDS"} in config["unresolved"]


def test_import_marks_call_expression_budget_and_metric_unresolved(tmp_path):
    (tmp_path / "client.py").write_text(
        """
def train():
    pass
""",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def get_metric():
    return "accuracy"


def get_rounds():
    return 5


def main():
    recipe = FedAvgRecipe(
        name="demo",
        min_clients=2,
        num_rounds=get_rounds(),
        train_script="client.py",
        key_metric=get_metric(),
    )
    recipe.execute(SimEnv(num_clients=2))
""",
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["objective"] == _objective("accuracy", source="default")
    assert config["budget"]["fixed_training_budget"] == {"min_clients": 2, "num_clients": 2}
    assert {
        "field": "budget.fixed_training_budget.num_rounds",
        "reason": "call:get_rounds",
    } in config["unresolved"]
    assert {"field": "objective.metric", "reason": "call:get_metric"} in config["unresolved"]
    assert {"field": "job.FedAvgRecipe.key_metric", "reason": "call:get_metric"} in config["unresolved"]
    assert {"field": "job.FedAvgRecipe.num_rounds", "reason": "call:get_rounds"} in config["unresolved"]
    assert config["job"]["recipe_args"]["num_rounds"] == {
        "value": "get_rounds()",
        "source": "call:get_rounds",
        "confidence": "low",
    }


def test_import_marks_unsupported_custom_job_as_partial(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
def main():
    run_custom_workflow()


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["import"]["support"]["status"] == "partial"
    assert config["job"]["surface"] == "unknown"
    unresolved_fields = {item["field"] for item in config["unresolved"]}
    assert "job.surface" in unresolved_fields
    assert "job.train_script" in unresolved_fields
    assert "budget.fixed_training_budget" in unresolved_fields


def test_import_resolves_nvflare_fed_job_alias_and_script_runner(tmp_path):
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.workflows.fedavg import FedAvgJob as ImportedJob
from nvflare.app_common.executors.script_runner import ScriptRunner as Runner


def main():
    job = ImportedJob(
        name="fedavg-alias",
        n_clients=8,
        min_clients=4,
        num_rounds=10,
        key_metric="loss",
        key_metric_mode="min",
    )
    runner = Runner(script="train.py")
    return job, runner
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["import"]["support"]["status"] == "supported"
    assert config["job"]["fed_job"] == "FedAvgJob"
    assert config["job"]["fed_job_class"] == "nvflare.app_common.workflows.fedavg.FedAvgJob"
    assert config["job"]["train_script"] == "train.py"
    assert config["objective"] == _objective("loss", source="literal", mode="min", mode_source="job:key_metric_mode")
    assert config["budget"]["fixed_training_budget"] == {
        "num_rounds": 10,
        "min_clients": 4,
        "num_clients": 8,
    }


def test_import_resolves_module_aliases_for_nvflare_job_subclasses(tmp_path):
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import nvflare.app_common.workflows as workflows
import nvflare.app_common.executors.script_runner as runner_module


def main():
    job = workflows.CCWFJob(name="ccwf", n_clients=3, min_clients=2, num_rounds=4)
    runner = runner_module.ScriptRunner(script="train.py")
    return job, runner
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["job"]["fed_job"] == "CCWFJob"
    assert config["job"]["fed_job_class"] == "nvflare.app_common.workflows.CCWFJob"
    assert config["job"]["train_script"] == "train.py"
    assert config["budget"]["fixed_training_budget"]["num_clients"] == 3


def test_import_recognizes_future_nvflare_job_subclasses_but_not_generic_or_local_jobs(tmp_path):
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    stats_job = tmp_path / "stats_job.py"
    stats_job.write_text(
        """
from nvflare.app_common.workflows import StatsJob
from nvflare.app_common.executors.script_runner import ScriptRunner

StatsJob(name="stats", n_clients=2, min_clients=2, num_rounds=1)
ScriptRunner(script="train.py")
""".lstrip(),
        encoding="utf-8",
    )
    assert import_job_to_autofl_config(str(stats_job), workspace_root=str(tmp_path))["job"]["fed_job"] == "StatsJob"

    local_job = tmp_path / "local_job.py"
    local_job.write_text(
        """
class CustomJob:
    pass

CustomJob()
""".lstrip(),
        encoding="utf-8",
    )
    local_config = import_job_to_autofl_config(str(local_job), workspace_root=str(tmp_path))
    assert local_config["import"]["support"]["status"] == "partial"
    assert "local or non-NVFlare Job subclass" in next(
        item["reason"] for item in local_config["unresolved"] if item["field"] == "job.surface"
    )

    generic_job = tmp_path / "generic_job.py"
    generic_job.write_text("from nvflare.apis.job_def import Job\nJob()\n", encoding="utf-8")
    generic_config = import_job_to_autofl_config(str(generic_job), workspace_root=str(tmp_path))
    assert generic_config["import"]["support"]["status"] == "partial"


def test_import_leaves_multiple_script_runners_unresolved(tmp_path):
    for name in ("train_a.py", "train_b.py"):
        tmp_path.joinpath(name).write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_common.workflows import EdgeJob
from nvflare.app_common.executors.script_runner import ScriptRunner

EdgeJob(name="edge", n_clients=2, num_rounds=1)
ScriptRunner(script="train_a.py")
ScriptRunner(script="train_b.py")
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "train_script" not in config["job"]
    assert {"field": "job.train_script", "reason": "no train_script was found or resolved"} in config["unresolved"]


def test_import_keeps_async_function_assignments_out_of_module_scope(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

NUM_ROUNDS = 3

async def helper():
    NUM_ROUNDS = 99
    return NUM_ROUNDS

FedAvgRecipe(name="demo", model=object(), num_rounds=NUM_ROUNDS, min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["budget"]["fixed_training_budget"]["num_rounds"] == 3


def test_import_marks_augmented_budget_assignment_unresolved(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

NUM_ROUNDS = 3
NUM_ROUNDS += 97
FedAvgRecipe(name="demo", model=object(), num_rounds=NUM_ROUNDS, min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "num_rounds" not in config["budget"]["fixed_training_budget"]
    assert any(item["field"] == "budget.fixed_training_budget.num_rounds" for item in config["unresolved"])


def test_import_ignores_add_argument_on_non_argparse_objects(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

class Registry:
    def add_argument(self, *args, **kwargs):
        pass

registry = Registry()
registry.add_argument("--num_rounds", default=99)
parser = argparse.ArgumentParser()
parser.add_argument("--num_rounds", type=int, default=4)
args = parser.parse_args()
FedAvgRecipe(name="demo", model=object(), num_rounds=args.num_rounds, min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["budget"]["fixed_training_budget"]["num_rounds"] == 4


def test_import_does_not_admit_local_recipe_classes(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
class MyRecipe:
    pass

MyRecipe(name="local", num_rounds=1, min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["import"]["support"]["status"] == "partial"
    assert config["job"]["surface"] == "unknown"


@pytest.mark.parametrize(
    "recipe_name",
    ["FedEvalRecipe", "FedStatsRecipe", "NumpyCrossSiteEvalRecipe"],
)
def test_import_refuses_non_optimization_recipes_before_execution(tmp_path, recipe_name):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
from nvflare.recipe import {recipe_name}

{recipe_name}(name="not-training", min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    support = config["import"]["support"]
    assert support["status"] == "partial"
    assert "no training loop" in support["reason"]
    assert any(item["field"] == "job.surface" for item in config["unresolved"])


def test_import_refuses_nested_flower_recipe_before_execution(tmp_path):
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
from nvflare.app_opt.flower.recipe import FlowerRecipe

FlowerRecipe(name="flower", flower_content="app", min_clients=2)
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    support = config["import"]["support"]
    assert support["status"] == "partial"
    assert "nested application" in support["reason"]


def test_import_selects_conditional_training_recipe_from_job_args(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse

from nvflare.app_common.np.recipes import NumpyCrossSiteEvalRecipe, NumpyFedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrained", "training"], default="pretrained")
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=1)
    return parser.parse_args()


def run_eval(n_clients):
    NumpyCrossSiteEvalRecipe(name="eval", min_clients=n_clients)
    SimEnv(num_clients=n_clients)


def run_training(n_clients, num_rounds):
    NumpyFedAvgRecipe(
        name="training",
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script="client.py",
    )
    SimEnv(num_clients=n_clients)


def main():
    args = define_parser()
    if args.mode == "pretrained":
        run_eval(args.n_clients)
    elif args.mode == "training":
        run_training(args.n_clients, args.num_rounds)


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )

    default_config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))
    training_config = import_job_to_autofl_config(
        str(job_path),
        workspace_root=str(tmp_path),
        job_args=["--mode", "training", "--num_rounds=3"],
    )

    assert default_config["job"]["recipe"] == "NumpyCrossSiteEvalRecipe"
    assert default_config["import"]["support"]["status"] == "partial"
    assert training_config["job"]["recipe"] == "NumpyFedAvgRecipe"
    assert training_config["job"]["train_script"] == "client.py"
    assert training_config["import"]["support"]["status"] == "supported"
    assert training_config["budget"]["fixed_training_budget"] == {
        "num_rounds": 3,
        "min_clients": 2,
        "num_clients": 2,
    }


@pytest.mark.parametrize(
    ("example", "expected_status", "expected_recipe"),
    [
        ("hello-cyclic", "supported", "CyclicRecipe"),
        ("hello-lightning", "supported", "FedAvgRecipe"),
        ("hello-lr", "supported", "FedAvgLrRecipe"),
        ("hello-flower", "partial", "FlowerRecipe"),
        ("hello-lightning-eval", "partial", "FedEvalRecipe"),
        ("hello-numpy-cross-val", "partial", "NumpyCrossSiteEvalRecipe"),
        ("hello-tabular-stats", "partial", "FedStatsRecipe"),
        ("hello-tf", "supported", "FedAvgRecipe"),
    ],
)
def test_import_classifies_hello_world_release_gate_examples(example, expected_status, expected_recipe):
    repo_root = Path(__file__).parents[3]
    example_root = repo_root / "examples" / "hello-world" / example

    config = import_job_to_autofl_config(
        str(example_root / "job.py"),
        workspace_root=str(example_root),
        metric="accuracy",
        max_candidates=12,
    )

    assert config["import"]["support"]["status"] == expected_status
    assert config["job"]["recipe"] == expected_recipe


@pytest.mark.parametrize(
    ("algorithm", "expected_recipe"),
    [("fedavg", "FedAvgRecipe"), ("fedprox", "FedProxRecipe"), ("scaffold", "ScaffoldRecipe")],
)
def test_import_selects_hello_lightning_algorithm_mode(algorithm, expected_recipe):
    repo_root = Path(__file__).parents[3]
    example_root = repo_root / "examples" / "hello-world" / "hello-lightning"

    config = import_job_to_autofl_config(
        str(example_root / "job.py"),
        workspace_root=str(example_root),
        max_candidates=12,
        job_args=["--algorithm", algorithm],
    )

    assert config["import"]["support"]["status"] == "supported"
    assert config["job"]["recipe"] == expected_recipe
    assert config["job"]["train_script"] == "client.py"
    assert config["objective"] == _objective("accuracy", source="core_default")


def test_import_selects_hello_numpy_cross_val_training_mode():
    repo_root = Path(__file__).parents[3]
    example_root = repo_root / "examples" / "hello-world" / "hello-numpy-cross-val"

    config = import_job_to_autofl_config(
        str(example_root / "job.py"),
        workspace_root=str(example_root),
        metric="weight_mean",
        max_candidates=12,
        job_args=["--mode", "training"],
    )

    assert config["import"]["support"]["status"] == "supported"
    assert config["job"]["recipe"] == "NumpyFedAvgRecipe"
    assert config["job"]["train_script"] == "client.py"


def test_import_returns_clean_error_for_missing_job(tmp_path):
    with pytest.raises(job_importer.JobImportError, match="job.py not found"):
        import_job_to_autofl_config(str(tmp_path / "missing.py"), workspace_root=str(tmp_path))


def test_conflicting_reachable_argparse_destinations_are_unresolved_until_explicitly_overridden(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", dest="num_rounds", type=int, default=1)
    parser.add_argument("--rounds", dest="num_rounds", type=int, default=2)
    return parser.parse_args()


def main():
    args = define_parser()
    recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=args.num_rounds, train_script="client.py")
    recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    ambiguous = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))
    explicit = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path), job_args=["--rounds", "7"])

    assert any(
        item["field"] == "budget.fixed_training_budget.num_rounds"
        and "conflicting argparse definitions" in item["reason"]
        for item in ambiguous["unresolved"]
    )
    assert "num_rounds" not in ambiguous["budget"]["fixed_training_budget"]
    assert explicit["budget"]["fixed_training_budget"]["num_rounds"] == 7
    assert not any(item["field"] == "budget.fixed_training_budget.num_rounds" for item in explicit["unresolved"])


@pytest.mark.parametrize(
    ("argument", "job_args"),
    [
        ('parser.add_argument("--x", dest=["y"], default=1)', []),
        ('parser.add_argument("--flagx", action=["store_true"])', ["--flagx"]),
    ],
)
def test_import_ignores_malformed_argparse_dest_and_action_literals(tmp_path, argument, job_args):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    {argument}
    return parser.parse_args()


def main():
    define_parser()
    recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=1, train_script="client.py")
    recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path), job_args=job_args)

    assert config["import"]["support"]["status"] == "supported"


def test_import_keeps_flags_derived_name_for_dynamic_argparse_dest(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def make_name():
    return "num_rounds"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", dest=make_name(), type=int, default=3)
    return parser.parse_args()


def main():
    args = define_parser()
    recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=args.num_rounds, train_script="client.py")
    recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["budget"]["fixed_training_budget"]["num_rounds"] == 3


def test_dynamic_key_metric_uses_documented_fallback_until_explicitly_overridden(tmp_path):
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        """
import argparse
import os
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_metric", default=os.environ.get("METRIC"))
    return parser.parse_args()


def main():
    args = define_parser()
    recipe = FedAvgRecipe(
        name="demo", min_clients=2, num_rounds=1, train_script="client.py", key_metric=args.key_metric
    )
    recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    fallback = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))
    explicit = import_job_to_autofl_config(
        str(job_path), workspace_root=str(tmp_path), job_args=["--key_metric", "val_auc"]
    )
    requested = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path), metric="accuracy")

    assert fallback["objective"]["metric"] == "accuracy"
    assert fallback["objective"]["metric_contract_source"] == "default"
    assert any(item["field"] == "objective.metric" for item in fallback["unresolved"])
    assert explicit["objective"]["metric"] == "val_auc"
    assert explicit["objective"]["metric_contract_source"] == "arg:key_metric"
    assert any(item["field"] == "objective.job_key_metric" for item in requested["unresolved"])
    assert not any(item["field"] == "objective.metric" for item in requested["unresolved"])


@pytest.mark.parametrize(
    "metric",
    [
        "trainingloss",
        "val_mseloss",
        "neg_loss",
        "negative_class_loss",
        "dice",
        "-mse",
        "neg.mse",
        "cost",
        "divergence",
        "energy",
    ],
)
def test_lower_is_better_metric_heuristic_matches_nvflare_core(metric):
    assert job_importer._fallback_looks_lower_is_better(metric) is ims._looks_lower_is_better(metric)


def test_lower_is_better_metric_fallback_matches_core_across_token_boundaries():
    metrics = {"mse2", "2mse", "neg_mse2", "negative_mse2"}
    for token in ims._LOWER_IS_BETTER_TOKEN_HINTS:
        metrics.update(
            {
                token,
                f"val_{token}",
                f"val-{token}",
                f"{token}.score",
                f"{token}2",
                f"2{token}",
                f"{token}_2",
                f"2_{token}",
                f"neg_{token}",
                f"negative_{token}",
            }
        )

    for metric in metrics:
        assert job_importer._fallback_looks_lower_is_better(metric) is ims._looks_lower_is_better(metric), metric


def test_core_lower_is_better_metric_heuristic_is_resolved_and_cached(monkeypatch):
    calls = []

    def core_heuristic(metric):
        calls.append(metric)
        return False

    monkeypatch.setattr(ims, "_looks_lower_is_better", core_heuristic)
    monkeypatch.setattr(job_importer, "_core_looks_lower_is_better", job_importer._UNRESOLVED)

    assert job_importer.likely_lower_is_better_metric("val_loss") is False
    assert job_importer._core_looks_lower_is_better is core_heuristic
    assert job_importer.likely_lower_is_better_metric("val_loss") is False
    assert calls == ["val_loss", "val_loss"]


@pytest.mark.parametrize("metric", ["dice", "neg_loss"])
def test_lower_is_better_dispatcher_uses_fallback_for_false_metrics(monkeypatch, metric):
    monkeypatch.setattr(job_importer, "_core_looks_lower_is_better", None)

    assert job_importer.likely_lower_is_better_metric(metric) is False


def test_fallback_lower_is_better_metric_hints_match_nvflare_core():
    assert set(job_importer.LOWER_IS_BETTER_METRIC_SUBSTRINGS) == set(ims._LOWER_IS_BETTER_SUBSTRING_HINTS)
    assert job_importer.LOWER_IS_BETTER_METRIC_TOKENS == ims._LOWER_IS_BETTER_TOKEN_HINTS
    assert job_importer.ALREADY_NEGATED_METRIC_TOKEN == ims._ALREADY_NEGATED_TOKEN


@pytest.mark.parametrize("import_error", [ImportError, RuntimeError])
def test_importer_loads_and_imports_job_without_nvflare_in_agent_environment(
    tmp_path, monkeypatch, caplog, import_error
):
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nvflare.app_common.widgets.intime_model_selector":
            raise import_error("NVFlare is unavailable in the agent environment")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    isolated_importer = _load_importer()
    job_path = _write_recipe_job(tmp_path)

    config = isolated_importer.import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert isolated_importer._core_looks_lower_is_better is isolated_importer._UNRESOLVED
    assert caplog.records == []
    with caplog.at_level("WARNING", logger=isolated_importer.__name__):
        assert isolated_importer.likely_lower_is_better_metric("val_loss") is True
        assert isolated_importer.likely_lower_is_better_metric("val_loss") is True
    assert isolated_importer._core_looks_lower_is_better is isolated_importer._UNRESOLVED
    assert len(caplog.records) == 1
    assert f"{import_error.__name__}: NVFlare is unavailable in the agent environment" in caplog.text
    assert "retrying on the next check" in caplog.text
    assert config["import"]["support"]["status"] == "supported"


def test_core_heuristic_resolution_retries_after_transient_import_failure(monkeypatch, caplog):
    original_import = builtins.__import__
    import_attempts = 0
    core_calls = []

    def core_heuristic(metric):
        core_calls.append(metric)
        return False

    def flaky_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal import_attempts
        if name == "nvflare.app_common.widgets.intime_model_selector":
            import_attempts += 1
            if import_attempts == 1:
                raise RuntimeError("transient NVFlare initialization failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(ims, "_looks_lower_is_better", core_heuristic)
    monkeypatch.setattr(builtins, "__import__", flaky_import)
    isolated_importer = _load_importer()

    with caplog.at_level("WARNING", logger=isolated_importer.__name__):
        assert isolated_importer.likely_lower_is_better_metric("val_loss") is True
    assert isolated_importer._core_looks_lower_is_better is isolated_importer._UNRESOLVED
    assert isolated_importer.likely_lower_is_better_metric("val_loss") is False
    assert isolated_importer._core_looks_lower_is_better is core_heuristic
    assert len(caplog.records) == 1
    assert "RuntimeError: transient NVFlare initialization failure" in caplog.text
    assert isolated_importer.likely_lower_is_better_metric("val_loss") is False
    assert import_attempts == 2
    assert len(caplog.records) == 1
    assert core_calls == ["val_loss", "val_loss"]


def test_train_script_outside_workspace_is_not_admitted_to_trust_contract(tmp_path):
    outside_script = tmp_path.parent / f"{tmp_path.name}-external-client.py"
    outside_script.write_text("print('external train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

recipe = FedAvgRecipe(
    name="demo", min_clients=2, num_rounds=1, train_script="../{outside_script.name}"
)
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert "train_script" not in config["job"]
    assert str(outside_script) not in config["trust_contract"]["allowed_edit_paths"]
    assert any(item["field"] == "job.train_script" for item in config["unresolved"])


@pytest.mark.parametrize(
    "contents",
    [b"\xff\xfe", b"def broken(:\n", ("value = " + "(" * 10000 + "1" + ")" * 10000).encode("utf-8")],
)
def test_malformed_job_sources_raise_job_import_error(tmp_path, contents):
    job_path = tmp_path / "job.py"
    job_path.write_bytes(contents)

    with pytest.raises(job_importer.JobImportError, match="failed to parse"):
        import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))


def test_importer_never_executes_job_source(tmp_path):
    marker = tmp_path / "executed"
    tmp_path.joinpath("client.py").write_text("print('train')\n", encoding="utf-8")
    job_path = tmp_path / "job.py"
    job_path.write_text(
        f"""
from pathlib import Path
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

Path({str(marker)!r}).write_text("executed")
recipe = FedAvgRecipe(name="demo", min_clients=2, num_rounds=1, train_script="client.py")
recipe.execute(SimEnv(num_clients=2))
""".lstrip(),
        encoding="utf-8",
    )

    config = import_job_to_autofl_config(str(job_path), workspace_root=str(tmp_path))

    assert config["import"]["support"]["status"] == "supported"
    assert not marker.exists()


def test_inspect_job_cli_flag_aliases_returns_parser_spellings(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    source = job_path.read_text(encoding="utf-8").replace(
        'parser.add_argument("--n_clients", type=int, default=3)',
        'parser.add_argument("-n", "--n_clients", type=int, default=3)',
    )
    job_path.write_text(source, encoding="utf-8")

    groups = job_importer.inspect_job_cli_flag_aliases(str(job_path))

    assert ["--n_clients", "-n"] in groups
    assert ["--num_rounds"] in groups
