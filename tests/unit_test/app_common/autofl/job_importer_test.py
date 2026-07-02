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

import yaml

from nvflare.app_common.autofl import (
    AUTOFL_CONFIG_SCHEMA_VERSION,
    DeterministicJobImporter,
    dump_autofl_yaml,
    import_job_to_autofl_config,
    job_importer,
)


def _objective(metric, source="user_request"):
    return {
        "metric": metric,
        "requested_metric": metric,
        "optimization_metric": metric,
        "metric_extraction_order": [metric],
        "mode": "max",
        "source": source,
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
    assert config["objective"] == _objective("AUC")
    assert config["budget"]["max_candidates"] == 8
    assert config["budget"]["fixed_training_budget"] == {
        "num_rounds": 5,
        "min_clients": 3,
        "num_clients": 3,
    }
    assert config["environment"]["requested"] == "prod"
    assert config["environment"]["profiles"]["sim"] == {"num_clients": 3}
    assert config["search_space"]["suggested"]["lr"]["default"] == 0.01
    assert config["search_space"]["suggested"]["batch_size"]["type"] == "int"
    assert config["trust_contract"]["allowed_edit_paths"] == ["job.py", "client.py", "model.py"]
    assert config["job"]["allowed_create_patterns"] == ["**/*.py"]
    assert config["trust_contract"]["allowed_create_patterns"] == ["**/*.py"]
    assert config["trust_contract"]["agent_controls"]["must_not_edit_outside_allowed_paths"] is True
    assert config["unresolved"] == []


def test_import_is_repeatable_and_yaml_round_trips(tmp_path):
    job_path = _write_recipe_job(tmp_path)
    importer = DeterministicJobImporter(workspace_root=str(tmp_path))

    first = importer.import_job(str(job_path), max_candidates=4)
    second = importer.import_job(str(job_path), max_candidates=4)
    yaml_text = dump_autofl_yaml(first)

    assert first == second
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
    job = ImportedJob(name="fedavg-alias", n_clients=8, min_clients=4, num_rounds=10, key_metric="AUC")
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
    assert config["objective"] == _objective("AUC", source="literal")
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


def test_main_returns_clean_error_for_missing_job(tmp_path, capsys):
    output_path = tmp_path / "autofl.yaml"

    exit_code = job_importer.main([str(tmp_path / "missing.py"), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: job.py not found:" in captured.err
    assert not output_path.exists()
