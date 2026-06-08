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
)


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
    assert config["objective"] == {"metric": "AUC", "mode": "max", "source": "user_request"}
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
