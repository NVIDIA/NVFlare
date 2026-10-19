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

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe
from nvflare.job_config.task_execution import CLIENT_API_EXECUTOR_PATH, prepare_task_execution
from nvflare.recipe import add_cross_site_evaluation, add_final_global_evaluation


def _recipe(**kwargs):
    return FedOptRecipe(model=torch.nn.Linear(2, 1), min_clients=2, train_script=__file__, **kwargs)


def _export(recipe, directory):
    recipe.export(str(directory))
    config_dir = directory / recipe.name / "app" / "config"
    return [json.loads((config_dir / f"config_fed_{side}.json").read_text()) for side in ("server", "client")]


def test_fedopt_lifetime_only_changes_client_lifetime_and_format(tmp_path):
    recipe = _recipe()
    add_final_global_evaluation(recipe)
    server_job, client_job = _export(recipe, tmp_path / "job")
    recipe.add_client_config({"execution_lifetime": "task"})
    server_task, client_task = _export(recipe, tmp_path / "task")
    assert server_job == server_task
    assert client_job.pop("execution_lifetime") == "job"
    assert client_task.pop("execution_lifetime") == "task"
    assert client_job.pop("format_version") == 2
    assert client_task.pop("format_version") == 3
    assert client_job == client_task
    assert client_task["executors"][0]["executor"]["path"] == CLIENT_API_EXECUTOR_PATH


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_fedopt_constructor_exports_global_evaluation_with_existing_locator(tmp_path, lifetime):
    recipe = _recipe(execution_lifetime=lifetime)
    add_final_global_evaluation(recipe)
    server, client = _export(recipe, tmp_path)
    assert client["execution_lifetime"] == lifetime
    assert server["workflows"][0]["path"].endswith(".ScatterAndGather")
    evaluation = server["workflows"][1]
    assert evaluation["args"]["model_locator_id"] == "locator"
    assert evaluation["args"]["submit_model_task_name"] == ""
    assert sum(c["path"].endswith(".PTFileModelLocator") for c in server["components"]) == 1
    plan = prepare_task_execution(client)
    assert (plan is not None) == (lifetime == "task")


def test_fedopt_constructor_rejects_invalid_lifetime():
    with pytest.raises(ValueError, match="execution_lifetime"):
        _recipe(execution_lifetime="resident")


def test_fedopt_external_task_launch_remains_rejected(tmp_path):
    recipe = _recipe(execution_lifetime="task", launch_external_process=True)
    with pytest.raises(ValueError, match="requires in_process"):
        recipe.export(str(tmp_path))


def test_fedopt_local_model_submission_remains_rejected(tmp_path):
    recipe = _recipe(execution_lifetime="task")
    add_cross_site_evaluation(recipe)
    with pytest.raises(ValueError, match="local-model submission"):
        recipe.export(str(tmp_path))
