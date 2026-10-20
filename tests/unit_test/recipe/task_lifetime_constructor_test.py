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

"""Constructor parity for recipes using the existing Client API task path."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe
from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe
from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe
from nvflare.job_config.task_execution import CLIENT_API_EXECUTOR_PATH, prepare_task_execution

RECIPE_CLASSES = [NumpyFedAvgRecipe, SklearnFedAvgRecipe, KMeansFedAvgRecipe, SVMFedAvgRecipe, FedCERecipe]


def _recipe(recipe_class, *, train_script=__file__, **kwargs):
    if recipe_class is NumpyFedAvgRecipe:
        kwargs["model"] = np.zeros(2)
    elif recipe_class is FedCERecipe:
        kwargs["model"] = torch.nn.Linear(2, 1)
    return recipe_class(name="lifetime", min_clients=2, train_script=train_script, **kwargs)


def _export(recipe, directory):
    recipe.export(str(directory))
    config_dir = directory / recipe.name / "app" / "config"
    return [json.loads((config_dir / f"config_fed_{side}.json").read_text()) for side in ("server", "client")]


@pytest.mark.parametrize("recipe_class", RECIPE_CLASSES)
def test_constructor_matches_common_config_and_preserves_default_job(tmp_path, recipe_class):
    script = tmp_path / "client.py"
    script.write_text("pass\n")
    recipe = _recipe(recipe_class, train_script=str(script))
    server_job, client_job = _export(recipe, tmp_path / "default")
    recipe.add_client_config({"execution_lifetime": "task"})
    server_common, client_common = _export(recipe, tmp_path / "common")
    explicit = _recipe(recipe_class, train_script=str(script), execution_lifetime="task")
    server_task, client_task = _export(explicit, tmp_path / "constructor")
    assert server_job == server_common == server_task
    assert client_task == client_common
    assert client_job.pop("execution_lifetime", "job") == "job"
    assert client_task.pop("execution_lifetime") == "task"
    assert client_job.pop("format_version") == 2
    assert client_task.pop("format_version") == 3
    assert client_job == client_task
    assert client_task["executors"][0]["executor"]["path"] == CLIENT_API_EXECUTOR_PATH
    assert prepare_task_execution(client_common) is not None


@pytest.mark.parametrize("recipe_class", RECIPE_CLASSES)
def test_invalid_lifetime_is_rejected(recipe_class):
    with pytest.raises(ValueError, match="execution_lifetime"):
        _recipe(recipe_class, execution_lifetime="resident")


@pytest.mark.parametrize("recipe_class", RECIPE_CLASSES)
def test_external_task_launch_still_rejected(tmp_path, recipe_class):
    recipe = _recipe(recipe_class, execution_lifetime="task", launch_external_process=True)
    with pytest.raises(ValueError, match="requires in_process"):
        recipe.export(str(tmp_path))


@pytest.mark.parametrize(
    "catalog_name", ["fedavg-numpy", "fedavg-sklearn", "kmeans-sklearn", "svm-sklearn", "fedce-pt"]
)
def test_generated_catalog_exposes_lifetime(catalog_name):
    root = Path(__file__).parents[3]
    catalog = json.loads((root / "nvflare/tool/recipe/recipe_catalog.json").read_text())
    recipe = next(entry for entry in catalog["recipes"] if entry["summary"]["name"] == catalog_name)
    parameter = next(p for p in recipe["detail"]["parameters"] if p["name"] == "execution_lifetime")
    assert parameter["default"] == "job" and parameter["type"] == "str"
    assert not parameter["required"]
