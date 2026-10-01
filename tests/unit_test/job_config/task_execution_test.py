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
from copy import deepcopy

import pytest

from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob
from nvflare.job_config.task_execution import TASK_EXECUTOR_PATH, prepare_task_execution
from nvflare.recipe.spec import Recipe


def _component(component_id, **args):
    return {"id": component_id, "path": "example.Component", "args": args}


def test_task_config_partitions_transitive_executor_and_filter_dependencies():
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "tasks": ["train"],
                "executor": {"path": "example.Executor", "args": {"learner_id": "learner"}},
            }
        ],
        "components": [
            _component("learner", optimizer_id="optimizer"),
            _component("optimizer"),
            _component("filter_state", source_id="filter_source"),
            _component("filter_source"),
            _component("job_handler"),
        ],
        "task_data_filters": [
            {"tasks": ["train"], "filters": [{"path": "example.Filter", "args": {"state_id": "filter_state"}}]}
        ],
        "task_result_filters": [],
    }

    original = deepcopy(config)
    plan = prepare_task_execution(config)

    assert plan.executors[0].executor == original["executors"][0]["executor"]
    assert [component["id"] for component in plan.executors[0].components] == ["learner", "optimizer"]
    assert [component["id"] for component in plan.resident_components] == [
        "filter_state",
        "filter_source",
        "job_handler",
    ]
    assert config == original


def test_task_config_rejects_component_shared_by_executor_and_filter():
    config = {
        "execution_lifetime": "task",
        "executors": [{"tasks": ["train"], "executor": {"path": "example.Executor", "args": {"state_id": "shared"}}}],
        "components": [_component("shared")],
        "task_data_filters": [
            {"tasks": ["train"], "filters": [{"path": "example.Filter", "args": {"state_id": "shared"}}]}
        ],
        "task_result_filters": [],
    }

    with pytest.raises(ValueError, match="both an Executor and CJ filters.*shared"):
        prepare_task_execution(config)


def test_task_config_rejects_duplicate_component_ids():
    config = {
        "execution_lifetime": "task",
        "executors": [{"tasks": ["train"], "executor": {"path": "example.Executor", "args": {}}}],
        "components": [_component("duplicate"), _component("duplicate")],
    }

    with pytest.raises(ValueError, match="duplicate client component id 'duplicate'"):
        prepare_task_execution(config)


def test_task_config_rejects_job_owned_launcher_selection():
    config = {
        "execution_lifetime": "task",
        "task_launcher": {"path": "job.CustomLauncher", "args": {}},
        "executors": [{"tasks": ["train"], "executor": {"path": "example.Executor", "args": {}}}],
    }

    with pytest.raises(ValueError, match="site/runtime configuration.*resources.json"):
        prepare_task_execution(config)


def test_fed_job_task_lifetime_exports_original_executor(tmp_path):
    executor = NPTrainer()
    job = FedJob(name="task-export", execution_lifetime="task")
    job.to_server(FedAvg())
    job.to_clients(executor, tasks=["train"])

    job.export_job(str(tmp_path))

    config = json.loads((tmp_path / job.name / "app/config/config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert "task_launcher" not in config
    assert config["executors"][0]["executor"] == {
        "path": "nvflare.app_common.np.np_trainer.NPTrainer",
        "args": {},
    }


def test_recipe_setter_updates_already_configured_client_apps(tmp_path):
    executor = NPTrainer()
    job = FedJob(name="recipe-task-export")
    job.to_server(FedAvg())
    job.to_clients(executor, tasks=["train"])
    recipe = Recipe(job)

    assert recipe.set_execution_lifetime("task") is recipe
    recipe.export(str(tmp_path))

    config = json.loads((tmp_path / job.name / "app/config/config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert config["executors"][0]["executor"]["path"] == "nvflare.app_common.np.np_trainer.NPTrainer"


def test_task_config_rejects_job_selected_framework_supervisor():
    config = {
        "execution_lifetime": "task",
        "executors": [{"tasks": ["train"], "executor": {"path": TASK_EXECUTOR_PATH, "args": {}}}],
    }

    with pytest.raises(ValueError, match="framework-managed and cannot be selected by a job"):
        prepare_task_execution(config)


@pytest.mark.parametrize("execution_lifetime", [None, "process", "TASK"])
def test_fed_job_rejects_invalid_execution_lifetime(execution_lifetime):
    with pytest.raises(ValueError, match="execution_lifetime"):
        FedJob(name="bad-lifetime", execution_lifetime=execution_lifetime)


def test_task_mode_rejects_client_gpu_resource_reservation(tmp_path):
    executor = NPTrainer()
    job = FedJob(name="task-gpu", execution_lifetime="task")
    job.to_server(FedAvg())
    job.to_clients(executor, tasks=["train"])
    job.job.add_resource_spec("site-1", {"num_of_gpus": 1})

    with pytest.raises(ValueError, match="CPU Process workers only.*num_of_gpus"):
        job.export_job(str(tmp_path))


def test_resident_mode_preserves_client_gpu_resource_reservation(tmp_path):
    executor = NPTrainer()
    job = FedJob(name="resident-gpu")
    job.to_server(FedAvg())
    job.to_clients(executor, tasks=["train"])
    job.job.add_resource_spec("site-1", {"num_of_gpus": 1})

    job.export_job(str(tmp_path))

    meta = json.loads((tmp_path / job.name / "meta.json").read_text())
    assert meta["resource_spec"]["site-1"]["num_of_gpus"] == 1


def test_client_api_task_lifetime_preserves_script_and_arguments(tmp_path):
    job = FedJob(name="client-api-task", execution_lifetime="task")
    job.to_server(FedAvg())
    job.to_clients(
        ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py", task_script_args=["a b"])
    )
    job.export_job(str(tmp_path))
    config = json.loads((tmp_path / job.name / "app/config/config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert config["executors"][0]["executor"]["args"]["task_script_args"] == ["a b"]
    assert config["executors"][0]["executor"]["args"]["task_script_path"] == "train.py"


@pytest.mark.parametrize("execution_mode", ["external_process", "attach"])
def test_client_api_task_lifetime_rejects_unimplemented_session_mapping(execution_mode):
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                    "args": {"execution_mode": execution_mode},
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="in_process scripts only"):
        prepare_task_execution(config)


def test_client_api_result_timeout_is_carried_to_worker_supervision():
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                    "args": {"execution_mode": "in_process", "result_wait_timeout": 15},
                },
            }
        ],
    }
    assert prepare_task_execution(config).executors[0].worker_timeout == 15
