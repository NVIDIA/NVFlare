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
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig
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
    assert [component["id"] for component in plan.job_components] == [
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


@pytest.mark.parametrize("execution_lifetime", [None, "process", "TASK", "resident"])
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


def test_job_based_mode_preserves_client_gpu_resource_reservation(tmp_path):
    executor = NPTrainer()
    job = FedJob(name="job-gpu")
    job.to_server(FedAvg())
    job.to_clients(executor, tasks=["train"])
    job.job.add_resource_spec("site-1", {"num_of_gpus": 1})

    job.export_job(str(tmp_path))

    meta = json.loads((tmp_path / job.name / "meta.json").read_text())
    assert meta["resource_spec"]["site-1"]["num_of_gpus"] == 1


@pytest.mark.parametrize("wildcard", [False, True])
@pytest.mark.parametrize("resource_kind", ["resource_spec", "launcher_spec"])
def test_mixed_task_and_job_apps_preserve_job_based_gpu_resources(tmp_path, wildcard, resource_kind):
    task_app, job_app = ClientAppConfig(), ClientAppConfig()
    task_app.execution_lifetime = "task"
    task_app.add_executor(["train"], NPTrainer())
    job_app.add_executor(["train"], NPTrainer())
    settings = {"site-gpu": {"num_of_gpus": 1}}
    job = FedJobConfig("mixed-apps", min_clients=1, meta_props={resource_kind: settings})
    job.add_fed_app("cpu", FedAppConfig(client_app=task_app))
    job.add_fed_app("gpu", FedAppConfig(client_app=job_app))
    job.set_site_app("@ALL" if wildcard else "site-cpu", "cpu")
    job.set_site_app("site-gpu", "gpu")
    job.generate_job_config(str(tmp_path))
    assert json.loads((tmp_path / "mixed-apps/meta.json").read_text())[resource_kind] == settings


def test_task_resource_validation_uses_effective_defaults_and_overrides():
    task_app = ClientAppConfig()
    task_app.execution_lifetime = "task"
    job = FedJobConfig("task-defaults", min_clients=1)
    job.add_fed_app("cpu", FedAppConfig(client_app=task_app))
    job.set_site_app("site-cpu", "cpu")
    job.add_resource_spec("@default", {"num_of_gpus": 1})
    job.add_resource_spec("site-cpu", {"num_of_gpus": 0})
    job._prepare_meta()
    job.resource_specs["site-cpu"]["num_of_gpus"] = 1
    with pytest.raises(ValueError, match="CPU Process workers only"):
        job._prepare_meta()


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


@pytest.mark.parametrize("config", [None, []])
def test_task_placement_requires_client_config_mapping(config):
    with pytest.raises(TypeError, match="client_config"):
        prepare_task_execution(config)


def test_job_based_config_does_not_create_task_placement():
    assert prepare_task_execution({}) is None
    assert prepare_task_execution({"execution_lifetime": "job"}) is None


@pytest.mark.parametrize("lifetime", ["job", "task"])
@pytest.mark.parametrize("setting", ["task_launcher", "task_execution"])
def test_job_cannot_override_site_owned_runtime_settings(lifetime, setting):
    with pytest.raises(ValueError, match="site/runtime configuration.*resources.json"):
        prepare_task_execution({"execution_lifetime": lifetime, setting: {"artifact_cleanup": "retain"}})


@pytest.mark.parametrize(
    "changes", [{"components": {}}, {"components": [None]}, {"executors": [{}]}, {"executors": []}]
)
def test_task_placement_rejects_malformed_compute_graph(changes):
    config = {"execution_lifetime": "task", "executors": [{"executor": {"path": "example.Executor"}}]}
    with pytest.raises(ValueError):
        prepare_task_execution(config | changes)


@pytest.mark.parametrize("timeout", [-1, True, "1", float("nan"), float("inf")])
def test_client_api_task_timeout_is_validated(timeout):
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "executor": {
                    "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                    "args": {"execution_mode": "in_process", "result_wait_timeout": timeout},
                }
            }
        ],
    }
    with pytest.raises(ValueError, match="result_wait_timeout"):
        prepare_task_execution(config)


def test_component_closure_handles_cycles_without_duplicate_placement():
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "args": {"ids": ["a", "b"]}}}],
        "components": [_component("a", dependency="b"), _component("b", dependency="a")],
    }
    assert len(prepare_task_execution(config).executors[0].components) == 2


def test_task_resource_validation_handles_nested_defaults_and_non_mapping_settings():
    job = FedJobConfig("resources", min_clients=1, meta_props={"resource_spec": [], "launcher_spec": None})
    job._validate_task_execution_resources()
    assert job._merge_resource_settings({"nested": {"cpu": 1, "gpu": 0}}, {"nested": {"cpu": 2}}) == {
        "nested": {"cpu": 2, "gpu": 0}
    }
    assert job._merge_resource_settings(None, None) == {}
    assert job._find_nonempty_gpu_setting({"nested": [None, {"gpu": "device-0"}]}) == "nested[1].gpu"
    assert job._find_nonempty_gpu_setting({"nested": [None, {"gpu": False}]}) is None
