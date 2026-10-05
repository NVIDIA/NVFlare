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

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.api import FedJob
from nvflare.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig
from nvflare.job_config.task_execution import TASK_EXECUTOR_PATH, prepare_task_execution
from nvflare.recipe.spec import Recipe


def _component(component_id, execution_scope=None, component_dependencies=None, **args):
    config = {"id": component_id, "path": "example.Component", "args": args}
    if execution_scope is not None:
        config["execution_scope"] = execution_scope
    if component_dependencies is not None:
        config["component_dependencies"] = component_dependencies
    return config


def test_task_config_uses_explicit_component_scope_and_keeps_filters_in_cj():
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
            _component("filter_state", execution_scope="job", component_dependencies=["filter_source"]),
            _component("filter_source", execution_scope="job"),
            _component("job_handler", execution_scope="job"),
        ],
        "task_data_filters": [
            {
                "tasks": ["train"],
                "filters": [
                    {
                        "path": "example.Filter",
                        "args": {"state_id": "filter_state"},
                        "component_dependencies": ["filter_state"],
                    }
                ],
            }
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


def test_task_config_rejects_filter_dependency_on_worker_component():
    config = {
        "execution_lifetime": "task",
        "executors": [{"tasks": ["train"], "executor": {"path": "example.Executor", "args": {"state_id": "shared"}}}],
        "components": [_component("shared")],
        "task_data_filters": [
            {
                "tasks": ["train"],
                "filters": [{"path": "example.Filter", "component_dependencies": ["shared"]}],
            }
        ],
        "task_result_filters": [],
    }

    with pytest.raises(ValueError, match="CJ filters.*another scope.*shared"):
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


@pytest.mark.parametrize("lifetime", ["job", "task"])
@pytest.mark.parametrize("target", ["@ALL", "site-1"])
def test_base_fed_job_explicitly_retains_federation_converter(tmp_path, lifetime, target):
    job = BaseFedJob(name="federation-converter")
    job.set_execution_lifetime(lifetime)
    job.to_server(FedAvg())
    job.to(NPTrainer(), target, tasks=["train"])
    job.to(FLComponent(), target, id="compute")
    job.export_job(str(tmp_path))

    config_path = next((tmp_path / job.name).rglob("config_fed_client.json"))
    config = json.loads(config_path.read_text())
    components = {component["id"]: component for component in config["components"]}
    assert components["event_to_fed"]["execution_scope"] == "job"
    assert "execution_scope" not in components["compute"]
    assert "execution_scope" not in components["event_to_fed"]["args"]
    plan = prepare_task_execution(config)
    if lifetime == "job":
        assert "execution_lifetime" not in config
        assert plan is None
    else:
        assert [component["id"] for component in plan.job_components] == ["event_to_fed"]
        assert [component["id"] for component in plan.executors[0].components] == ["compute"]


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
    plan = prepare_task_execution(config).executors[0]
    assert plan.result_wait_timeout == 15
    assert plan.worker_timeout is None


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


def test_declared_dependency_cycles_do_not_duplicate_components():
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "args": {"component_ids": ["a", "b"]}}}],
        "components": [
            _component("a", component_dependencies=["b"]),
            _component("b", component_dependencies=["a"]),
        ],
    }
    assert len(prepare_task_execution(config).executors[0].components) == 2


@pytest.mark.parametrize("argument", ["label", "dataset_id", "run_id", "dataset_ids"])
def test_ordinary_arguments_do_not_move_explicitly_job_owned_components(argument):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "args": {argument: "state"}}}],
        "components": [_component("state", execution_scope="job")],
    }
    plan = prepare_task_execution(config)
    assert not plan.executors[0].components
    assert plan.job_components[0]["id"] == "state"
    config["executors"][0]["executor"]["component_dependencies"] = ["state"]
    with pytest.raises(ValueError, match="Executor.*another scope.*state"):
        prepare_task_execution(config)


def test_job_widget_cannot_reference_worker_owned_component():
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "args": {"state_id": "state"}}}],
        "components": [
            _component("state"),
            _component("widget", execution_scope="job", component_dependencies=["state"]),
        ],
    }
    with pytest.raises(ValueError, match="component 'widget'.*another scope.*state"):
        prepare_task_execution(config)


@pytest.mark.parametrize(
    "options",
    [
        {"args": [1, 2]},
        {"name": "ordinary-data", "args": [1, 2]},
        {"path": "/data/model", "args": [1, 2]},
        {"config_type": "dict", "path": "example.Data", "args": [1, 2]},
        {"config_type": "dict", "path": "example.Data", "execution_scope": "ordinary-data"},
        {"component_dependencies": "ordinary-data", "nested": {"args": [1, 2]}},
    ],
)
def test_ordinary_nested_argument_dictionaries_are_not_component_specs(options):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "args": {"options": options}}}],
        "components": [_component("unused")],
    }
    plan = prepare_task_execution(config)
    assert plan.executors[0].executor["args"]["options"] == options
    assert plan.executors[0].components[0]["id"] == "unused"
    assert not plan.job_components


@pytest.mark.parametrize("declare_dependencies", [False, True])
def test_nonstandard_and_dynamic_lookups_reach_all_worker_default_components(declare_dependencies):
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "executor": {
                    "path": "example.Executor",
                    "args": {"source_model": "model"},
                }
            }
        ],
        "components": [
            _component("model", component_dependencies=["weights"], weights_id="weights"),
            _component("weights"),
            _component("widget", execution_scope="job"),
            _component("dynamic_component"),
        ],
    }
    if declare_dependencies:
        config["executors"][0]["executor"]["component_dependencies"] = ["model"]
    plan = prepare_task_execution(config)
    assert [component["id"] for component in plan.executors[0].components] == ["model", "weights", "dynamic_component"]
    assert [component["id"] for component in plan.job_components] == ["widget"]


@pytest.mark.parametrize("dependencies", ["state", [None], ["missing"]])
def test_explicit_component_dependencies_are_validated(dependencies):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor", "component_dependencies": dependencies}}],
        "components": [_component("state")],
    }
    with pytest.raises(ValueError, match="component_dependencies"):
        prepare_task_execution(config)


def test_all_worker_default_components_are_available_to_every_executor_without_duplication():
    config = {
        "execution_lifetime": "task",
        "executors": [
            {"tasks": ["train"], "executor": {"path": "example.TrainExecutor"}},
            {"tasks": ["evaluate"], "executor": {"path": "example.EvaluateExecutor"}},
        ],
        "components": [_component("model"), _component("dynamic"), _component("explicit", execution_scope="task")],
    }
    original = deepcopy(config)
    plan = prepare_task_execution(config)

    assert not plan.job_components
    for executor in plan.executors:
        assert [component["id"] for component in executor.components] == ["model", "dynamic", "explicit"]
    plan.executors[0].components[0]["args"]["changed"] = True
    assert not plan.executors[1].components[0]["args"]
    assert config == original


@pytest.mark.parametrize("scope", [None, True, "worker", "client", "JOB", [], {}])
def test_task_components_reject_invalid_execution_scope(scope):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor"}}],
        "components": [_component("model") | {"execution_scope": scope}],
    }
    with pytest.raises(ValueError, match="execution_scope must be 'job' or 'task'"):
        prepare_task_execution(config)


@pytest.mark.parametrize("lifetime", [None, "job"])
def test_job_lifetime_does_not_interpret_task_component_metadata(lifetime):
    config = {"components": [{"id": "legacy", "execution_scope": "legacy", "component_dependencies": "not-a-list"}]}
    if lifetime is not None:
        config["execution_lifetime"] = lifetime
    original = deepcopy(config)
    assert prepare_task_execution(config) is None
    assert config == original


def test_task_state_declares_same_named_records_for_every_supervisor():
    config = {
        "execution_lifetime": "task",
        "task_state": {"names": ["optimizer", "metrics.round"]},
        "executors": [
            {"tasks": ["train"], "executor": {"path": "example.TrainExecutor"}},
            {"tasks": ["evaluate"], "executor": {"path": "example.EvaluateExecutor"}},
        ],
    }
    original = deepcopy(config)
    assert [executor.state_names for executor in prepare_task_execution(config).executors] == [
        ("optimizer", "metrics.round"),
        ("optimizer", "metrics.round"),
    ]
    assert config == original


@pytest.mark.parametrize("state_config", [None, [], "optimizer", {"unknown": []}, {"names": [], "backend": "disk"}])
def test_task_state_rejects_malformed_or_unknown_settings(state_config):
    config = {
        "execution_lifetime": "task",
        "task_state": state_config,
        "executors": [{"executor": {"path": "example.Executor"}}],
    }
    with pytest.raises((TypeError, ValueError), match="task_state"):
        prepare_task_execution(config)


@pytest.mark.parametrize(
    "names", ["optimizer", ["optimizer", "optimizer"], [None], [""], ["../optimizer"], ["x" * 129]]
)
def test_task_state_rejects_invalid_or_duplicate_names(names):
    config = {
        "execution_lifetime": "task",
        "task_state": {"names": names},
        "executors": [{"executor": {"path": "example.Executor"}}],
    }
    with pytest.raises((TypeError, ValueError)):
        prepare_task_execution(config)


@pytest.mark.parametrize("state_config", [None, {"names": "not-a-list"}, {"unknown": True}])
def test_job_lifetime_does_not_enable_or_interpret_task_state(state_config):
    config = {"execution_lifetime": "job", "task_state": state_config}
    original = deepcopy(config)
    assert prepare_task_execution(config) is None
    assert config == original


@pytest.mark.parametrize("owner_scope,target_scope", [("task", "job"), ("job", "task")])
def test_declared_component_dependencies_cannot_cross_execution_scopes(owner_scope, target_scope):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor"}}],
        "components": [
            _component("owner", execution_scope=owner_scope, component_dependencies=["target"]),
            _component("target", execution_scope=target_scope),
        ],
    }
    with pytest.raises(ValueError, match="component 'owner'.*another scope.*target"):
        prepare_task_execution(config)


@pytest.mark.parametrize("filter_key", ["task_data_filters", "task_result_filters"])
def test_filter_nested_components_validate_declared_job_scope_dependencies(filter_key):
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor"}}],
        "components": [_component("filter_state", execution_scope="job")],
        filter_key: [
            {
                "tasks": ["train"],
                "filters": [
                    {
                        "path": "example.Filter",
                        "args": {"helper": {"path": "example.Helper", "component_dependencies": ["filter_state"]}},
                    }
                ],
            }
        ],
    }
    assert not prepare_task_execution(config).executors[0].components
    config["components"][0]["execution_scope"] = "task"
    with pytest.raises(ValueError, match="CJ filters.*another scope.*filter_state"):
        prepare_task_execution(config)


@pytest.mark.parametrize("dependencies", [["job_state"], ["missing"], "job_state"])
def test_executor_nested_components_validate_explicit_dependencies(dependencies):
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "executor": {
                    "path": "example.Executor",
                    "args": {"helper": {"path": "example.Helper", "component_dependencies": dependencies}},
                }
            }
        ],
        "components": [_component("job_state", execution_scope="job")],
    }
    with pytest.raises(ValueError, match="another scope|component_dependencies"):
        prepare_task_execution(config)


@pytest.mark.parametrize("scope", [None, "job", "worker", True])
def test_nested_worker_components_cannot_select_another_or_invalid_scope(scope):
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "executor": {
                    "path": "example.Executor",
                    "args": {"nested": {"path": "example.Helper", "execution_scope": scope}},
                }
            }
        ],
    }
    with pytest.raises(ValueError, match="execution_scope"):
        prepare_task_execution(config)


def test_nested_worker_components_can_explicitly_confirm_their_enclosing_scope():
    config = {
        "execution_lifetime": "task",
        "executors": [
            {
                "executor": {
                    "path": "example.Executor",
                    "args": {"nested": {"path": "example.Helper", "execution_scope": "task"}},
                }
            }
        ],
    }
    assert prepare_task_execution(config).executors[0].executor == config["executors"][0]["executor"]


def test_explicit_job_component_keeps_job_scoped_dynamic_dependencies():
    config = {
        "execution_lifetime": "task",
        "executors": [{"executor": {"path": "example.Executor"}}],
        "components": [
            _component("compute"),
            _component("widget", execution_scope="job", component_dependencies=["job_state"]),
            _component("job_state", execution_scope="job"),
        ],
    }
    plan = prepare_task_execution(config)
    assert [component["id"] for component in plan.executors[0].components] == ["compute"]
    assert [component["id"] for component in plan.job_components] == ["widget", "job_state"]


def test_task_resource_validation_handles_nested_defaults_and_non_mapping_settings():
    job = FedJobConfig("resources", min_clients=1, meta_props={"resource_spec": [], "launcher_spec": None})
    job._validate_task_execution_resources()
    assert job._merge_resource_settings({"nested": {"cpu": 1, "gpu": 0}}, {"nested": {"cpu": 2}}) == {
        "nested": {"cpu": 2, "gpu": 0}
    }
    assert job._merge_resource_settings(None, None) == {}
    assert job._find_nonempty_gpu_setting({"nested": [None, {"gpu": "device-0"}]}) == "nested[1].gpu"
    assert job._find_nonempty_gpu_setting({"nested": [None, {"gpu": False}]}) is None


@pytest.mark.parametrize("lifetime", ["job", "task"])
@pytest.mark.parametrize("declare_before_clients", [False, True])
def test_job_api_task_state_declarations_reach_existing_and_future_clients(tmp_path, lifetime, declare_before_clients):
    job = FedJob(name="declared-state", execution_lifetime=lifetime)
    job.to_server(FedAvg())
    if declare_before_clients:
        assert job.set_task_state(["optimizer", "metrics"]) is job
    job.to(NPTrainer(), "site-1", tasks=["train"])
    if not declare_before_clients:
        job.set_task_state(["optimizer", "metrics"])
    job.to(NPTrainer(), "site-2", tasks=["train"])
    job.export_job(str(tmp_path))
    configs = list((tmp_path / job.name).rglob("config_fed_client.json"))
    assert len(configs) == 2
    for path in configs:
        assert json.loads(path.read_text())["task_state"] == {"names": ["optimizer", "metrics"]}
    for path in (tmp_path / job.name).rglob("config_fed_server.json"):
        assert "task_state" not in json.loads(path.read_text())


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_job_api_omits_task_state_by_default_and_explicitly_exports_empty_declarations(tmp_path, lifetime):
    job = FedJob(name="state-default", execution_lifetime=lifetime)
    job.to_server(FedAvg())
    job.to_clients(NPTrainer(), tasks=["train"])
    job.export_job(str(tmp_path))
    path = tmp_path / job.name / "app/config/config_fed_client.json"
    assert "task_state" not in json.loads(path.read_text())
    job.set_task_state([])
    job.export_job(str(tmp_path))
    assert json.loads(path.read_text())["task_state"] == {"names": []}


def test_job_api_invalid_state_declaration_keeps_every_client_declaration_unchanged():
    job = FedJob(name="state-validation")
    job.to_clients(NPTrainer())
    job.set_task_state(["optimizer"])
    with pytest.raises(ValueError, match="unique"):
        job.set_task_state(["other", "other"])
    assert job._deploy_map["@ALL"].app_config.task_state_names == ("optimizer",)
    assert job._task_state_names == ("optimizer",)


@pytest.mark.parametrize("lifetime", ["job", "task"])
def test_job_api_explicit_component_scope_is_export_metadata_only(tmp_path, lifetime):
    job = FedJob(name="component-scope", execution_lifetime=lifetime)
    job.to_server(FedAvg())
    server_component = FLComponent()
    job.to_server(server_component, id="server_component")
    job.to_clients(NPTrainer(), tasks=["train"])
    component = FLComponent()
    attributes = dict(vars(component))
    component_id = job.to_clients(component, id="cj_handler")
    assert job.set_component_execution_scope(component_id, "job") is job
    job.export_job(str(tmp_path))
    config_dir = tmp_path / job.name / "app/config"
    client = json.loads((config_dir / "config_fed_client.json").read_text())
    assert client["components"] == [
        {"id": component_id, "path": "nvflare.apis.fl_component.FLComponent", "args": {}, "execution_scope": "job"}
    ]
    assert vars(component) == attributes
    server = json.loads((config_dir / "config_fed_server.json").read_text())
    assert all("execution_scope" not in component for component in server["components"])
    if lifetime == "task":
        assert not prepare_task_execution(client).executors[0].components


def test_component_scope_targets_one_registered_client_app_without_mutating_shared_object():
    job = FedJob(name="site-scopes")
    shared = FLComponent()
    job.to(shared, "site-1", id="shared")
    job.to(shared, "site-2", id="shared")
    job.set_component_execution_scope("shared", "job", target="site-1")
    assert job._deploy_map["site-1"].app_config.component_execution_scopes == {"shared": "job"}
    assert job._deploy_map["site-2"].app_config.component_execution_scopes == {}
    assert not hasattr(shared, "execution_scope")


@pytest.mark.parametrize("target", ["server", "not-registered", "invalid/site"])
def test_component_scope_does_not_create_an_app_or_change_server_placement(target):
    job = FedJob(name="scope-target")
    job.to_server(FedAvg())
    original_targets = set(job._deploy_map)
    with pytest.raises(ValueError):
        job.set_component_execution_scope("component", "job", target=target)
    assert set(job._deploy_map) == original_targets
