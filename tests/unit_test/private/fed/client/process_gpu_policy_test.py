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

import argparse
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from pyhocon import ConfigFactory as HoconFactory
from pyhocon import HOCONConverter

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.task_launcher_spec import UnsupportedTaskResourceError
from nvflare.apis.workspace import Workspace
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.logging.constants import is_log_streaming_allowed
from nvflare.app_common.resource_managers.list_resource_manager import ListResourceManager
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.task_worker.worker import ComponentNotAuthorized, _authorize_bootstrap
from nvflare.utils.job_launcher_utils import get_resource_manager_spec, normalize_cj_resource_spec

GPU = "GPU-00000000-0000-0000-0000-000000000001"


def _config(tmp_path):
    return {
        "components": [
            {
                "id": "resource_manager",
                "path": "nvflare.app_common.resource_managers.list_resource_manager.ListResourceManager",
                "args": {"resources": {}},
            }
        ],
        "task_execution": {
            "launcher": {
                "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
                "args": {"gpu_devices": [GPU]},
            }
        },
    }


def _write_site_config(path, config):
    if ".conf" in path.name:
        text = HOCONConverter.to_hocon(HoconFactory.from_dict(config))
    elif ".yml" in path.name or ".yaml" in path.name:
        text = OmegaConf.to_yaml(OmegaConf.create(config))
    else:
        text = json.dumps(config)
    path.write_text(text)


def _workspace(tmp_path, config):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(config))
    return SimpleNamespace(get_resources_file_path=lambda: str(path))


def test_task_gpu_count_uses_launcher_spec_without_job_reservation(tmp_path):
    policy = TaskExecutionPolicy.from_workspace(_workspace(tmp_path, _config(tmp_path)))
    meta = {"launcher_spec": {"default": {"process": {"num_of_gpus": 1}}}}
    assert policy.task_resources(meta, "site").gpu_count == 1
    assert policy.task_resources({}, "site").gpu_count == 0
    with pytest.raises(ValueError, match="whole job"):
        policy.task_resources({"resource_spec": {"site": {"num_of_gpus": 1}}}, "site")
    with pytest.raises(ValueError, match="process launcher_spec"):
        policy.task_resources({"launcher_spec": {"site": {"docker": {"num_of_gpus": 1}}}}, "site")


@pytest.mark.parametrize(
    "resource_spec",
    [
        {"@default": {"num_of_gpus": 1}},
        {"site": {"num_of_gpus": 1}},
        {"site": {"process": {"num_of_gpus": 1}}},
    ],
)
def test_normalized_task_gpu_compute_is_separate_from_cj_admission(tmp_path, resource_spec):
    workspace = _workspace(tmp_path, _config(tmp_path))
    policy = TaskExecutionPolicy.from_workspace(workspace)
    meta = {"resource_spec": resource_spec}
    normalize_cj_resource_spec(meta, {"site"})
    assert get_resource_manager_spec(meta, "site") == {}
    assert policy.task_resources(meta, "site").gpu_count == 1
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment(meta, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert meta["resource_spec"] == resource_spec


def test_portable_site_zero_gpu_override_clears_compute_default(tmp_path):
    policy = TaskExecutionPolicy.from_workspace(_workspace(tmp_path, _config(tmp_path)))
    meta = {"resource_spec": {"@default": {"num_of_gpus": 1}, "site": {"num_of_gpus": 0}}}
    normalize_cj_resource_spec(meta, {"site"})
    assert get_resource_manager_spec(meta, "site") == {}
    assert policy.task_resources(meta, "site").gpu_count == 0


@pytest.mark.parametrize("count", [0, 2])
def test_conflicting_compute_and_process_launcher_gpu_counts_are_rejected(tmp_path, count):
    policy = TaskExecutionPolicy.from_workspace(_workspace(tmp_path, _config(tmp_path)))
    meta = {
        "resource_spec": {"@default": {"num_of_gpus": 1}},
        "launcher_spec": {"default": {"process": {"num_of_gpus": count}}},
    }
    normalize_cj_resource_spec(meta, {"site"})
    with pytest.raises(ValueError, match="counts.*must agree"):
        policy.task_resources(meta, "site")


@pytest.mark.parametrize(
    "meta",
    [
        {"resource_spec": {"site": {"license": 1}}},
        {"resource_spec": {"site": {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 1}}},
        {"resource_spec": {"@default": {"num_of_cpus": 1}}},
        {"resource_spec": {"@default": {"memory": "64Mi"}}},
        {"launcher_spec": {"site": {"process": {"num_of_cpus": 1}}}},
        {"launcher_spec": {"site": {"process": {"memory": "64Mi"}}}},
    ],
)
def test_gpu_visibility_keeps_shared_process_compute_capability_guards(tmp_path, meta):
    policy = TaskExecutionPolicy.from_workspace(_workspace(tmp_path, _config(tmp_path)))
    normalize_cj_resource_spec(meta, {"site"})
    with pytest.raises(UnsupportedTaskResourceError, match="cannot enforce"):
        policy.task_resources(meta, "site")
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    with pytest.raises(UnsupportedTaskResourceError, match="cannot enforce"):
        ClientProcessJobLauncher().get_command(meta, context)


@pytest.mark.parametrize("budget", [{"num_of_cpus": 0.25}, {"memory": "64Mi"}, {"num_of_gpus": 1}])
def test_process_rejects_unenforceable_independent_cj_budget_before_launch(budget):
    meta = {"cj_resource_spec": {"site": budget}}
    normalize_cj_resource_spec(meta, {"site"})
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    with pytest.raises(ValueError, match="Process CJ cannot enforce"):
        ClientProcessJobLauncher().get_command(meta, context)


def test_task_gpu_capacity_is_checked_before_cp_changes_environment(tmp_path):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, _config(tmp_path)))
    meta = {"resource_spec": {"@default": {"num_of_gpus": 2}}}
    normalize_cj_resource_spec(meta, {"site"})
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    with pytest.raises(UnsupportedTaskResourceError, match="static visibility assignment"):
        ClientProcessJobLauncher().prepare_environment(meta, context, environment)
    assert environment == {"CUDA_VISIBLE_DEVICES": "all"}


def test_cp_hides_gpus_after_study_overrides_only_for_opted_in_profile(tmp_path):
    workspace = _workspace(tmp_path, _config(tmp_path))
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    manager = ListResourceManager(resources={})
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: manager))
    environment = {"CUDA_VISIBLE_DEVICES": "all", "NVIDIA_VISIBLE_DEVICES": "all", "UNCHANGED": "yes"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment == {
        "CUDA_VISIBLE_DEVICES": "",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "UNCHANGED": "yes",
    }
    _workspace(tmp_path, {})
    legacy_environment = {"CUDA_VISIBLE_DEVICES": "0"}
    ClientProcessJobLauncher().prepare_environment(
        {"resource_spec": {"@default": {"num_of_gpus": 1}}}, context, legacy_environment
    )
    assert legacy_environment == {"CUDA_VISIBLE_DEVICES": "0"}


@pytest.mark.parametrize("resources", [{}, {"cpu": [1]}, {"gpu": [0, 1]}])
def test_task_gpu_profile_leaves_generic_job_resource_inventories_unchanged(tmp_path, resources):
    config = _config(tmp_path)
    config["components"][0]["args"]["resources"] = resources
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, config))
    manager = ListResourceManager(resources)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: manager))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert {key: list(value) for key, value in manager.resources.items()} == resources


@pytest.mark.parametrize(
    "resource_spec",
    [
        {"@default": {"num_of_gpus": 1}},
        {"site": {"num_of_gpus": 1}},
        {"site": {"process": {"num_of_gpus": 1}}},
    ],
)
def test_task_gpu_profile_rejects_job_gpu_requests_before_changing_environment(tmp_path, resource_spec):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, _config(tmp_path)))
    environment = {"CUDA_VISIBLE_DEVICES": "unchanged"}
    with pytest.raises(ValueError, match="rejects job-level GPU requests"):
        ClientProcessJobLauncher().prepare_environment({"resource_spec": resource_spec}, context, environment)
    assert environment == {"CUDA_VISIBLE_DEVICES": "unchanged"}


@pytest.mark.parametrize(
    "meta",
    [
        {"resource_spec": {"other-site": {"num_of_gpus": 1}}},
        {"resource_spec": {"@default": {"num_of_gpus": 1}, "site": {"num_of_gpus": 0}}},
        {"resource_spec": {"site": {"cpu": 1}}},
        {"launcher_spec": {"site": {"process": {"num_of_gpus": 1}}}},
    ],
)
def test_task_gpu_profile_resolves_only_this_sites_job_gpu_request(tmp_path, meta):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, _config(tmp_path)))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment(meta, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_only_process_policy_still_rejects_gpu_tasks():
    with pytest.raises(ValueError, match="CPU Process"):
        TaskExecutionPolicy.from_config({}).task_resources(
            {"launcher_spec": {"site": {"process": {"num_of_gpus": 1}}}}, "site"
        )


@pytest.mark.parametrize("extension", [".json", ".conf", ".yml", ".json.default", ".conf.default", ".yml.default"])
def test_cp_and_cj_use_same_gpu_visibility_from_supported_site_formats(tmp_path, extension):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    _write_site_config(tmp_path / "local" / ("resources" + extension), _config(tmp_path))
    workspace = Workspace(str(tmp_path), "site")
    policy = TaskExecutionPolicy.from_workspace(workspace)
    assert policy.launcher.gpu_devices == (GPU,)
    assert policy.task_resources({"launcher_spec": {"default": {"process": {"num_of_gpus": 1}}}}, "site").gpu_count == 1
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: ListResourceManager({})))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_nominal_json_path_is_resolved_by_config_service_before_assuming_no_policy(tmp_path):
    nominal = tmp_path / "resources.json"
    _write_site_config(nominal.with_suffix(".conf"), _config(tmp_path))
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(nominal))
    assert TaskExecutionPolicy.from_workspace(workspace).launcher.gpu_devices == (GPU,)


def test_canonical_conf_is_not_shadowed_by_nested_json_in_cp_or_cj(tmp_path):
    (tmp_path / "startup").mkdir()
    custom = tmp_path / "local" / "custom"
    custom.mkdir(parents=True)
    _write_site_config(tmp_path / "local" / "resources.conf", _config(tmp_path))
    (custom / "resources.json").write_text("{}")
    workspace = Workspace(str(tmp_path), "site")
    assert TaskExecutionPolicy.from_workspace(workspace).launcher.gpu_devices == (GPU,)
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: ListResourceManager({})))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("extension", [".conf", ".yaml"])
@pytest.mark.parametrize("relative_workspace", [False, True], ids=["absolute-root", "relative-root"])
def test_actual_cp_startup_and_config_sections_keep_the_selected_site_file(
    tmp_path, monkeypatch, extension, relative_workspace
):
    monkeypatch.setattr(DataBus(), "data_store", {})
    (tmp_path / "startup").mkdir()
    custom = tmp_path / "local" / "custom"
    custom.mkdir(parents=True)
    (tmp_path / "startup" / "fed_client.json").write_text(json.dumps({"client": {}, "servers": []}))
    canonical = _config(tmp_path)
    canonical["task_execution"]["worker_timeout"] = 33
    _write_site_config(tmp_path / "local" / ("resources" + extension), canonical)
    (custom / "resources.json").write_text(json.dumps({"task_execution": {"worker_timeout": 99}}))
    monkeypatch.setattr(
        "nvflare.private.fed.app.fl_conf.BaseClientDeployer", lambda: SimpleNamespace(build=lambda _: None)
    )
    if relative_workspace:
        monkeypatch.chdir(tmp_path)
    workspace_root = "." if relative_workspace else str(tmp_path)
    workspace = Workspace(workspace_root, "site")
    args = argparse.Namespace(job_id=None, workspace=workspace_root)
    ConfigService.reset()
    try:
        starter = FLClientStarterConfiger(workspace, args, ["uid=site"])
        starter.configure()
        assert isinstance(starter.components["resource_manager"], ListResourceManager)
        assert starter.components["resource_manager"].resources == {}
        assert starter.config_data["task_execution"]["worker_timeout"] == 33
        assert ConfigService.get_section(SystemConfigs.RESOURCES_CONF)["task_execution"]["worker_timeout"] == 33
        assert TaskExecutionPolicy.from_workspace(workspace).worker_timeout == 33
    finally:
        ConfigService.reset()


@pytest.mark.parametrize("extension", [".conf", ".yaml", ".conf.default", ".yaml.default"])
def test_worker_authorization_and_log_opt_out_use_canonical_site_formats(tmp_path, extension):
    (tmp_path / "startup").mkdir()
    custom = tmp_path / "local" / "custom"
    custom.mkdir(parents=True)
    resources = tmp_path / "local" / ("resources" + extension)
    _write_site_config(resources, {"class_allow_list": ["nvflare."], "allow_log_streaming": False})
    (custom / "resources.json").write_text(json.dumps({"class_allow_list": ["*"], "allow_log_streaming": True}))
    workspace = Workspace(str(tmp_path), "site")
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site")
    context.set_prop(FLContextKey.WORKSPACE_ROOT, str(tmp_path))
    context.set_prop(FLContextKey.JOB_META, {"byoc": False})
    ConfigService.reset()
    try:
        bootstrap = SimpleNamespace(
            to_dict=lambda: {"executor": {"path": "nvflare.app_common.np.np_trainer.NPTrainer"}, "components": []}
        )
        _authorize_bootstrap(bootstrap, workspace, context)
        denied = SimpleNamespace(to_dict=lambda: {"executor": {"path": "custom.Component"}, "components": []})
        with pytest.raises(ComponentNotAuthorized, match="not in allow_list"):
            _authorize_bootstrap(denied, workspace, context)
        assert is_log_streaming_allowed(context) is False
    finally:
        ConfigService.reset()
