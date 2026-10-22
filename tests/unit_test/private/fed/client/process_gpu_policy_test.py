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
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.resource_managers.gpu_resource_manager import GPUResourceManager
from nvflare.app_common.resource_managers.list_resource_manager import ListResourceManager
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

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
                "args": {
                    "gpu_allocator": {
                        "path": "nvflare.app_common.resource_managers.gpu_reservation_ledger.GpuReservationLedger",
                        "args": {
                            "pool_id": "task-only",
                            "devices": [GPU],
                            "reservation_root": str(tmp_path / "pool"),
                        },
                    }
                },
            }
        },
    }


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
    ClientProcessJobLauncher().prepare_environment({}, context, legacy_environment)
    assert legacy_environment == {"CUDA_VISIBLE_DEVICES": "0"}


def test_cp_rejects_stale_live_gpu_pool_even_when_site_file_requests_zero(tmp_path):
    workspace = _workspace(tmp_path, _config(tmp_path))
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    stale_manager = GPUResourceManager(num_of_gpus=1, mem_per_gpu_in_GiB=1, ignore_host=True)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: stale_manager))
    with pytest.raises(ValueError, match="zero job GPU capacity"):
        ClientProcessJobLauncher().prepare_environment({}, context, {})


@pytest.mark.parametrize("capacity", [None, True, -1, 1])
def test_cp_rejects_unknown_or_nonzero_gpu_capacity_from_custom_manager(tmp_path, capacity):
    class SiteResourceManager(ListResourceManager):
        def get_gpu_capacity(self):
            return capacity

    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, _config(tmp_path)))
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: SiteResourceManager({})))
    environment = {"CUDA_VISIBLE_DEVICES": "unchanged"}
    with pytest.raises(ValueError, match="zero job GPU capacity"):
        ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment == {"CUDA_VISIBLE_DEVICES": "unchanged"}


def test_cp_accepts_custom_manager_declaring_zero_gpu_capacity(tmp_path):
    class SiteResourceManager(ListResourceManager):
        def get_gpu_capacity(self):
            return 0

    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, _workspace(tmp_path, _config(tmp_path)))
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: SiteResourceManager({"cpu": [1]})))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_only_process_policy_still_rejects_gpu_tasks():
    with pytest.raises(ValueError, match="CPU Process"):
        TaskExecutionPolicy.from_config({}).task_resources(
            {"launcher_spec": {"site": {"process": {"num_of_gpus": 1}}}}, "site"
        )


@pytest.mark.parametrize("extension", [".json", ".conf", ".yml", ".json.default", ".conf.default", ".yml.default"])
def test_cp_and_cj_use_same_gpu_allocator_from_supported_site_formats(tmp_path, extension):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / ("resources" + extension)).write_text(json.dumps(_config(tmp_path)))
    workspace = Workspace(str(tmp_path), "site")
    policy = TaskExecutionPolicy.from_workspace(workspace)
    assert policy.launcher.gpu_allocator is not None
    assert policy.task_resources({"launcher_spec": {"default": {"process": {"num_of_gpus": 1}}}}, "site").gpu_count == 1
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: ListResourceManager({})))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_nominal_json_path_is_resolved_by_config_service_before_assuming_no_policy(tmp_path):
    nominal = tmp_path / "resources.json"
    nominal.with_suffix(".conf").write_text(json.dumps(_config(tmp_path)))
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(nominal))
    assert TaskExecutionPolicy.from_workspace(workspace).launcher.gpu_allocator is not None


def test_canonical_conf_is_not_shadowed_by_nested_json_in_cp_or_cj(tmp_path):
    (tmp_path / "startup").mkdir()
    custom = tmp_path / "local" / "custom"
    custom.mkdir(parents=True)
    (tmp_path / "local" / "resources.conf").write_text(json.dumps(_config(tmp_path)))
    (custom / "resources.json").write_text("{}")
    workspace = Workspace(str(tmp_path), "site")
    assert TaskExecutionPolicy.from_workspace(workspace).launcher.gpu_allocator is not None
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: ListResourceManager({})))
    environment = {"CUDA_VISIBLE_DEVICES": "all"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
