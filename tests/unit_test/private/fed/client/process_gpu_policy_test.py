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
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.resource_managers.gpu_resource_manager import GPUResourceManager
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

GPU = "GPU-00000000-0000-0000-0000-000000000001"


def _config(tmp_path):
    return {
        "components": [
            {
                "id": "resource_manager",
                "path": "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager",
                "args": {"num_of_gpus": 0, "mem_per_gpu_in_GiB": 0},
            }
        ],
        "task_execution": {
            "launcher": {
                "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
                "args": {
                    "gpu_partition": {
                        "id": "task-only",
                        "devices": [GPU],
                        "exclusive": True,
                        "reservation_root": str(tmp_path / "pool"),
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


@pytest.mark.parametrize("gpu_count", [1, True, None])
def test_task_gpu_profile_rejects_nonempty_or_ambiguous_cp_job_pool(tmp_path, gpu_count):
    config = _config(tmp_path)
    config["components"][0]["args"]["num_of_gpus"] = gpu_count
    with pytest.raises(ValueError, match="num_of_gpus=0"):
        TaskExecutionPolicy.from_workspace(_workspace(tmp_path, config))


def test_cp_hides_gpus_after_study_overrides_only_for_opted_in_profile(tmp_path):
    workspace = _workspace(tmp_path, _config(tmp_path))
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    manager = GPUResourceManager(num_of_gpus=0, mem_per_gpu_in_GiB=0)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_component=lambda _: manager))
    environment = {"CUDA_VISIBLE_DEVICES": "all", "NVIDIA_VISIBLE_DEVICES": "all", "UNCHANGED": "yes"}
    ClientProcessJobLauncher().prepare_environment({}, context, environment)
    assert environment == {
        "CUDA_VISIBLE_DEVICES": "",
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
    with pytest.raises(ValueError, match="empty live CP"):
        ClientProcessJobLauncher().prepare_environment({}, context, {})


def test_cpu_only_process_policy_still_rejects_gpu_tasks():
    with pytest.raises(ValueError, match="CPU Process"):
        TaskExecutionPolicy.from_config({}).task_resources(
            {"launcher_spec": {"site": {"process": {"num_of_gpus": 1}}}}, "site"
        )
