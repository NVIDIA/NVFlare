# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.task_launcher_spec import UnsupportedTaskResourceError
from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.gpu_utils import get_cuda_device_environment
from nvflare.job_config.task_execution import validate_process_task_resources
from nvflare.utils.job_launcher_utils import (
    generate_client_command,
    get_cj_resource_spec,
    get_resource_manager_spec,
    has_cj_resource_spec,
)


class ClientProcessJobLauncher(ProcessJobLauncher):
    def prepare_environment(self, job_meta, fl_ctx, environment):
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        path = workspace.get_resources_file_path()
        data = ConfigService.load_config_dict(path, prefer_exact_file=True) if path else {}
        launcher = data.get("task_execution", {}).get("launcher", {})
        if launcher.get("args", {}).get("gpu_devices") is None:
            return
        from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

        policy = TaskExecutionPolicy.from_workspace(workspace)
        if policy.launcher.launch_mode != "process" or getattr(policy.launcher, "gpu_devices", None) is None:
            raise ValueError("Process task GPU profile requires the site-owned ProcessTaskLauncher")
        if get_resource_manager_spec(job_meta, fl_ctx.get_identity_name()).get("num_of_gpus", 0):
            raise ValueError(
                "Process task GPU profile rejects job-level GPU requests; use process launcher_spec for task GPUs"
            )
        if has_cj_resource_spec(job_meta, fl_ctx.get_identity_name()):
            resources = policy.task_resources(job_meta, fl_ctx.get_identity_name())
            if resources.gpu_count > len(policy.launcher.gpu_devices):
                raise UnsupportedTaskResourceError(
                    "Process task requests more GPUs than its static visibility assignment"
                )
        # GPU visibility belongs to disposable workers; resident CJs receive no CUDA devices.
        # Existing job-mode deployments without the opt-in keep their behavior.
        environment.update(get_cuda_device_environment(()))
        environment.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none")

    def get_command(self, job_meta, fl_ctx) -> str:
        cj_resources = get_cj_resource_spec(job_meta, fl_ctx.get_identity_name())
        if any(key in cj_resources for key in ("num_of_cpus", "memory")) or cj_resources.get("num_of_gpus", 0):
            raise ValueError("Process CJ cannot enforce CPU/memory limits or assign a CJ GPU")
        if has_cj_resource_spec(job_meta, fl_ctx.get_identity_name()):
            validate_process_task_resources(job_meta, fl_ctx.get_identity_name())
        return generate_client_command(fl_ctx)
