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

import os

from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.app_common.resource_managers.gpu_resource_manager import GPUResourceManager
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.utils.job_launcher_utils import generate_client_command


class ClientProcessJobLauncher(ProcessJobLauncher):
    def prepare_environment(self, job_meta, fl_ctx, environment):
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        path = workspace.get_resources_file_path()
        data = ConfigService.load_config_dict(path) if path and os.path.isfile(path) else {}
        launcher = data.get("task_execution", {}).get("launcher", {})
        if launcher.get("args", {}).get("gpu_partition") is None:
            return
        from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

        policy = TaskExecutionPolicy.from_workspace(workspace)
        if policy.launcher.launch_mode != "process" or getattr(policy.launcher, "gpu_pool", None) is None:
            raise ValueError("Process task GPU profile requires the site-owned ProcessTaskLauncher")
        manager = fl_ctx.get_engine().get_component(SystemComponents.RESOURCE_MANAGER)
        if type(manager) is not GPUResourceManager or manager.resources:
            raise ValueError("Process task GPU profile requires an empty live CP GPUResourceManager; restart the site")
        # This task-exclusive site profile exposes no GPU to any resident CJ.
        # Existing job-mode deployments without the opt-in keep their behavior.
        environment.update(
            CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none"
        )

    def get_command(self, job_meta, fl_ctx) -> str:
        return generate_client_command(fl_ctx)
