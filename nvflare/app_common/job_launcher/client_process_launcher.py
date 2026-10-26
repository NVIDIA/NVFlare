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

from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.job_config.task_execution import validate_process_task_resources
from nvflare.utils.job_launcher_utils import generate_client_command, get_cj_resource_spec, has_cj_resource_spec


class ClientProcessJobLauncher(ProcessJobLauncher):
    def get_command(self, job_meta, fl_ctx) -> str:
        cj_resources = get_cj_resource_spec(job_meta, fl_ctx.get_identity_name())
        if any(key in cj_resources for key in ("num_of_cpus", "memory")) or cj_resources.get("num_of_gpus", 0):
            raise ValueError("Process CJ cannot enforce CPU/memory limits or assign a CJ GPU")
        if has_cj_resource_spec(job_meta, fl_ctx.get_identity_name()):
            validate_process_task_resources(job_meta, fl_ctx.get_identity_name())
        return generate_client_command(fl_ctx)
