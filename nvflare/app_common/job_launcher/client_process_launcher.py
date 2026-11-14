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

from nvflare.apis.job_launcher_spec import JobLaunchOptions
from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.utils.job_launcher_utils import generate_client_command


class ClientProcessJobLauncher(ProcessJobLauncher):
    def _task_coordinator_options(self, job_meta, fl_ctx):
        """Keep CP-host coordinators CPU-only and reject unenforceable budgets."""
        from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES
        from nvflare.utils.job_launcher_utils import get_cj_resource_spec

        if any(get_cj_resource_spec(job_meta, fl_ctx.get_identity_name()).values()):
            raise ValueError("process CJ cannot enforce cj_resource_spec budgets; use a Slurm or Docker CJ")
        environment = os.environ.copy()
        environment.update({name: "none" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in GPU_ENVIRONMENT_NAMES})
        return JobLaunchOptions(environment=environment, apply_study_runtime=False)

    def get_command(self, job_meta, fl_ctx) -> str:
        return generate_client_command(fl_ctx)
