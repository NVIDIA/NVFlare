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

"""Slurm binding and gateway requirements for ordinary coordinator launchers."""

import copy
from dataclasses import replace

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.client.task_runtime import TaskJobRuntime


def prepare_gateway_runtime(job_meta, fl_ctx, options):
    from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService

    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend
    from nvflare.app_opt.job_launcher.slurm.task_policy import component_args

    # CJ uses the proxy; CP needs a separately bound physical backend per job.
    site_config = ConfigService.load_config_dict(workspace.get_resources_file_path())
    backend = _SlurmTaskBackend(**component_args(site_config["task_execution"]["launcher"]))
    backend.bind_job(job_meta[JobConstants.JOB_ID], fl_ctx.get_identity_name(), workspace, job_meta)
    environment = dict(options.environment or {})
    for name in tuple(environment):
        if name.startswith(("SLURM_", "SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_")):
            environment.pop(name)
    return TaskJobRuntime(
        backend,
        SlurmTaskService,
        ENV_TASK_OWNER,
        ENV_TASK_TOKEN,
        copy.deepcopy(job_meta),
        replace(options, environment=environment),
        30.0,
    )
