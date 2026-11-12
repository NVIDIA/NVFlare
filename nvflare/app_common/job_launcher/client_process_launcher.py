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

import secrets

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.job_launcher_spec import publish_job_handle
from nvflare.app_common.job_launcher.process_launcher import ProcessJobLauncher
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import get_client_execution_lifetime
from nvflare.utils.job_launcher_utils import generate_client_command, get_cj_resource_spec


class ClientProcessJobLauncher(ProcessJobLauncher):
    def launch_job(self, job_meta, fl_ctx):
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        job_id = job_meta.get(JobConstants.JOB_ID)
        config = (
            ConfigService.load_config_dict(workspace.get_client_app_config_file_path(job_id), raise_exception=False)
            or {}
        )
        if get_client_execution_lifetime(config) != "task":
            return super().launch_job(job_meta, fl_ctx)

        from nvflare.app_opt.job_launcher.slurm.launcher import _TaskCJHandle
        from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend
        from nvflare.app_opt.job_launcher.slurm.task_policy import component_args
        from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService
        from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy

        policy = TaskExecutionPolicy.from_workspace(workspace)
        policy.validate_job_launcher(self)
        if any(value for value in get_cj_resource_spec(job_meta, fl_ctx.get_identity_name()).values()):
            raise ValueError("process CJ cannot enforce cj_resource_spec budgets; use a Slurm or Docker CJ")
        data = ConfigService.load_config_dict(workspace.get_resources_file_path())
        backend = _SlurmTaskBackend(**component_args(data["task_execution"]["launcher"]))
        backend.bind_job(job_id, fl_ctx.get_identity_name(), workspace, job_meta)
        # The CP-owned Slurm backend already resolved study policy for workers.
        environment = self._job_environment(job_meta, fl_ctx, include_study_runtime=False)
        for name in list(environment):
            if name.startswith(("SLURM_", "SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_")):
                environment.pop(name)
        environment.update({name: "none" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in GPU_ENVIRONMENT_NAMES})
        cell = fl_ctx.get_engine().get_cell()
        service = SlurmTaskService.get_or_create(cell)
        self._task_service = service
        token = secrets.token_hex(32)
        owner = service.add_job(backend, job_id, fl_ctx.get_identity_name(), token)
        environment.update({ENV_TASK_OWNER: cell.get_fqcn(), ENV_TASK_TOKEN: token})
        try:
            native = self._launch_job(job_meta, fl_ctx, environment)
        except BaseException:
            owner.release()
            raise
        owner.attach_cj(native)
        handle = _TaskCJHandle(native, owner, policy.finalization_timeout, allow_restart=False)
        try:
            publish_job_handle(handle, fl_ctx)
        except BaseException:
            handle.terminate()
            handle.wait()
            raise
        return handle

    def handle_event(self, event_type, fl_ctx):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.SYSTEM_END:
            service = getattr(self, "_task_service", None)
            if service is not None:
                service.shutdown(timeout=30.0)

    def get_command(self, job_meta, fl_ctx) -> str:
        return generate_client_command(fl_ctx)
