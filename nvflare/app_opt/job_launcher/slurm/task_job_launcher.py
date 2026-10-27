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
"""Slurm CP adapter with separate CJ and per-task compute allocations."""

import copy
import math
import secrets
from dataclasses import replace

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobReturnCode, publish_job_handle
from nvflare.apis.utils.format_check import check_job_id
from nvflare.app_opt.job_launcher.slurm.config import JobResources, SlurmLauncherError
from nvflare.app_opt.job_launcher.slurm.launcher import ClientSlurmJobLauncher
from nvflare.app_opt.job_launcher.slurm.task_policy import TASK_LAUNCHER_PATH, component_args, component_path
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.defs import ExecutionLifetime
from nvflare.job_config.task_execution import get_client_execution_lifetime
from nvflare.utils.job_launcher_utils import (
    get_cj_resource_spec,
    portable_memory_to_mib,
    validate_cj_execution_lifetime,
)

_GPU_ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")


class _TaskCJHandle(JobHandleSpec):
    """CJ exit alone cannot authorize recovery of independent task allocations."""

    def __init__(self, handle, owner):
        self.handle = handle
        self.owner = owner

    def terminate(self):
        self.owner.release()
        return self.handle.terminate()

    def _terminate_for_heartbeat_cleanup(self):
        self.owner.release()
        return self.handle._terminate_for_heartbeat_cleanup()

    def poll(self):
        code = self.handle.poll()
        if code != JobReturnCode.UNKNOWN:
            self.owner.release()
            if self.owner.thread.is_alive():
                return JobReturnCode.UNKNOWN
        return code

    def wait(self):
        self.handle.wait()
        self.owner.release()
        self.owner.thread.join()


class ClientSlurmTaskJobLauncher(ClientSlurmJobLauncher):
    def __init__(self, *, cj_cpu_cores=1, cj_memory_mb=1024, cj_gpu_count=0, **kwargs):
        super().__init__(**kwargs)
        from nvflare.apis.task_launcher_spec import TaskResourceRequest

        request = TaskResourceRequest(cj_cpu_cores, cj_memory_mb, cj_gpu_count)
        if request.cpu_cores is None or not float(request.cpu_cores).is_integer():
            raise ValueError("Slurm CJ CPU request must be whole cores")
        self.cj_resources = JobResources(
            cpus_per_node=int(request.cpu_cores),
            mem_per_node=request.memory_mb,
            gpus_per_node=request.gpu_count,
            pending_timeout=self.config.pending_timeout,
        )

    def _cj_resources_for_job(self, job_meta, site_name):
        requested = get_cj_resource_spec(job_meta, site_name)
        if requested.get("mem_per_gpu_in_GiB", 0):
            raise ValueError("Slurm CJ allocations cannot enforce per-GPU memory requirements")
        # Slurm admits whole cores. Allocate at least the requested CPU share.
        cpus = math.ceil(requested.get("num_of_cpus", self.cj_resources.cpus_per_node))
        memory = (
            portable_memory_to_mib(requested["memory"]) if "memory" in requested else self.cj_resources.mem_per_node
        )
        return replace(
            self.cj_resources,
            cpus_per_node=cpus,
            mem_per_node=memory,
            gpus_per_node=requested.get("num_of_gpus", self.cj_resources.gpus_per_node),
        )

    def _prepare_launch(self, job_meta, fl_ctx):
        if not isinstance(job_meta, dict):
            raise ValueError("Slurm job metadata must be a dictionary")
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        job_id = job_meta.get(JobConstants.JOB_ID)
        check_job_id(job_id)
        config = ConfigService.load_config_dict(workspace.get_client_app_config_file_path(job_id))
        lifetime = get_client_execution_lifetime(config)
        validate_cj_execution_lifetime(job_meta, fl_ctx.get_identity_name(), lifetime)
        if lifetime == ExecutionLifetime.JOB:
            plan = super()._build_launch_plan(job_meta, fl_ctx)
            if plan.resources.gpus_per_node == 0:
                environment = {
                    **plan.study_env,
                    **{name: "none" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in _GPU_ENV},
                }
                return replace(plan, study_env=environment), None
            return plan, None
        from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend

        data = ConfigService.load_config_dict(workspace.get_resources_file_path())
        launcher_config = data.get("task_execution", {}).get("launcher", {})
        if component_path(launcher_config) != TASK_LAUNCHER_PATH:
            raise ValueError("Slurm CPU CJ requires the exact site SlurmTaskLauncher")
        task_launcher = _SlurmTaskBackend(**component_args(launcher_config))
        task_launcher.validate_site_config(data)
        runtime = self._load_study_runtime(job_meta.get(JobMetaKey.STUDY.value))
        task_launcher.bind_job(job_id, fl_ctx.get_identity_name(), workspace, job_meta, runtime=runtime)
        cj_resources = self._cj_resources_for_job(job_meta, fl_ctx.get_identity_name())
        # The original metadata stays intact on disk and in the CJ. This copy
        # is solely the physical CJ allocation, never the compute request.
        cj_meta = copy.deepcopy(job_meta)
        cj_meta[JobMetaKey.RESOURCE_SPEC.value] = {}
        from nvflare.utils.job_launcher_utils import get_job_launcher_spec

        compute = get_job_launcher_spec(job_meta, fl_ctx.get_identity_name(), "slurm")
        timing = {key: compute[key] for key in ("time", "pending_timeout") if key in compute}
        if "pending_timeout" in timing:
            timing["pending_timeout"] = min(timing["pending_timeout"], self.config.pending_timeout)
        cj_meta[JobMetaKey.JOB_LAUNCHER_SPEC.value] = {fl_ctx.get_identity_name(): {"slurm": timing}}
        plan = super()._build_launch_plan(cj_meta, fl_ctx, runtime=runtime)
        if plan.sandbox != "none" or plan.image:
            raise ValueError("Slurm task CJ requires the bare shared-filesystem profile")
        environment = dict(plan.study_env)
        if cj_resources.gpus_per_node == 0:
            environment.update({name: "none" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in _GPU_ENV})
        return (
            replace(
                plan,
                resources=replace(
                    cj_resources, time_limit=plan.resources.time_limit, pending_timeout=plan.resources.pending_timeout
                ),
                study_env=environment,
                additional_node_command=(),
                node_app_dir=None,
                execution_lifetime=ExecutionLifetime.TASK,
            ),
            task_launcher,
        )

    def _build_launch_plan(self, job_meta, fl_ctx):
        return self._prepare_launch(job_meta, fl_ctx)[0]

    def launch_job(self, job_meta, fl_ctx):
        if self.manager is None:
            raise SlurmLauncherError("nested Slurm job launch is unavailable inside a Slurm child process")
        plan, backend = self._prepare_launch(job_meta, fl_ctx)
        if backend is None:
            return self.manager.launch(plan)
        from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService

        cell = fl_ctx.get_engine().get_cell()
        service = SlurmTaskService.get_or_create(cell)
        self._task_service = service
        token = secrets.token_hex(32)
        owner = service.add_job(backend, plan.job_id, plan.site_name, token)
        plan = replace(
            plan,
            study_secret_env={
                **plan.study_secret_env,
                ENV_TASK_OWNER: cell.get_fqcn(),
                ENV_TASK_TOKEN: token,
            },
        )
        try:
            handle = self.manager.launch(plan)
        except BaseException:
            owner.release()
            raise
        owner.attach_cj(handle)
        result = _TaskCJHandle(handle, owner)
        try:
            publish_job_handle(result, fl_ctx)
        except BaseException:
            result.terminate()
            result.wait()
            raise
        return result

    def handle_event(self, event_type, fl_ctx):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.SYSTEM_END:
            service = getattr(self, "_task_service", None)
            if service is not None:
                service.shutdown()
