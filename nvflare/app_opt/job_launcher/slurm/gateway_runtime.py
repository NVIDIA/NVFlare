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
import os
import sys
from dataclasses import replace
from urllib.parse import urlsplit

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.job_launcher_spec import JobLaunchOptions, JobProcessArgs
from nvflare.apis.task_launcher_spec import TaskResourceRequest
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES
from nvflare.private.fed.task_worker.parent_runtime import TaskJobRuntime
from nvflare.utils.job_launcher_utils import get_cj_resource_spec, get_job_launcher_spec, portable_memory_to_bytes


def docker_cj_limits(launcher, job_meta, site):
    requested = get_cj_resource_spec(job_meta, site)
    if requested.get("num_of_gpus", 0) or requested.get("mem_per_gpu_in_GiB", 0):
        raise ValueError("gateway Docker CJ must be CPU-only; request GPUs for tasks")
    defaults = launcher.default_job_container_kwargs
    unsupported = set(defaults) - {"nano_cpus", "mem_limit", "cap_drop", "security_opt"}
    if unsupported:
        raise ValueError(f"gateway Docker CJ has unsupported site container settings: {sorted(unsupported)}")
    cpu = requested.get("num_of_cpus", defaults.get("nano_cpus", 0) / 1_000_000_000)
    memory = portable_memory_to_bytes(requested["memory"]) if "memory" in requested else defaults.get("mem_limit")
    TaskResourceRequest(cpu, memory)
    if memory is None or int(cpu * 1_000_000_000) < 1:
        raise ValueError("gateway Docker CJ requires explicit CPU and memory limits")
    return {"nano_cpus": int(cpu * 1_000_000_000), "mem_limit": memory}


def prepare_gateway_runtime(backend, launcher, job_meta, fl_ctx):
    from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService

    backend.validate_job_launcher(launcher)
    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend
    from nvflare.app_opt.job_launcher.slurm.task_policy import component_args

    # A CJ uses the proxy, but CP must bind a separate physical backend for
    # every job. Never dispatch CP requests back through the CJ proxy.
    site_config = ConfigService.load_config_dict(workspace.get_resources_file_path())
    backend = _SlurmTaskBackend(**component_args(site_config["task_execution"]["launcher"]))
    job_id, site = job_meta[JobConstants.JOB_ID], fl_ctx.get_identity_name()
    meta = copy.deepcopy(job_meta)
    if backend.cj_runtime == "process":
        if any(get_cj_resource_spec(meta, site).values()):
            raise ValueError("process CJ cannot enforce cj_resource_spec budgets; use a Slurm or Docker CJ")
        environment = os.environ.copy()
        options = JobLaunchOptions(apply_study_runtime=False)
    else:
        if sys.platform != "linux" or os.getuid() == 0:
            raise ValueError("gateway Docker CJ requires Linux host networking and a nonroot CP UID")
        if (
            launcher.workspace != workspace.get_root_dir()
            or launcher.workspace != backend.manager.config.workspace_path
        ):
            raise ValueError("Docker CP/CJ/task workspace paths must match exactly")
        if launcher.network != "host" or not launcher.default_image:
            raise ValueError("gateway Docker CJ requires host networking and a site-owned default_image")
        if get_job_launcher_spec(meta, site, "docker"):
            raise ValueError("gateway Docker CJ uses site-owned settings; job Docker options are unsupported")
        parent = (fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS) or {}).get(JobProcessArgs.PARENT_URL)
        if not parent or urlsplit(str(parent[1])).scheme not in ("tcp", "stcp"):
            raise ValueError("gateway Docker CJ requires a TCP or mTLS TCP parent connection")
        limits = docker_cj_limits(launcher, meta, site)
        client = launcher._get_docker_client()
        info = client.info()
        if (
            client.api.base_url != "http+docker://localhost"
            or info.get("OSType") != "linux"
            or not all(info.get(key) is True for key in ("MemoryLimit", "CpuCfsPeriod", "CpuCfsQuota"))
        ):
            raise ValueError("gateway Docker daemon must be local and enforce Linux CPU/memory limits")
        # This physical launch copy cannot turn compute budgets into CJ budgets.
        meta["resource_spec"] = {}
        meta["cj_resource_spec"] = {site: {}}
        meta["launcher_spec"] = {site: {"docker": {"image": launcher.default_image}}}
        environment = dict(launcher.default_job_env)
        options = JobLaunchOptions(
            apply_study_runtime=False,
            workspace_mount=workspace.get_root_dir(),
            parent_host=backend.get_worker_listener_host(),
            container_kwargs={**limits, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"]},
        )
    for name in tuple(environment):
        if name.startswith(("SLURM_", "SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_")):
            environment.pop(name)
    environment.update(
        {
            name: (
                "void"
                if name == "NVIDIA_VISIBLE_DEVICES" and backend.cj_runtime == "docker"
                else "none" if name == "NVIDIA_VISIBLE_DEVICES" else ""
            )
            for name in GPU_ENVIRONMENT_NAMES
        }
    )
    options = replace(options, environment=environment)
    backend.bind_job(job_id, site, workspace, job_meta)
    return TaskJobRuntime(backend, SlurmTaskService, ENV_TASK_OWNER, ENV_TASK_TOKEN, meta, options, 30.0)
