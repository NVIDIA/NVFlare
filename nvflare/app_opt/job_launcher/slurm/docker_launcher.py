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

"""CPU Docker coordinators on the gateway; all Slurm authority stays in CP."""

import copy
import ipaddress
import os
import re
import secrets
import sys
from urllib.parse import urlsplit

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.job_launcher_spec import JobProcessArgs, publish_job_handle
from nvflare.apis.task_launcher_spec import TaskResourceRequest
from nvflare.apis.utils.format_check import check_job_id
from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher, _rewrite_parent_url
from nvflare.app_opt.job_launcher.slurm.config import _require_positive_number, normalize_slurm_workspace_path
from nvflare.app_opt.job_launcher.slurm.launcher import _TaskCJHandle
from nvflare.app_opt.job_launcher.slurm.task_service import ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import get_client_execution_lifetime
from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy
from nvflare.utils.job_launcher_utils import get_cj_resource_spec, get_job_launcher_spec, portable_memory_to_bytes


class _GatewayContainerLauncher(ClientDockerJobLauncher):
    """One launch's immutable Docker inputs, never shared between jobs."""

    def _get_docker_client(self):
        client = super()._get_docker_client()
        if client.api.base_url != "http+docker://localhost":
            raise ValueError("gateway Docker CJ requires a local Unix-socket Docker daemon")
        info = client.info()
        if info.get("OSType") != "linux" or not all(
            info.get(key) is True for key in ("MemoryLimit", "CPUCfsPeriod", "CPUCfsQuota")
        ):
            raise ValueError("gateway Docker daemon must enforce Linux CPU quotas and memory limits")
        return client

    def _resolve_study_runtime(self, study):
        # The CP's Slurm backend resolves study policy. No study secret, mount,
        # container option or image is inherited by the coordinator.
        return None

    def get_module_args(self, job_args):
        job_args, _ = _rewrite_parent_url(job_args, self.parent_host)
        return super().get_module_args(job_args)

    def _job_created(self, job_handle, fl_ctx):
        self.owner.attach_cj(job_handle)
        self.task_handle = _TaskCJHandle(job_handle, self.owner, self.cleanup_timeout)
        try:
            publish_job_handle(self.task_handle, fl_ctx)
        except BaseException:
            self.task_handle.terminate()
            self.task_handle.wait()
            raise


class ClientDockerSlurmJobLauncher(ClientDockerJobLauncher):
    """Opt-in, task-only Linux gateway profile with an isolated CPU-only CJ.

    CP, Docker daemon and Slurm workers see ``workspace`` at the same canonical
    absolute path. Host networking exposes CJ's dynamically allocated listener;
    ``parent_host`` and the task launcher's ``worker_listener_host`` are supplied
    by the site. Images and startup policy are site owned, not job selectable.
    """

    def __init__(
        self,
        *,
        workspace,
        cj_image,
        parent_host,
        cj_python_path="/usr/local/bin/python",
        cj_cpu_cores=1,
        cj_memory_mb=1024,
        timeout=30,
        task_shutdown_timeout=30.0,
    ):
        canonical = normalize_slurm_workspace_path(workspace)
        if os.path.realpath(workspace) != workspace or canonical != workspace:
            raise ValueError("Docker Slurm workspace must be a canonical absolute shared path")
        if not isinstance(cj_image, str) or not cj_image.strip():
            raise ValueError("Docker Slurm cj_image must be an explicit site-owned image")
        if not isinstance(parent_host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", parent_host):
            raise ValueError("Docker Slurm parent_host must be a gateway DNS name or IPv4 address without a port")
        try:
            wildcard = ipaddress.ip_address(parent_host).is_unspecified
        except ValueError:
            wildcard = parent_host == "0"
        if wildcard:
            raise ValueError("Docker Slurm parent_host cannot be a wildcard address")
        request = TaskResourceRequest(cj_cpu_cores, cj_memory_mb)
        if request.cpu_cores is None or request.memory_mb is None:
            raise ValueError("Docker CJ requires explicit CPU and memory budgets")
        super().__init__(
            workspace=workspace,
            network="host",
            default_python_path=cj_python_path,
            timeout=_require_positive_number(timeout, "timeout"),
        )
        self.network = "host"  # Do not inherit NVFL_DOCKER_NETWORK from a bridge deployment.
        self.cj_image, self.parent_host, self.cj_resources = cj_image, parent_host, request
        self.task_shutdown_timeout = _require_positive_number(task_shutdown_timeout, "task_shutdown_timeout")

    def _cj_limits(self, job_meta, site_name):
        requested = get_cj_resource_spec(job_meta, site_name)
        if requested.get("num_of_gpus", 0) or requested.get("mem_per_gpu_in_GiB", 0):
            raise ValueError("gateway Docker CJ must be CPU-only; request GPUs for Slurm tasks instead")
        nano_cpus = int(requested.get("num_of_cpus", self.cj_resources.cpu_cores) * 1_000_000_000)
        if nano_cpus < 1:
            raise ValueError("Docker CJ CPU budget must be at least one nanocpu")
        return {
            "nano_cpus": nano_cpus,
            "mem_limit": (
                portable_memory_to_bytes(requested["memory"])
                if "memory" in requested
                else self.cj_resources.memory_mb * 1024 * 1024
            ),
        }

    def launch_job(self, job_meta, fl_ctx):
        if sys.platform != "linux" or os.getuid() == 0:
            raise ValueError("gateway Docker CJ requires Linux host networking and a nonroot CP UID")
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace is None or workspace.get_root_dir() != self.workspace:
            raise ValueError("Docker CP/CJ/task workspace paths must match exactly")
        job_id = job_meta.get(JobConstants.JOB_ID)
        check_job_id(job_id)
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS) or {}
        parent = job_args.get(JobProcessArgs.PARENT_URL)
        if not parent or urlsplit(str(parent[1])).scheme not in ("tcp", "stcp"):
            raise ValueError("gateway Docker CJ requires a TCP or mTLS TCP parent connection")
        config = ConfigService.load_config_dict(workspace.get_client_app_config_file_path(job_id))
        if get_client_execution_lifetime(config) != "task":
            raise ValueError("ClientDockerSlurmJobLauncher supports only execution_lifetime='task'")
        policy = TaskExecutionPolicy.from_workspace(workspace)
        policy.validate_job_launcher(self)
        if get_job_launcher_spec(job_meta, fl_ctx.get_identity_name(), "docker"):
            raise ValueError("gateway Docker CJ uses site-owned settings; job Docker options are unsupported")
        limits = self._cj_limits(job_meta, fl_ctx.get_identity_name())
        from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend
        from nvflare.app_opt.job_launcher.slurm.task_policy import component_args

        data = ConfigService.load_config_dict(workspace.get_resources_file_path())
        backend = _SlurmTaskBackend(**component_args(data["task_execution"]["launcher"]))
        backend.bind_job(job_id, fl_ctx.get_identity_name(), workspace, job_meta)
        cell = fl_ctx.get_engine().get_cell()
        service = SlurmTaskService.get_or_create(cell)
        self._task_service = service
        token = secrets.token_hex(32)
        owner = service.add_job(backend, job_id, fl_ctx.get_identity_name(), token)
        try:
            driver = _GatewayContainerLauncher(
                workspace=self.workspace,
                network="host",
                default_python_path=self.default_python_path,
                timeout=self.timeout,
                default_job_env={
                    **{name: "void" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in GPU_ENVIRONMENT_NAMES},
                    ENV_TASK_OWNER: cell.get_fqcn(),
                    ENV_TASK_TOKEN: token,
                },
                default_job_container_kwargs={
                    **limits,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges:true"],
                },
            )
            driver.network = "host"
            driver.WORKSPACE_MOUNT = self.workspace
            driver.parent_host, driver.owner = self.parent_host, owner
            driver.cleanup_timeout = self.task_shutdown_timeout
            driver._docker_client = self._docker_client
            # Preserve the original compute metadata on disk and in the backend.
            # This private copy is only the physical coordinator's Docker request.
            cj_meta = copy.deepcopy(job_meta)
            cj_meta["resource_spec"] = {}
            cj_meta["launcher_spec"] = {fl_ctx.get_identity_name(): {"docker": {"image": self.cj_image}}}
            driver.launch_job(cj_meta, fl_ctx)
            return driver.task_handle
        except BaseException:
            owner.release()
            raise

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            # Compute GPUs belong exclusively to Slurm; never reserve them for CJ.
            try:
                self._cj_limits(fl_ctx.get_prop(FLContextKey.JOB_META) or {}, fl_ctx.get_identity_name())
            except ValueError as error:
                fl_ctx.set_prop(FLContextKey.JOB_BLOCK_REASON, str(error), private=True, sticky=False)
            return
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.SYSTEM_END:
            service = getattr(self, "_task_service", None)
            if service is not None:
                service.shutdown(timeout=self.task_shutdown_timeout)
