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

"""Site-owned selection and resource policy for disposable task execution."""

import importlib
import os
import re
from dataclasses import dataclass

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.task_launcher_spec import TaskLauncherSpec, TaskResourceRequest
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.gpu_utils import get_cuda_device_environment
from nvflare.job_config.task_execution import _timeout, validate_task_resources
from nvflare.private.fed.task_worker.protocol import _require_payload_limit
from nvflare.private.fed.task_worker.supervisor import TaskSupervisorOptions
from nvflare.utils.job_launcher_utils import get_job_launcher_spec, get_resource_manager_spec

GPU_ENVIRONMENT_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}
FORBIDDEN_ENVIRONMENT_NAMES = (
    set(JobProcessEnv.ALL)
    | GPU_ENVIRONMENT_NAMES
    | {
        BOOTSTRAP_FILE_ENV_VAR,
        CLIENT_API_TYPE_KEY,
        "NVFLARE_CLIENT_API_CONFIG",
        "NVFLARE_CLIENT_API_CONFIG_PATH",
        "NVFLARE_CLIENT_API_SERVICE_CONF",
        "NVFLARE_DOCKER_TASK_SESSION",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "DOCKER_API_VERSION",
    }
)
PASSTHROUGH_ENVIRONMENT_NAMES = {
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "NVFLARE_SECURE_LOGGING",
    "FL_LOG_LEVEL",
}

_NATIVE_TASK_LAUNCHERS = {
    "docker": ("nvflare.app_opt.task_launcher.docker_launcher", "DockerTaskLauncher"),
    "k8s": ("nvflare.app_opt.task_launcher.k8s_launcher", "K8sTaskLauncher"),
    "slurm": ("nvflare.app_opt.job_launcher.slurm.task_launcher", "SlurmTaskLauncher"),
}


def _supported_native_launcher(launcher):
    """Import only the selected backend; optional backends stay optional."""
    target = _NATIVE_TASK_LAUNCHERS.get(launcher.launch_mode)
    if target is None:
        return False
    module_name, class_name = target
    return type(launcher) is getattr(importlib.import_module(module_name), class_name)


class _SiteBuilder(ComponentBuilder):
    def get_module_scanner(self):
        return ModuleScanner(["nvflare"], ["apis", "app_common"], True)


@dataclass(frozen=True)
class TaskExecutionPolicy:
    launcher: TaskLauncherSpec
    environment_variables: tuple[str, ...] = ()
    artifact_cleanup: str = "receipt"
    worker_timeout: float | None = None
    settlement_timeout: float = 10.0
    poll_interval: float = 0.05
    max_payload_bytes: int | None = None
    finalization_timeout: float = 30.0

    @classmethod
    def from_workspace(cls, workspace):
        # Do not consult APPLICATION_CONF or job-supplied resources. Only the
        # canonical site's resources.json (or its default fallback) owns policy.
        path = workspace.get_resources_file_path()
        data = ConfigService.load_config_dict(path) if path and os.path.isfile(path) else {}
        policy = cls.from_config(data.get("task_execution", {}))
        # Only the canonical site file may establish the CP/CJ backend. A
        # container environment or job meta cannot select its own authority.
        if _supported_native_launcher(policy.launcher):
            policy.launcher.validate_site_config(data)
            return policy
        launchers = []
        for item in data.get("components", []):
            if not isinstance(item, dict):
                continue
            path = item.get("path", item.get("class_path", item.get("name", "")))
            if isinstance(path, str) and path.endswith("JobLauncher"):
                launchers.append((path, item))
        if launchers and (
            len(launchers) != 1
            or launchers[0][0]
            not in {
                "ClientProcessJobLauncher",
                "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher",
            }
        ):
            raise ValueError("task and job launch modes must match; configure a supported site TaskLauncher")
        return policy

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            raise ValueError("site task_execution policy must be a dict")
        allowed = {
            "launcher",
            "environment_variables",
            "artifact_cleanup",
            "worker_timeout",
            "settlement_timeout",
            "poll_interval",
            "max_payload_bytes",
            "finalization_timeout",
        }
        if set(config) - allowed:
            raise ValueError(f"unknown site task_execution settings: {sorted(set(config) - allowed)}")
        names = config.get("environment_variables", [])
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names
        ):
            raise ValueError("environment_variables must be a list of valid environment names")
        if set(names) & FORBIDDEN_ENVIRONMENT_NAMES:
            raise ValueError(
                "task environment policy cannot forward federation credentials, API bootstrap, or GPU visibility"
            )
        cleanup = config.get("artifact_cleanup", "receipt")
        if cleanup not in ("receipt", "retain"):
            raise ValueError("artifact_cleanup must be 'receipt' or 'retain'")
        options = TaskSupervisorOptions(
            worker_timeout=_timeout("worker_timeout", config.get("worker_timeout")),
            settlement_timeout=config.get("settlement_timeout", 10.0),
            poll_interval=config.get("poll_interval", 0.05),
        )
        limit = config.get("max_payload_bytes")
        _require_payload_limit(limit)
        finalization_timeout = _timeout("finalization_timeout", config.get("finalization_timeout", 30.0))
        if finalization_timeout is None or finalization_timeout <= 0:
            raise ValueError("finalization_timeout must be finite and positive")
        launcher_config = config.get("launcher")
        launcher = ProcessTaskLauncher() if launcher_config is None else _SiteBuilder().build_component(launcher_config)
        if not isinstance(launcher, TaskLauncherSpec) or not (
            launcher.launch_mode == "process" or _supported_native_launcher(launcher)
        ):
            raise ValueError("task execution requires a supported Process, Docker, Kubernetes or Slurm TaskLauncher")
        if launcher.launch_mode != "process" and names:
            raise ValueError("native worker environment belongs to trusted backend configuration")
        return cls(
            launcher,
            tuple(dict.fromkeys(names)),
            cleanup,
            options.worker_timeout,
            options.settlement_timeout,
            options.poll_interval,
            limit,
            finalization_timeout,
        )

    def worker_environment(self):
        if self.launcher.launch_mode != "process":
            return {}
        allowed = PASSTHROUGH_ENVIRONMENT_NAMES | set(self.environment_variables)
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in allowed and name not in FORBIDDEN_ENVIRONMENT_NAMES
        }
        environment.update(
            {
                **get_cuda_device_environment(()),
                "HIP_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "ROCR_VISIBLE_DEVICES": "",
            }
        )
        return environment

    def task_resources(self, meta, site_name):
        native_mode = _supported_native_launcher(self.launcher)
        process_gpu = type(self.launcher) is ProcessTaskLauncher and self.launcher.gpu_allocator is not None
        validate_task_resources(meta, site_name, allow_gpu=native_mode or process_gpu)
        if (native_mode or process_gpu) and get_resource_manager_spec(meta, site_name).get("num_of_gpus", 0):
            raise ValueError("task GPU admission requires native launcher_spec; portable GPUs reserve the whole job")
        if process_gpu:
            for mode, field in (("docker", "num_of_gpus"), ("k8s", "num_of_gpus"), ("slurm", "gpus_per_node")):
                if get_job_launcher_spec(meta, site_name, mode).get(field, 0):
                    raise ValueError("Process GPU tasks require process launcher_spec")
            spec = get_job_launcher_spec(meta, site_name, "process")
            if set(spec) - {"num_of_gpus"}:
                raise ValueError("Process task GPU launcher_spec supports only whole-GPU counts")
            return TaskResourceRequest(gpu_count=spec.get("num_of_gpus", 0))
        if not native_mode or self.launcher.launch_mode != "docker":
            return TaskResourceRequest()
        spec = get_job_launcher_spec(meta, site_name, "docker")
        return TaskResourceRequest(gpu_count=spec.get("num_of_gpus", 0))
