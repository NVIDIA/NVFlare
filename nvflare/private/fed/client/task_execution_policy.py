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
from dataclasses import dataclass

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.task_launcher_spec import TaskLauncherSpec, TaskResourceRequest
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import _timeout, reject_job_gpu_reservation, validate_task_resources
from nvflare.private.fed.task_worker.protocol import _require_payload_limit
from nvflare.private.fed.task_worker.supervisor import TaskSupervisorOptions

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
_NATIVE_TASK_LAUNCHERS = {
    "k8s": ("nvflare.app_opt.task_launcher.k8s_launcher", "K8sTaskLauncher"),
    "slurm": ("nvflare.app_opt.job_launcher.slurm.task_launcher", "SlurmTaskLauncher"),
}
_NATIVE_JOB_LAUNCHERS = {
    "k8s": ("nvflare.app_opt.job_launcher.k8s_launcher", "ClientK8sJobLauncher"),
    "slurm": ("nvflare.app_opt.job_launcher.slurm.launcher", "ClientSlurmJobLauncher"),
}


def _supported_native_launcher(launcher):
    """Import only the selected backend; optional backends stay optional."""
    target = _NATIVE_TASK_LAUNCHERS.get(launcher.launch_mode)
    if target is None:
        return False
    module_name, class_name = target
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and (error.name == module_name or module_name.startswith(error.name + ".")):
            return False
        raise
    return type(launcher) is getattr(module, class_name, None)


def site_job_launchers(site_config):
    """Find launcher components using the supported component field spellings."""
    result = []
    for item in site_config.get("components", []):
        if not isinstance(item, dict):
            continue
        path = item.get("path", item.get("class_path", item.get("name", "")))
        if isinstance(path, str) and path.endswith("JobLauncher"):
            result.append((path, item))
    return result


class _SiteBuilder(ComponentBuilder):
    def get_module_scanner(self):
        return ModuleScanner(["nvflare"], ["apis", "app_common"], True)


@dataclass(frozen=True)
class TaskExecutionPolicy:
    launcher: TaskLauncherSpec
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
        policy.launcher.validate_site_config(data)
        return policy

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            raise ValueError("site task_execution policy must be a dict")
        if "environment_variables" in config:
            raise ValueError(
                "environment_variables is not supported; configure worker environment on the native backend"
            )
        allowed = {
            "launcher",
            "artifact_cleanup",
            "worker_timeout",
            "settlement_timeout",
            "poll_interval",
            "max_payload_bytes",
            "finalization_timeout",
        }
        if set(config) - allowed:
            raise ValueError(f"unknown site task_execution settings: {sorted(set(config) - allowed)}")
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
        if launcher_config is None:
            raise ValueError(
                "task execution requires an explicit site Kubernetes or Slurm TaskLauncher; "
                "local Process and Docker task execution are deferred; use job lifetime on local sites"
            )
        if isinstance(launcher_config, dict):
            # Give old full paths and short-name aliases the same actionable
            # error before importing a removed or never-shipped implementation.
            for field in ("path", "class_path", "name"):
                path = launcher_config.get(field)
                if isinstance(path, str) and path.rsplit(".", 1)[-1] in ("ProcessTaskLauncher", "DockerTaskLauncher"):
                    raise ValueError("local Process and Docker task execution are deferred; use job lifetime")
        launcher = _SiteBuilder().build_component(launcher_config)
        if not isinstance(launcher, TaskLauncherSpec) or not _supported_native_launcher(launcher):
            raise ValueError("task execution requires a supported Kubernetes or Slurm TaskLauncher")
        return cls(
            launcher=launcher,
            artifact_cleanup=cleanup,
            worker_timeout=options.worker_timeout,
            settlement_timeout=options.settlement_timeout,
            poll_interval=options.poll_interval,
            max_payload_bytes=limit,
            finalization_timeout=finalization_timeout,
        )

    def worker_environment(self):
        # Native backend configuration, never the resident CJ's environment,
        # owns worker credentials, GPU visibility and application variables.
        return {}

    def validate_job_launcher(self, launcher):
        """Fail closed before starting a CJ on an incompatible site runtime."""
        module_name, class_name = _NATIVE_JOB_LAUNCHERS[self.launcher.launch_mode]
        expected = getattr(importlib.import_module(module_name), class_name)
        if type(launcher) is not expected:
            raise ValueError("task lifetime requires the site's supported Kubernetes or Slurm client-job launcher")

    def task_resources(self, meta, site_name):
        validate_task_resources(meta, site_name)
        reject_job_gpu_reservation(meta, site_name)
        # Each native backend derives its request from the validated site/job
        # binding; do not reinterpret it as local process or Docker resources.
        return TaskResourceRequest()
