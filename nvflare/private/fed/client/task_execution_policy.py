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

"""Site-owned policy for the initial CPU Process task execution profile."""

import os
import re
from dataclasses import dataclass

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.task_launcher_spec import TaskLauncherSpec
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR
from nvflare.fuel.utils.class_utils import ModuleScanner
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import _timeout
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
        return cls.from_config(data.get("task_execution", {}))

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
        if not isinstance(launcher, TaskLauncherSpec) or launcher.launch_mode != "process":
            raise ValueError("this task execution profile requires a Process TaskLauncherSpec")
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
        allowed = PASSTHROUGH_ENVIRONMENT_NAMES | set(self.environment_variables)
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in allowed and name not in FORBIDDEN_ENVIRONMENT_NAMES
        }
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "ROCR_VISIBLE_DEVICES": "",
            }
        )
        return environment
