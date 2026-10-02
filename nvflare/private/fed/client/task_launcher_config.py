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

"""Build and inject the site-owned task launcher into client-job supervisors."""

import copy
import json
from typing import Optional

from nvflare.apis.launcher import LauncherMode
from nvflare.apis.task_execution import TaskArtifactCleanup
from nvflare.apis.task_launcher_spec import TaskLauncherSpec
from nvflare.fuel.utils.class_loader import load_class
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor

TASK_LAUNCHER_CONFIG = "task_launcher"
TASK_LAUNCHER_PATH = "path"
TASK_LAUNCHER_ARGS = "args"
DEFAULT_TASK_LAUNCHER_PATH = "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher"


def _site_resources(workspace):
    if workspace is None:
        return {}
    path = workspace.get_resources_file_path()
    if path is None:
        return {}
    # Do not use ConfigService/ConfigFactory: their recursive filename search
    # can select a job's resources.json before the site's resources.json.default.
    with open(path, encoding="utf-8") as stream:
        resources = json.load(stream)
    if not isinstance(resources, dict):
        raise TypeError("the site resources configuration must be a dict")
    return resources


def _site_artifact_cleanup(resources):
    config = resources.get("task_execution", {})
    if not isinstance(config, dict):
        raise TypeError("task_execution must be a dict")
    unknown = set(config).difference({"artifact_cleanup"})
    if unknown:
        raise ValueError(f"task_execution contains unsupported settings: {sorted(unknown)}")
    return TaskArtifactCleanup.validate(config.get("artifact_cleanup", TaskArtifactCleanup.JOB))


def build_task_launcher(config: Optional[dict] = None, workspace=None) -> TaskLauncherSpec:
    """Build the launcher selected by trusted site/runtime configuration.

    If ``config`` is omitted, ``task_launcher`` is read only from the site's
    resources configuration. A missing setting selects the local Process
    backend, which keeps simulator and basic local deployments working.
    """
    if config is None:
        config = _site_resources(workspace).get(TASK_LAUNCHER_CONFIG, {})

    if not isinstance(config, dict):
        raise TypeError(f"{TASK_LAUNCHER_CONFIG} must be a dict but got {type(config)}")
    unknown = set(config).difference({TASK_LAUNCHER_PATH, TASK_LAUNCHER_ARGS, "environment_variables"})
    if unknown:
        raise ValueError(f"{TASK_LAUNCHER_CONFIG} contains unsupported settings: {sorted(unknown)}")
    TaskWorkerExecutor.validate_environment_variables(config.get("environment_variables", []))

    class_path = config.get(TASK_LAUNCHER_PATH, DEFAULT_TASK_LAUNCHER_PATH)
    if not isinstance(class_path, str) or not class_path.strip():
        raise ValueError(f"{TASK_LAUNCHER_CONFIG}.{TASK_LAUNCHER_PATH} must be a non-empty string")
    args = config.get(TASK_LAUNCHER_ARGS, {})
    if not isinstance(args, dict):
        raise TypeError(f"{TASK_LAUNCHER_CONFIG}.{TASK_LAUNCHER_ARGS} must be a dict")

    launcher_type = load_class(class_path)
    if not isinstance(launcher_type, type) or not issubclass(launcher_type, TaskLauncherSpec):
        raise TypeError(f"configured task launcher {class_path!r} must implement TaskLauncherSpec")
    return launcher_type(**copy.deepcopy(args))


def configure_task_launchers(runner_config, job_launcher_mode=None, workspace=None) -> Optional[TaskLauncherSpec]:
    """Inject a matching site-owned launcher into all framework task supervisors."""
    task_router = getattr(runner_config, "task_router", None)
    task_table = getattr(task_router, "task_table", {})
    supervisors = []
    seen = set()
    for executor in task_table.values():
        # Only the exact framework adapter receives privileged runtime
        # injection. A job-supplied lookalike or subclass cannot opt in by
        # exposing a similarly named method.
        if type(executor) is TaskWorkerExecutor and id(executor) not in seen:
            supervisors.append(executor)
            seen.add(id(executor))
    if not supervisors:
        return None

    resources = _site_resources(workspace)
    artifact_cleanup = _site_artifact_cleanup(resources)
    launcher_config = resources.get(TASK_LAUNCHER_CONFIG, {})
    launcher = build_task_launcher(launcher_config)
    job_mode = LauncherMode.validate(job_launcher_mode, "selected JobLauncher mode")
    task_mode = LauncherMode.validate(launcher.launch_mode, "configured TaskLauncher mode")
    if job_mode != task_mode:
        raise RuntimeError(
            f"JobLauncher mode {job_mode!r} does not match configured TaskLauncher mode {task_mode!r}; "
            "CP, CJ, and task workers must use the same launch mode"
        )
    for supervisor in supervisors:
        supervisor.set_task_launcher(
            launcher,
            environment_variables=launcher_config.get("environment_variables", []),
            artifact_cleanup=artifact_cleanup,
        )

    handlers = getattr(runner_config, "handlers", None)
    if not isinstance(handlers, list):
        raise TypeError("runner_config.handlers must be a list")
    if not any(handler is launcher for handler in handlers):
        handlers.append(launcher)
    return launcher
