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

"""Validation and placement planning for task-lifetime execution."""

import copy
import math
from dataclasses import dataclass
from typing import Optional

from nvflare.apis.task_execution import ExecutionLifetime

EXECUTION_LIFETIME_KEY = "execution_lifetime"
TASK_EXECUTOR_PATH = "nvflare.private.fed.client.task_worker_executor.TaskWorkerExecutor"
CLIENT_API_EXECUTOR_PATH = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
TASK_LAUNCHER_KEY = "task_launcher"


@dataclass(frozen=True)
class TaskExecutorConfig:
    """Inert application configuration assigned to one runtime-created supervisor."""

    executor: dict
    components: tuple[dict, ...]
    worker_timeout: Optional[float] = None


@dataclass(frozen=True)
class TaskExecutionConfig:
    """Runtime placement plan derived from an unmodified submitted job config."""

    executors: tuple[TaskExecutorConfig, ...]
    job_components: tuple[dict, ...]


def _collect_strings(value, result):
    if isinstance(value, str):
        result.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, result)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, result)


def _referenced_component_ids(config, component_ids):
    values = set()
    _collect_strings(config, values)
    return values.intersection(component_ids)


def _component_dependency_closure(config, component_by_id):
    component_ids = set(component_by_id)
    pending = list(_referenced_component_ids(config, component_ids))
    result = set()
    while pending:
        component_id = pending.pop()
        if component_id in result:
            continue
        result.add(component_id)
        pending.extend(_referenced_component_ids(component_by_id[component_id], component_ids) - result)
    return result


def prepare_task_execution(client_config: dict) -> Optional[TaskExecutionConfig]:
    """Validate task mode and derive CJ/worker placement without rewriting the job.

    The returned application specs remain inert. The trusted client runtime
    authorizes them and constructs :class:`TaskWorkerExecutor` directly, so the
    framework supervisor is never a job-selected component or allow-list entry.
    """
    if not isinstance(client_config, dict):
        raise TypeError("client_config must be a dict")

    lifetime = client_config.get(EXECUTION_LIFETIME_KEY, ExecutionLifetime.JOB)
    ExecutionLifetime.validate(lifetime)
    for setting in (TASK_LAUNCHER_KEY, "task_execution"):
        if setting in client_config:
            raise ValueError(
                f"{setting!r} is site/runtime configuration and cannot be selected by a job; "
                "configure it in the site's resources.json"
            )
    if lifetime == ExecutionLifetime.JOB:
        return None

    executors = client_config.get("executors")
    if not isinstance(executors, list) or not executors:
        raise ValueError("task execution requires at least one client executor")

    components = client_config.get("components", [])
    if not isinstance(components, list):
        raise ValueError("client components must be a list")
    component_by_id = {}
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("id"), str):
            raise ValueError("each client component must have a string id")
        if component["id"] in component_by_id:
            raise ValueError(f"duplicate client component id {component['id']!r}")
        component_by_id[component["id"]] = component

    filter_configs = {
        "task_data_filters": client_config.get("task_data_filters", []),
        "task_result_filters": client_config.get("task_result_filters", []),
    }
    filter_component_ids = _component_dependency_closure(filter_configs, component_by_id)
    worker_component_ids = set()
    executor_worker_component_ids = []
    executor_worker_timeouts = []
    for executor_def in executors:
        if not isinstance(executor_def, dict) or not isinstance(executor_def.get("executor"), dict):
            raise ValueError("each client executor entry must contain an executor component config")
        executor_config = executor_def["executor"]
        configured_path = executor_config.get("path", executor_config.get("class_path", executor_config.get("name")))
        worker_timeout = None
        if configured_path == TASK_EXECUTOR_PATH:
            raise ValueError(
                "TaskWorkerExecutor is framework-managed and cannot be selected by a job; "
                "set execution_lifetime='task' on the original application Executor"
            )
        if configured_path == CLIENT_API_EXECUTOR_PATH:
            execution_mode = executor_config.get("args", {}).get("execution_mode")
            if execution_mode != "in_process":
                raise ValueError(
                    "execution_lifetime='task' currently supports ClientAPIExecutor in_process scripts only; "
                    "external_process and attach require job-based execution"
                )
            worker_timeout = executor_config.get("args", {}).get("result_wait_timeout")
            if worker_timeout is not None and (
                isinstance(worker_timeout, bool)
                or not isinstance(worker_timeout, (int, float))
                or not math.isfinite(worker_timeout)
                or worker_timeout < 0
            ):
                raise ValueError("Client API task result_wait_timeout must be finite and >= 0 or None")
        executor_ids = _component_dependency_closure(executor_config, component_by_id)
        worker_component_ids.update(executor_ids)
        executor_worker_component_ids.append(executor_ids)
        executor_worker_timeouts.append(worker_timeout)

    shared_ids = worker_component_ids.intersection(filter_component_ids)
    if shared_ids:
        raise ValueError(
            "task execution cannot place components referenced by both an Executor and CJ filters: "
            f"{sorted(shared_ids)}"
        )

    job_components = tuple(copy.deepcopy(c) for c in components if c["id"] not in worker_component_ids)
    task_executors = []
    for executor_def, executor_component_ids, worker_timeout in zip(
        executors, executor_worker_component_ids, executor_worker_timeouts
    ):
        executor_config = executor_def["executor"]
        worker_components = tuple(copy.deepcopy(c) for c in components if c["id"] in executor_component_ids)
        task_executors.append(
            TaskExecutorConfig(
                executor=copy.deepcopy(executor_config),
                components=worker_components,
                worker_timeout=worker_timeout,
            )
        )

    return TaskExecutionConfig(executors=tuple(task_executors), job_components=job_components)
