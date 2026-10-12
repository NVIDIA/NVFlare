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

"""Derive a task compute graph without importing or constructing application code."""

import copy
import math
from dataclasses import dataclass
from typing import Optional

from nvflare.apis.task_execution import ExecutionLifetime
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class_path_from_config
from nvflare.fuel.utils.component_builder import ConfigType

CLIENT_API_EXECUTOR_PATH = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
TASK_EXECUTOR_PATH = "nvflare.private.fed.client.task_worker_executor.TaskWorkerExecutor"


def _effective_settings(settings, site_name, default_key):
    if not isinstance(settings, dict):
        return {}
    result = copy.deepcopy(settings.get(default_key, {}))
    if not isinstance(result, dict):
        result = {}
    override = settings.get(site_name, {})
    if isinstance(override, dict):
        for name, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(name), dict):
                result[name] = _effective_settings({"default": result[name], "site": value}, "site", "default")
            else:
                result[name] = copy.deepcopy(value)
    return result


def _gpu_resource(value):
    def nonempty(item):
        if isinstance(item, dict):
            return any(nonempty(child) for child in item.values())
        return item is not None and item is not False and item != 0 and item != [] and item != ""

    if isinstance(value, dict):
        for name, item in value.items():
            if "gpu" in str(name).lower() and nonempty(item):
                return True
            if _gpu_resource(item):
                return True
    elif isinstance(value, list):
        return any(_gpu_resource(item) for item in value)
    return False


def validate_task_resources(meta, site_name):
    """Only inspect resources of the site receiving this task-lifetime app."""
    for key, default in (("resource_spec", "@default"), ("launcher_spec", "default")):
        if _gpu_resource(_effective_settings(meta.get(key, {}), site_name, default)):
            raise ValueError("task lifetime currently supports CPU Process execution; this site requests GPUs")


@dataclass(frozen=True)
class TaskExecutorConfig:
    executor: dict
    components: tuple[dict, ...]
    worker_timeout: Optional[float] = None
    result_wait_timeout: Optional[float] = None


@dataclass(frozen=True)
class TaskExecutionConfig:
    executors: tuple[TaskExecutorConfig, ...]
    job_components: tuple[dict, ...]


def _timeout(name, value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative or None")
    return value


def _dependencies(config, ids, scope=None, component=False):
    """Track declared IDs and reference arguments without changing their values.

    Argument names do not determine references. In particular, the whole args
    mapping is a container, even when a constructor argument is named 'path'.
    Ordinary dictionaries are preserved; this function never instantiates them.
    """
    found = set()
    if isinstance(config, dict):
        if not component and config.get("config_type", ConfigType.COMPONENT) == ConfigType.COMPONENT:
            try:
                scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)
                get_class_path_from_config(config, resolve_name=scanner.get_module_name)
                component = True
            except ConfigError:
                pass
        if component:
            if config.get("execution_scope", scope) != scope:
                raise ValueError("nested components must share their enclosing execution_scope")
            declared = config.get("component_dependencies", [])
            if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
                raise ValueError("component_dependencies must be a list of component IDs")
            if set(declared) - ids:
                raise ValueError(f"unknown component_dependencies: {sorted(set(declared) - ids)}")
            found.update(declared)
            args = config.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("component args must be a dict")
            children = args.values()
        else:
            children = config.values()
        for value in children:
            found.update(_dependencies(value, ids, scope))
    elif isinstance(config, (list, tuple)):
        for value in config:
            found.update(_dependencies(value, ids, scope))
    elif isinstance(config, str) and config in ids:
        found.add(config)
    return found


def prepare_task_execution(client_config):
    if not isinstance(client_config, dict):
        raise TypeError("client config must be a dict")
    lifetime = ExecutionLifetime.validate(client_config.get("execution_lifetime", ExecutionLifetime.JOB))
    for name in ("task_execution", "task_launcher"):
        if name in client_config:
            raise ValueError(f"{name} is site configuration; jobs cannot select it")
    for definition in client_config.get("executors", []):
        spec = definition.get("executor", {}) if isinstance(definition, dict) else {}
        if isinstance(spec, dict) and any(
            spec.get(key) == TASK_EXECUTOR_PATH for key in ("path", "class_path", "name")
        ):
            raise ValueError("TaskWorkerExecutor is framework-managed; select execution_lifetime='task'")
    if lifetime == ExecutionLifetime.JOB:
        return None
    if "task_state" in client_config:
        raise ValueError("declared task state is not supported by this task execution profile")
    executors = client_config.get("executors")
    components = client_config.get("components", [])
    if not isinstance(executors, list) or not executors:
        raise ValueError("task execution requires client executors")
    if not isinstance(components, list):
        raise ValueError("client components must be a list")
    scopes = {}
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("id"), str) or not component["id"]:
            raise ValueError("client components must have nonempty string IDs")
        if component["id"] in scopes:
            raise ValueError(f"duplicate component ID {component['id']!r}")
        scope = component.get("execution_scope", ExecutionLifetime.TASK)
        if scope not in (ExecutionLifetime.JOB, ExecutionLifetime.TASK):
            raise ValueError("component execution_scope must be 'job' or 'task'")
        scopes[component["id"]] = scope

    def validate_dependencies(config, scope, component=False):
        conflicts = {name for name in _dependencies(config, set(scopes), scope, component) if scopes[name] != scope}
        if conflicts:
            raise ValueError(f"{scope}-owned component depends on another execution scope: {sorted(conflicts)}")

    for component in components:
        validate_dependencies(component, scopes[component["id"]], component=True)
    for name in ("task_data_filters", "task_result_filters"):
        validate_dependencies(client_config.get(name, []), ExecutionLifetime.JOB)
    worker_components = tuple(copy.deepcopy(c) for c in components if scopes[c["id"]] == ExecutionLifetime.TASK)
    job_components = tuple(copy.deepcopy(c) for c in components if scopes[c["id"]] == ExecutionLifetime.JOB)
    scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)
    plans = []
    for definition in executors:
        if not isinstance(definition, dict) or not isinstance(definition.get("executor"), dict):
            raise ValueError("each executor must have a component specification")
        config = definition["executor"]
        path = get_class_path_from_config(config, resolve_name=scanner.get_module_name)
        if path == TASK_EXECUTOR_PATH:
            raise ValueError("TaskWorkerExecutor is framework-managed; select execution_lifetime='task'")
        if path == "nvflare.collab.runtime.executor.CollabExecutor":
            raise ValueError("CollabExecutor requires job lifetime for its resident federation RPC service")
        args = config.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("executor args must be a dict")
        result_timeout = None
        if path == CLIENT_API_EXECUTOR_PATH:
            if args.get("execution_mode") != "in_process":
                raise ValueError(
                    "task lifetime requires in_process Client API; external_process/attach use job lifetime"
                )
            result_timeout = _timeout("result_wait_timeout", args.get("result_wait_timeout"))
        validate_dependencies(config, "task", component=True)
        plans.append(
            TaskExecutorConfig(
                copy.deepcopy(config),
                worker_components,
                _timeout("worker_timeout", definition.get("worker_timeout")),
                result_timeout,
            )
        )
    return TaskExecutionConfig(tuple(plans), job_components)
