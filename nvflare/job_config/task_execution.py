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

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class_path_from_config
from nvflare.fuel.utils.component_builder import ConfigType
from nvflare.job_config.defs import ExecutionLifetime

CLIENT_API_EXECUTOR_PATH = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
TASK_EXECUTOR_PATH = "nvflare.private.fed.client.task_worker_executor.TaskWorkerExecutor"
SUPPORTED_TASK_EXECUTORS = {CLIENT_API_EXECUTOR_PATH, "nvflare.app_common.np.np_trainer.NPTrainer"}
DECOMPOSER_REGISTER_PATH = "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister"


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
            if name == "mem_per_gpu_in_GiB" and value.get("num_of_gpus") == 0:
                continue
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


def _dependencies(config, ids, component=False):
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
            if "execution_scope" in config:
                raise ValueError("component execution_scope is not supported; compute dependencies define placement")
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
            found.update(_dependencies(value, ids))
    elif isinstance(config, (list, tuple)):
        for value in config:
            found.update(_dependencies(value, ids))
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
    if "format_version" in client_config and client_config["format_version"] != 3:
        raise ValueError("task lifetime requires client format_version 3 so older clients reject it")
    if "task_state" in client_config:
        raise ValueError("declared task state is not supported by this task execution profile")
    executors = client_config.get("executors")
    components = client_config.get("components", [])
    if not isinstance(executors, list) or not executors:
        raise ValueError("task execution requires client executors")
    if not isinstance(components, list):
        raise ValueError("client components must be a list")
    by_id = {}
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("id"), str) or not component["id"]:
            raise ValueError("client components must have nonempty string IDs")
        if component["id"] in by_id:
            raise ValueError(f"duplicate component ID {component['id']!r}")
        by_id[component["id"]] = component
    ids = set(by_id)
    references = {cid: _dependencies(spec, ids, component=True) for cid, spec in by_id.items()}

    def closure(roots):
        selected = set()
        pending = set(roots)
        while pending:
            cid = pending.pop()
            if cid not in selected:
                selected.add(cid)
                pending.update(references[cid] - selected)
        return selected

    scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)
    serialization_ids = {
        cid
        for cid, spec in by_id.items()
        if get_class_path_from_config(spec, resolve_name=scanner.get_module_name) == DECOMPOSER_REGISTER_PATH
    }
    plans = []
    compute_ids = set()
    for definition in executors:
        if not isinstance(definition, dict) or not isinstance(definition.get("executor"), dict):
            raise ValueError("each executor must have a component specification")
        config = definition["executor"]
        path = get_class_path_from_config(config, resolve_name=scanner.get_module_name)
        if path == TASK_EXECUTOR_PATH:
            raise ValueError("TaskWorkerExecutor is framework-managed; select execution_lifetime='task'")
        if path not in SUPPORTED_TASK_EXECUTORS:
            raise ValueError(f"executor {path!r} is not supported by this task profile; use job lifetime")
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
        worker_ids = closure(_dependencies(config, ids, component=True))
        compute_ids.update(worker_ids)
        worker_components = tuple(copy.deepcopy(c) for c in components if c["id"] in worker_ids | serialization_ids)
        plans.append(
            TaskExecutorConfig(
                copy.deepcopy(config),
                worker_components,
                _timeout("worker_timeout", definition.get("worker_timeout")),
                result_timeout,
            )
        )
    # Existing filters and components outside the selected compute graph stay
    # with the CJ. Site filters are added by the normal privacy pipeline there.
    resident_roots = (ids - compute_ids) | serialization_ids
    for name in ("task_data_filters", "task_result_filters"):
        for definition in client_config.get(name, []):
            resident_roots.update(_dependencies(definition.get("filters", []), ids))
    shared = (closure(resident_roots) & compute_ids) - serialization_ids
    if shared:
        raise ValueError(f"components shared by compute and resident CJ services are not supported: {sorted(shared)}")
    job_components = tuple(
        copy.deepcopy(c) for c in components if c["id"] not in compute_ids or c["id"] in serialization_ids
    )
    return TaskExecutionConfig(tuple(plans), job_components)
