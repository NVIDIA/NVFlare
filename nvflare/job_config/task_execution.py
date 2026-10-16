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

"""Validate the CPU task profile without importing or constructing application code."""

import copy
import math
from dataclasses import dataclass
from typing import Optional

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class_path_from_config
from nvflare.fuel.utils.component_builder import ConfigType
from nvflare.job_config.defs import ExecutionLifetime
from nvflare.utils.job_launcher_utils import get_job_launcher_spec, get_resource_manager_spec

CLIENT_API_EXECUTOR_PATH = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
TASK_EXECUTOR_PATH = "nvflare.private.fed.client.task_worker_executor.TaskWorkerExecutor"
STATISTICS_EXECUTOR_PATH = "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor"
SUPPORTED_TASK_EXECUTORS = {
    CLIENT_API_EXECUTOR_PATH,
    STATISTICS_EXECUTOR_PATH,
    "nvflare.app_common.np.np_trainer.NPTrainer",
}
DECOMPOSER_REGISTER_PATH = "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister"


def validate_task_resources(meta, site_name):
    """Use the same receiving-site resource resolution as job launchers."""
    specs = [get_resource_manager_spec(meta, site_name)]
    specs.extend(get_job_launcher_spec(meta, site_name, mode) for mode in ("process", "docker", "k8s", "slurm"))
    for spec in specs:
        if spec.get("num_of_gpus", 0) or spec.get("gpus_per_node", 0):
            raise ValueError("task lifetime currently supports CPU Process execution; this site requests GPUs")


def validate_task_workflows(workflows):
    """Run optional workflow-specific checks for task-lifetime export."""
    for workflow in workflows:
        validate = getattr(workflow.controller, "validate_task_lifetime", None)
        if validate is not None:
            validate()


@dataclass(frozen=True)
class TaskExecutorConfig:
    executor: dict
    components: tuple[dict, ...]
    worker_timeout: Optional[float] = None
    result_wait_timeout: Optional[float] = None


@dataclass(frozen=True)
class TaskExecutionConfig:
    executors: tuple[TaskExecutorConfig, ...]
    compute_component_ids: frozenset[str] = frozenset()


def _timeout(name, value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative or None")
    return value


def task_constructor_args(config):
    """Task workers construct an executor directly, without component graphs."""
    for key in ("execution_scope", "component_dependencies"):
        if key in config:
            raise ValueError(f"{key} is not supported by this task profile; use job lifetime")
    args = config.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("component args must be a dict")
    # Match ComponentBuilder's existing constructor-argument wrapper without
    # restoring nested component construction in task workers.
    if "class_attributes" in args:
        args = args["class_attributes"]
        if not isinstance(args, dict):
            raise ValueError("component class_attributes must be a dict")
    scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)

    def check(value):
        if isinstance(value, dict):
            if value.get("config_type") == ConfigType.DICT:
                return
            try:
                get_class_path_from_config(value, resolve_name=scanner.get_module_name)
            except ConfigError:
                pass
            else:
                raise ValueError("nested components are not supported by task workers; use job lifetime")
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)

    for value in args.values():
        check(value)
    return args


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
    scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)
    serialization_ids = {
        cid
        for cid, spec in by_id.items()
        if get_class_path_from_config(spec, resolve_name=scanner.get_module_name) == DECOMPOSER_REGISTER_PATH
    }
    serialization_components = tuple(copy.deepcopy(c) for c in components if c["id"] in serialization_ids)
    for component in serialization_components:
        task_constructor_args(component)
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
        args = task_constructor_args(config)
        result_timeout = None
        worker_components = serialization_components
        if path == CLIENT_API_EXECUTOR_PATH:
            if args.get("execution_mode") != "in_process":
                raise ValueError(
                    "task lifetime requires in_process Client API; external_process/attach use job lifetime"
                )
            result_timeout = _timeout("result_wait_timeout", args.get("result_wait_timeout"))
        elif path == STATISTICS_EXECUTOR_PATH:
            generator_id = args.get("generator_id")
            if not isinstance(generator_id, str) or generator_id not in by_id or generator_id in serialization_ids:
                raise ValueError("StatisticsExecutor requires a configured statistics generator_id")
            generator = by_id[generator_id]
            task_constructor_args(generator)
            compute_ids.add(generator_id)
            worker_components = (*serialization_components, copy.deepcopy(generator))
        plans.append(
            TaskExecutorConfig(
                copy.deepcopy(config),
                worker_components,
                _timeout("worker_timeout", definition.get("worker_timeout")),
                result_timeout,
            )
        )
    return TaskExecutionConfig(tuple(plans), frozenset(compute_ids))
