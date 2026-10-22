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
"""Trusted Slurm profile matching and task-resource resolution."""

from dataclasses import replace

from nvflare.apis.task_launcher_spec import TaskResourceRequest
from nvflare.app_opt.job_launcher.slurm.config import _require_int
from nvflare.app_opt.job_launcher.slurm.launcher import _resolve_resources
from nvflare.utils.job_launcher_utils import (
    get_job_launcher_spec,
    get_portable_resource_spec,
    get_resource_manager_spec,
    validate_portable_resource_conflicts,
)

TASK_LAUNCHER_PATH = "nvflare.app_opt.job_launcher.slurm.task_launcher.SlurmTaskLauncher"
CJ_LAUNCHER_PATH = "nvflare.app_opt.job_launcher.slurm.task_job_launcher.ClientSlurmTaskJobLauncher"


def component_args(config):
    if not isinstance(config, dict):
        raise ValueError("Slurm site component must be a dictionary")
    args = config.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Slurm site component args must be a dictionary")
    args = args.get("class_attributes", args)
    if not isinstance(args, dict):
        raise ValueError("Slurm site class_attributes must be a dictionary")
    return args


def component_path(config):
    if not isinstance(config, dict):
        raise ValueError("Slurm site component must be a dictionary")
    return config.get("path", config.get("class_path"))


def validate_slurm_site(launcher, site_config):
    from nvflare.app_opt.job_launcher.slurm.task_job_launcher import ClientSlurmTaskJobLauncher

    if not isinstance(site_config, dict) or not isinstance(site_config.get("components", []), list):
        raise ValueError("Slurm canonical site configuration must contain a component list")
    components = site_config.get("components", [])
    matches = [item for item in components if component_path(item) == CJ_LAUNCHER_PATH]
    if len(matches) != 1:
        raise ValueError("Slurm task profile requires exactly one ClientSlurmTaskJobLauncher")
    if any(
        component_path(item) != CJ_LAUNCHER_PATH and str(component_path(item)).endswith("JobLauncher")
        for item in components
    ):
        raise ValueError("Slurm task profile cannot combine different CP/CJ job launchers")
    cj = ClientSlurmTaskJobLauncher(**component_args(matches[0]))
    if cj.config.workspace_path != launcher.manager.config.workspace_path or cj.config.sandbox != "none":
        raise ValueError("Slurm CP/CJ/task profile requires the same bare shared workspace")
    if cj.config.image:
        raise ValueError("Slurm task profile does not support a CJ container image")


def task_resources(launcher, meta, site_name):
    """Resolve native/default resources without retaining whole-job GPU tokens."""
    validate_portable_resource_conflicts(meta)
    reservation = get_resource_manager_spec(meta, site_name)
    if reservation.get("num_of_gpus", 0):
        raise ValueError(
            "Slurm task GPUs must use native launcher_spec gpus_per_node; "
            "portable/legacy GPU metadata would retain a whole-job CP reservation"
        )
    portable = get_portable_resource_spec(meta, site_name)
    native = get_job_launcher_spec(meta, site_name, "slurm")
    if native.get("nodes", 1) != 1 or native.get("additional_node_command"):
        raise ValueError("Slurm task profile supports one node without additional_node_command")
    if native.get("image"):
        raise ValueError("Slurm task profile does not support container images")
    default = launcher.resources
    gpu_count = native.get("gpus_per_node", portable.get("num_of_gpus", default.gpu_count))
    _require_int(gpu_count, "gpus_per_node", 0)
    spec = {"cpus_per_node": int(default.cpu_cores or 1), **native}
    if default.memory_mb is not None:
        spec.setdefault("mem_per_node", default.memory_mb)
    spec["gpus_per_node"] = gpu_count
    resolved = _resolve_resources(meta, site_name, "none", launcher.manager.config.pending_timeout, spec=spec)
    return replace(resolved, gpus_per_node=gpu_count)


def resource_request(resources):
    return TaskResourceRequest(resources.cpus_per_node, resources.mem_per_node, resources.gpus_per_node or 0)
