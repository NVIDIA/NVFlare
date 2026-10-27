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

"""Prepare public fixtures for a serial resident/task Colossus campaign.

This module never contacts Docker, copies startup kits, or starts/stops jobs.
The lease coordinator owns quiescence, supported parent shutdown, local policy
application and restart, submission, and evidence collection.
"""

import argparse
import copy
import json
from pathlib import Path

from nvflare.app_opt.task_launcher.docker_gpu import validate_partition
from nvflare.app_opt.task_launcher.docker_launcher import DockerTaskLauncher
from nvflare.app_opt.task_launcher.docker_service import validate_worker_config
from tests.integration_test.data.docker_task_execution.export_probe import export_probe

CP_PATH = "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher"


def phase_policies(worker_image, site_devices, *, partition_id="qualification-serial-pool"):
    if len(site_devices) != 2 or any(
        not isinstance(site, str) or not site or site == "server" for site in site_devices
    ):
        raise ValueError("the qualification matrix requires two named client sites")
    partition = validate_partition({"id": partition_id, "devices": list(site_devices.values()), "exclusive": True})
    policies = {"resident": {}, "task": {}}
    for site, device in site_devices.items():
        owned = validate_partition({"id": partition_id, "devices": [device], "exclusive": True})["devices"][0]
        for phase in policies:
            worker = {
                "image": worker_image,
                "python_path": "/usr/local/bin/python",
                "environment": {"OMP_NUM_THREADS": "1"},
            }
            if phase == "task":
                worker["gpu_partition"] = {
                    "id": partition["id"],
                    "devices": list(partition["devices"]),
                    "exclusive": True,
                }
            validate_worker_config(worker)
            policies[phase][site] = {
                "task_worker_config": worker,
                "default_job_env": (
                    {"NVIDIA_VISIBLE_DEVICES": owned, "CUDA_VISIBLE_DEVICES": "0"} if phase == "resident" else {}
                ),
                "device_requests": (
                    [{"Driver": "nvidia", "DeviceIDs": [owned], "Capabilities": [["gpu"]]}]
                    if phase == "resident"
                    else []
                ),
            }
    return policies


def apply_site_policy(site_config, policy):
    """Return a canonical config, preserving unrelated and security settings.

    Call only after the coordinator confirms there are no active jobs, task
    containers or reservation records and has stopped the owned parent.
    Removing a partition from a config never releases an existing reservation.
    """
    result = copy.deepcopy(site_config)
    launchers = [item for item in result.get("components", []) if item.get("path") == CP_PATH]
    if len(launchers) != 1:
        raise ValueError("expected exactly one canonical Docker client JobLauncher")
    args = launchers[0].setdefault("args", {})
    defaults = args.setdefault("default_job_container_kwargs", {})
    defaults.pop("device_requests", None)
    if policy["device_requests"]:
        defaults["device_requests"] = copy.deepcopy(policy["device_requests"])
    environment = args.setdefault("default_job_env", {})
    for name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        environment.pop(name, None)
    environment.update(policy["default_job_env"])
    worker = copy.deepcopy(args.get("task_worker_config", {}))
    worker_environment = {**worker.get("environment", {}), **policy["task_worker_config"].get("environment", {})}
    worker.pop("gpu_partition", None)
    worker.update(copy.deepcopy(policy["task_worker_config"]))
    worker["environment"] = worker_environment
    validate_worker_config(worker)
    args["task_worker_config"] = worker
    execution = result.setdefault("task_execution", {})
    execution["launcher"] = {"path": "nvflare.app_opt.task_launcher.docker_launcher.DockerTaskLauncher"}
    execution["artifact_cleanup"] = "retain"
    DockerTaskLauncher().validate_site_config(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--site-gpu", action="append", required=True, metavar="SITE=GPU-UUID")
    parser.add_argument("--partition-id", default="qualification-serial-pool")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sites = {}
    for value in args.site_gpu:
        site, separator, device = value.partition("=")
        if not separator or site in sites:
            parser.error("each site-gpu must name a distinct SITE=GPU-UUID")
        sites[site] = device
    policies = phase_policies(args.image, sites, partition_id=args.partition_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase-policies.json").write_text(json.dumps(policies, indent=2) + "\n")
    for lifetime in ("job", "task"):
        for device in ("cpu", "cuda"):
            export_probe(args.image, device, 1 if device == "cuda" else 0, 2, lifetime, str(output / "jobs"))
    (output / "campaign.json").write_text(
        json.dumps(
            {
                "phase_order": ["resident", "task"],
                "rows": [
                    {"lifetime": lifetime, "device": device, "state": "requested", "job_id": None}
                    for lifetime in ("job", "task")
                    for device in ("cpu", "cuda")
                ],
                "switch_requires": [
                    "coordinator rechecks existing lease ownership and unrelated workloads",
                    "all previous phase jobs are terminal; owned workers and reservations are absent",
                    "stop owned parents before applying canonical site resources; restart and verify live imports/config",
                ],
                "credential_boundary": "worker job bind still exposes job_cert/job.key; no credential-free claim",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared four jobs and public phase policies at {output}; no jobs launched")


if __name__ == "__main__":
    main()
