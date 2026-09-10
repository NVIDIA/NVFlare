#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Emit a minimal admin contract from hash-bound stage-09 evidence, not a host claim."""

import hashlib
import json
import re
import sys
from pathlib import Path

directory, profile_id, version, runtime, image, profile_hash, actual_hash = sys.argv[1:]
root = Path(directory)


def checked_json(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("Stage 09 evidence binding missing; re-finalize before exporting an admin profile")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("Approved evidence changed: " + str(path))
    return json.loads(raw)


p = checked_json(root / "approved-launch-profile.json", profile_hash)
a = checked_json(root / "rehearsal-collector-build/actual-launch.json", actual_hash)
if (
    p["runtime_class"] != runtime
    or runtime != "kata-qemu-nvidia-gpu-snp"
    or version != "3.29.0"
    or not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image)
):
    raise SystemExit("Unsupported runtime profile")
expected_resources = {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}
resources = {kind: {k: str(v) for k, v in values.items()} for kind, values in p["pod_resources"].items()}
if resources != expected_resources or a["pod_resources"] != [p["pod_resources"]]:
    raise SystemExit("Only the rehearsed single-GPU, omitted-CPU/memory profile is supported")
if a["artifacts"]["kata_config"]["sha256"] != p["kata_config_sha256"]:
    raise SystemExit("Approved configuration differs from captured launch")
inputs_hash = hashlib.sha256(json.dumps(a["launch_inputs"], sort_keys=True).encode()).hexdigest()
if inputs_hash != a["launch_inputs_sha256"]:
    raise SystemExit("Launch fingerprint is inconsistent")
vcpus, memory = p["runtime_default_vcpus"], p["runtime_default_memory_mib"]
if type(vcpus) is not int or vcpus <= 0 or type(memory) is not int or memory <= 0:
    raise SystemExit("Invalid runtime defaults")
if a["launch_inputs"]["smp"].split(",")[0] != str(vcpus) or a["launch_inputs"]["memory"] != f"{memory}M":
    raise SystemExit("Actual VM sizing differs from defaults; review the launch profile")
print(
    json.dumps(
        {
            "schema": "coco-approved-workload-launch/v1",
            "profile_id": profile_id,
            "runtime_class": runtime,
            "kata_version": version,
            "kata_deploy_image": image,
            "kata_config_sha256": p["kata_config_sha256"],
            "launch_inputs_sha256": inputs_hash,
            "vm_defaults": {"vcpus": vcpus, "memory_mib": memory},
            "pod_constraints": {
                "container_count": 1,
                "gpu_resource": "nvidia.com/pgpu",
                "gpu_count": 1,
                "cpu_memory_resources": "omitted",
                "host_namespaces": False,
                "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
            },
        },
        indent=2,
    )
)
