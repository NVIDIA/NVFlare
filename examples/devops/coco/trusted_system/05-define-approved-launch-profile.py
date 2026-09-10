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

"""Define a trusted single-container GPU rehearsal profile without guessing boot defaults."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

workload_path, config_path, output_path = map(Path, sys.argv[1:])
os.umask(0o077)
workload = yaml.safe_load(workload_path.read_text())
spec = workload["spec"]
if spec.get("runtimeClassName") != "kata-qemu-nvidia-gpu-snp":
    raise SystemExit("Expected the NVIDIA GPU SNP RuntimeClass")
if len(spec["containers"]) != 1 or spec.get("initContainers"):
    raise SystemExit("Only a single-container workload profile is supported")
if any(spec.get(key, False) for key in ["hostNetwork", "hostPID", "hostIPC"]):
    raise SystemExit("Host namespaces are not approved")
resources = spec["containers"][0].get("resources", {})
for name, value in resources.get("limits", {}).items():
    resources.setdefault("requests", {}).setdefault(name, value)
if str(resources.get("limits", {}).get("nvidia.com/pgpu", 0)) != "1":
    raise SystemExit("This trusted host profile requires exactly one passthrough GPU")
with config_path.open("rb") as stream:
    config = tomllib.load(stream)
qemu = config["hypervisor"]["qemu"]
for name in ["default_vcpus", "default_memory"]:
    if type(qemu.get(name)) is not int or qemu[name] <= 0:
        raise SystemExit(f"Explicit positive runtime {name} is required")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


artifacts = {}
for name in ["path", "firmware", "kernel", "initrd", "image"]:
    if qemu.get(name):
        path = Path(qemu[name]).resolve(strict=True)
        artifacts[name] = {"path": str(path), "sha256": digest(path)}
cpu = json.loads(subprocess.check_output(["lscpu", "-J"]))
data = {
    "schema": 1,
    "scope": "Trusted sec_sys single-container GPU launch; not a cross-machine portability claim",
    "host": platform.node(),
    "host_kernel": platform.release(),
    "cpu": cpu,
    "runtime_class": spec["runtimeClassName"],
    "workload_source": str(workload_path),
    "workload_yaml_sha256": digest(workload_path),
    "pod_resources": resources,
    "cpu_request_omitted": "cpu" not in resources.get("requests", {}),
    "memory_request_omitted": "memory" not in resources.get("requests", {}),
    "runtime_default_vcpus": qemu["default_vcpus"],
    "runtime_default_memory_mib": qemu["default_memory"],
    "kata_config_path": str(config_path),
    "kata_config_sha256": digest(config_path),
    "kata_config": config,
    "artifacts": artifacts,
    "actual_launch_capture_required": True,
    "fresh_report_signature_and_nonce_verification_required": True,
    "baseline_assumption": "User trusts sec_sys firmware baseline; signed reported TCB establishes minimums",
}
with output_path.open("x") as stream:
    json.dump(data, stream, indent=2)
    stream.write("\n")
print("Defined profile:", output_path)
print("Runtime defaults:", qemu["default_vcpus"], "vCPUs;", qemu["default_memory"], "MiB; passthrough GPUs: 1")
print("Workload resource fields are preserved; actual QEMU launch must be captured and compared.")
