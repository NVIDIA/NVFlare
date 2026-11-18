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


"""Shared factories for CoCo unit and integration contracts."""

import base64
import gzip
import json
from pathlib import Path

from nvflare.lighter.cc_provision import workload_security

API = vars(workload_security)

IMAGE = "registry.example.invalid/app@sha256:" + "a" * 64


def context(readonly=False):
    return {
        "privileged": False,
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "runAsUser": 65532,
        "runAsGroup": 65532,
        "readOnlyRootFilesystem": readonly,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def pod(readonly=False, image=IMAGE):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "spec": {
            "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
            "containers": [
                {
                    "name": "app",
                    "image": image,
                    "securityContext": context(readonly),
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                }
            ],
        },
    }


def contract(readonly=False):
    return {
        "schema": API["SCHEMA"],
        "profile_id": "reviewed",
        "runtime_class": "kata-qemu-nvidia-gpu-snp",
        "kata_version": "3.29.0",
        "guest_token_api": "guest-local-aa-token/v1",
        "kata_deploy_image": "registry.example.invalid/runtime@sha256:" + "b" * 64,
        "kata_config_sha256": "c" * 64,
        "launch_inputs_sha256": "d" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "workload_security_context": context(readonly),
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu",
            "gpu_count": 1,
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }


def policy_data(readonly=False, image=IMAGE):
    pause = json.loads(
        (
            (Path(__file__).resolve().parents[2] / "tests/unit_test/lighter/cc_provision/impl/fixtures")
            / "kata-3.29-pause-policy.json"
        ).read_text()
    )
    return {
        "common": pause["common"],
        "cluster_config": pause["cluster_config"],
        "request_defaults": {
            "ReadStreamRequest": False,
            "WriteStreamRequest": False,
            "ExecProcessRequest": {"allowed_commands": [], "regex": []},
        },
        "containers": [
            {
                "exec_commands": [],
                "OCI": {
                    "Annotations": {"io.kubernetes.cri.image-name": image},
                    "Process": {
                        "User": {"UID": 65532, "GID": 65532},
                        "NoNewPrivileges": True,
                        "Capabilities": {key: [] for key in API["CAPABILITY_SETS"]},
                    },
                    "Root": {"Readonly": readonly},
                },
            },
            pause["container"],
        ],
    }


def policy(data):
    rules = (
        (Path(__file__).resolve().parents[2] / "tests/unit_test/lighter/cc_provision/impl/fixtures")
        / "kata-3.29-hardened-rules.rego"
    ).read_text()
    return rules + "\npolicy_data := " + json.dumps(data)


def final_workload_pod(runtime_class="kata-qemu-nvidia-gpu-snp", stream_enabled=False, image=IMAGE):
    data = policy_data(image=image)
    data["request_defaults"]["ReadStreamRequest"] = stream_enabled
    data["containers"][0]["OCI"]["Process"]["Args"] = ["python3"]
    # Structural fixtures only; no claim of executing Rego in a guest.
    raw = '[data]\n"policy.rego" = ' + "'''\n" + policy(data) + "\n'''\n"
    value = pod(image=image)
    value["spec"]["runtimeClassName"] = runtime_class
    if runtime_class in ("kata-qemu-snp", "kata-qemu-tdx"):
        value["spec"]["containers"][0]["resources"] = {}
    value["metadata"] = {
        "name": "review",
        "annotations": {
            "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(gzip.compress(raw.encode())).decode()
        },
    }
    value["spec"].update(
        {
            "hostNetwork": False,
            "hostPID": False,
            "hostIPC": False,
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
        }
    )
    value["spec"]["containers"][0].update(
        {"stdin": False, "tty": False, "command": ["python3"], "imagePullPolicy": "Always"}
    )
    return value
