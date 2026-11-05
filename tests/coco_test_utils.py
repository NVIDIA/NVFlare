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

"""Shared offline CoCo fixtures and shell prerequisites, without collected tests."""

import base64
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import workload_security

ROOT = Path(__file__).resolve().parents[1] / "examples/devops/coco"
FIXTURES = Path(__file__).resolve().parent / "unit_test/lighter/cc_provision/impl/fixtures"
IMAGE = "registry.example.invalid/app@sha256:" + "a" * 64


def require_coco_bash():
    """Check shell prerequisites without skipping the Python/static tests."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("CoCo shell tests require bash on PATH")
    # Lowercase expansion and fractional read timeouts need Bash 4; empty
    # arrays under nounset also require the Bash 4.4 behavior used by the harness.
    probe = (
        "set -eu\n"
        "printf '%s\\n' \"$BASH_VERSION\"\n"
        "value=ABC; [[ ${value,,} == abc ]]\n"
        'items=(); for item in "${items[@]}"; do :; done\n'
        'read -r -t 0.05 item <<< ready; [[ "$item" == ready ]]\n'
    )
    try:
        result = subprocess.run([bash, "-c", probe], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        pytest.skip("CoCo shell prerequisite probe timed out")
    if result.returncode:
        pytest.skip(
            "CoCo shell tests require lowercase expansion, fractional read timeouts and nounset-safe empty arrays "
            f"(Bash 4.4+; macOS system Bash 3.2 is insufficient): {result.stdout.strip()} {result.stderr.strip()}"
        )
    return bash


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


def pod(readonly=False):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "spec": {
            "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
            "containers": [
                {
                    "name": "app",
                    "image": IMAGE,
                    "securityContext": context(readonly),
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                }
            ],
        },
    }


def policy_data(readonly=False):
    pause = json.loads((FIXTURES / "kata-3.29-pause-policy.json").read_text())
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
                    "Annotations": {"io.kubernetes.cri.image-name": IMAGE},
                    "Process": {
                        "User": {"UID": 65532, "GID": 65532},
                        "NoNewPrivileges": True,
                        "Capabilities": {key: [] for key in workload_security.CAPABILITY_SETS},
                    },
                    "Root": {"Readonly": readonly},
                },
            },
            pause["container"],
        ],
    }


def policy(data):
    rules = (FIXTURES / "kata-3.29-hardened-rules.rego").read_text()
    return rules + "\npolicy_data := " + json.dumps(data)


def final_workload_pod(runtime_class="kata-qemu-nvidia-gpu-snp", stream_enabled=False):
    data = policy_data()
    data["request_defaults"]["ReadStreamRequest"] = stream_enabled
    data["containers"][0]["OCI"]["Process"]["Args"] = ["python3"]
    # Structural fixtures only; no claim of executing Rego in a guest.
    raw = '[data]\n"policy.rego" = ' + "'''\n" + policy(data) + "\n'''\n"
    value = pod()
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
