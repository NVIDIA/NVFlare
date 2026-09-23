#!/usr/bin/env python3
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

"""Optional post-rehearsal GPU smoke test; never changes approved reference values."""

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import yaml

os.umask(0o077)
profile = Path(sys.argv[1]).resolve(strict=True)
first = profile / "rehearsal-collector-build"
run_id = secrets.token_hex(8)
out = profile / ("gpu-verification-" + run_id)
out.mkdir(mode=0o700)
namespace = "coco-gpu-check-" + run_id
registry = "coco-gpu-check-registry-" + run_id
kube = ["kubectl", "--kubeconfig", "/etc/kubernetes/admin.conf"]
if not os.access("/etc/kubernetes/admin.conf", os.R_OK):
    kube.insert(0, "sudo")


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


pod = yaml.safe_load((first / "collector-pod.yaml").read_text())
if pod["spec"]["runtimeClassName"] != "kata-qemu-nvidia-gpu-snp":
    raise SystemExit("Expected confidential SNP GPU runtime")
container = pod["spec"]["containers"][0]
if str(container.get("resources", {}).get("limits", {}).get("nvidia.com/pgpu")) != "1":
    raise SystemExit("Expected one passthrough GPU")
pod["metadata"]["name"] = "gpu-check"
pod["metadata"]["namespace"] = namespace
container["securityContext"] = {"privileged": False, "runAsUser": 0, "allowPrivilegeEscalation": False}
container["command"] = ["/bin/bash", "-c"]
container["args"] = [
    """set -Eeuo pipefail
echo 'GPU devices inside the non-privileged confidential container:'
ls -l /dev/nvidia* || true
command -v nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvidia-smi -q
/gpu-probe/cuda-probe
echo GPU_SMOKE_TEST_PASS
sleep 15
"""
]
container["volumeMounts"] = [{"name": "gpu-probe", "mountPath": "/gpu-probe", "readOnly": True}]
pod["spec"]["volumes"] = [{"name": "gpu-probe", "configMap": {"name": "gpu-probe", "defaultMode": 0o555}}]
pod_path = out / "pod.yaml"
pod_path.write_text(yaml.safe_dump(pod, sort_keys=False))
run(
    [
        "gcc",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(Path(__file__).with_name("gpu-probe.c")),
        "-o",
        str(out / "cuda-probe"),
        "-ldl",
    ]
)
registry_host = container["image"].split("/")[0]
created_registry = created_namespace = False
try:
    run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            registry,
            "--network",
            "host",
            "-e",
            "REGISTRY_HTTP_ADDR=" + registry_host,
            "-e",
            "REGISTRY_HTTP_TLS_CERTIFICATE=/tls/server.crt",
            "-e",
            "REGISTRY_HTTP_TLS_KEY=/tls/server.key",
            "-v",
            str(first / "registry-tls") + ":/tls:ro",
            "-v",
            str(first / "registry-data") + ":/var/lib/registry",
            "docker.io/library/registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373",
        ],
        stdout=subprocess.DEVNULL,
    )
    created_registry = True
    for _ in range(30):
        check = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--cacert",
                str(first / "registry-tls/ca.crt"),
                "https://" + registry_host + "/v2/",
            ],
            stdout=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Temporary TLS registry unavailable; its rehearsal certificate may have expired")
    run(kube + ["create", "namespace", namespace])
    created_namespace = True
    run(
        kube
        + ["-n", namespace, "create", "configmap", "gpu-probe", "--from-file=cuda-probe=" + str(out / "cuda-probe")]
    )
    run(kube + ["apply", "-f", str(pod_path)])
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        obj = json.loads(subprocess.check_output(kube + ["-n", namespace, "get", "pod", "gpu-check", "-o", "json"]))
        phase = obj["status"].get("phase")
        if phase in ("Succeeded", "Failed"):
            break
        time.sleep(2)
    else:
        raise RuntimeError("GPU Pod did not complete within 600 seconds")
    logs = subprocess.check_output(kube + ["-n", namespace, "logs", "gpu-check"], text=True)
    (out / "gpu-check.log").write_text(logs)
    (out / "pod-result.json").write_text(json.dumps(obj, indent=2) + "\n")
    print(logs)
    if phase != "Succeeded" or "GPU_SMOKE_TEST_PASS" not in logs.splitlines():
        raise RuntimeError("In-Pod GPU check failed; see " + str(out))
    print("Verified non-privileged confidential GPU Pod; evidence:", out)
finally:
    if created_namespace:
        subprocess.run(
            kube + ["-n", namespace, "describe", "pod", "gpu-check"],
            stdout=(out / "pod-describe.txt").open("w"),
            check=False,
        )
        subprocess.run(kube + ["delete", "namespace", namespace, "--wait=true", "--timeout=120s"], check=False)
    if created_registry:
        subprocess.run(["docker", "rm", "-f", registry], stdout=subprocess.DEVNULL, check=False)
