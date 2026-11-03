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

import base64
import gzip
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from nvflare.lighter.cc_provision.deployment import CPUTEE, GPUTEE
from nvflare.lighter.cc_provision.impl.coco_packager import COMMAND, CoCoPlanPackager
from nvflare.lighter.cc_provision.impl.coco_release import COCO_STARTUP_PROLOGUE, coco_runtime_class

RUNTIME_CASES = [
    (CPUTEE.AMD_SEV_SNP, GPUTEE.NVIDIA_CC, "kata-qemu-nvidia-gpu-snp"),
    (CPUTEE.AMD_SEV_SNP, GPUTEE.NONE, "kata-qemu-snp"),
    (CPUTEE.INTEL_TDX, GPUTEE.NVIDIA_CC, "kata-qemu-nvidia-gpu-tdx"),
    (CPUTEE.INTEL_TDX, GPUTEE.NONE, "kata-qemu-tdx"),
]


def coco_pod(runtime, gpu, image):
    pytest.importorskip("tomllib", reason="full CoCo Pod packaging requires a Python 3.11+ deployment host")
    from tests.unit_test.lighter.cc_provision.impl.workload_security_context_test import context, policy, policy_data

    data = policy_data()
    data["containers"][0]["OCI"]["Annotations"]["io.kubernetes.cri.image-name"] = image
    data["containers"][0]["OCI"]["Process"]["Args"] = COMMAND
    initdata = '[data]\n"policy.rego" = ' + "'''\n" + policy(data) + "\n'''\n"
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "annotations": {
                "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                    gzip.compress(initdata.encode())
                ).decode()
            }
        },
        "spec": {
            "runtimeClassName": runtime,
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "restartPolicy": "Never",
            "containers": [
                {
                    "image": image,
                    "command": COMMAND,
                    "imagePullPolicy": "Always",
                    "securityContext": context(),
                    "stdin": False,
                    "tty": False,
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}} if gpu == "nvidia" else {},
                }
            ],
        },
    }


def make_plan(tmp_path, *, cpu=CPUTEE.AMD_SEV_SNP, gpu=GPUTEE.NVIDIA_CC):
    context = tmp_path / "site"
    context.mkdir(exist_ok=True)
    (context / "Dockerfile").write_text("FROM reviewed-base\n")
    platform = tmp_path / "admin/platform.env"
    platform.parent.mkdir(exist_ok=True)
    platform.write_text("# fixture\n")
    runner = tmp_path / "build"
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o700)
    return SimpleNamespace(
        participant_name="site-1",
        participant_type="client",
        cpu_tee=cpu,
        gpu_tee=gpu,
        workload_source=SimpleNamespace(values={"context": context, "dockerfile": Path("Dockerfile")}),
        mode_config={
            "release_name": "site-v1",
            "registry_repository": "workloads/site",
            "platform_config_file": platform,
        },
        internal={"build_tools": {"build_command": str(runner)}},
    )


@pytest.mark.parametrize("cpu,gpu,runtime", RUNTIME_CASES)
def test_runtime_class_uses_normalized_plan(cpu, gpu, runtime):
    plan = SimpleNamespace(cpu_tee=cpu, gpu_tee=gpu)
    assert coco_runtime_class(plan) == runtime


@pytest.mark.parametrize("timeout", [None, True, False, 0, -1, 1.5, "60"])
def test_invalid_build_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="build_timeout"):
        CoCoPlanPackager(timeout)


def test_prepare_consumes_normalized_plan(tmp_path):
    plan = make_plan(tmp_path)
    owner = tmp_path / "private/site-1"
    kit = owner / "startup-kit"
    (kit / "startup").mkdir(parents=True)
    (kit / "startup/sub_start.sh").write_text(COCO_STARTUP_PROLOGUE + "exec nvflare\n")
    for name in ("startup/rootCA.pem", "startup/client.key", "signature.json"):
        (kit / name).write_text("fixture")

    with patch("nvflare.lighter.cc_provision.impl.coco_packager.verify_folder_signature", return_value=True):
        request, runner = CoCoPlanPackager().prepare(owner, plan)

    assert runner == Path(plan.internal["build_tools"]["build_command"])
    request_data = json.loads(request.read_text())
    workload = dict(
        shlex.split(line)[0].split("=", 1) for line in Path(request_data["workload_env"]).read_text().splitlines()
    )
    assert workload["RELEASE_NAME"] == plan.mode_config["release_name"]
    assert workload["REGISTRY_REPOSITORY"] == plan.mode_config["registry_repository"]
    assert workload["COCO_RUNTIME_CLASS"] == "kata-qemu-nvidia-gpu-snp"
    assert workload["COCO_GPU_COUNT"] == "1"
    assert (Path(workload["BUILD_CONTEXT"]) / ".nvflare-kit/signature.json").is_file()


@pytest.mark.parametrize("cpu,gpu,runtime", RUNTIME_CASES)
def test_validate_pod_uses_normalized_cpu_gpu_and_repository(tmp_path, cpu, gpu, runtime):
    plan = make_plan(tmp_path, cpu=cpu, gpu=gpu)
    image = "registry.example.com/" + plan.mode_config["registry_repository"] + "@sha256:" + "a" * 64
    pod = tmp_path / "pod.yaml"
    pod.write_text(yaml.safe_dump(coco_pod(runtime, "nvidia" if gpu is GPUTEE.NVIDIA_CC else "none", image)))
    CoCoPlanPackager.validate_pod(pod, plan)


@pytest.mark.parametrize("change", ["runtime", "gpu", "repository", "tag"])
def test_validate_pod_rejects_output_outside_normalized_plan(tmp_path, change):
    plan = make_plan(tmp_path)
    image = "registry.example.com/workloads/site@sha256:" + "a" * 64
    pod = coco_pod("kata-qemu-nvidia-gpu-snp", "nvidia", image)
    if change == "runtime":
        pod["spec"]["runtimeClassName"] = "kata-qemu-snp"
    elif change == "gpu":
        pod["spec"]["containers"][0]["resources"] = {}
    elif change == "repository":
        pod["spec"]["containers"][0]["image"] = "registry.example.com/other@sha256:" + "a" * 64
    else:
        pod["spec"]["containers"][0]["image"] = "registry.example.com/workloads/site:latest"
    path = tmp_path / "pod.yaml"
    path.write_text(yaml.safe_dump(pod))
    with pytest.raises(ValueError):
        CoCoPlanPackager.validate_pod(path, plan)
