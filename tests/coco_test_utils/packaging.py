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
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.lighter.cc_provision.deployment import CPUTEE, GPUTEE
from nvflare.lighter.cc_provision.impl.coco_packager import COMMAND


def coco_pod(runtime, gpu, image):
    pytest.importorskip("tomllib", reason="full CoCo Pod packaging requires a Python 3.11+ deployment host")
    from tests.coco_test_utils.workload_context import context, policy, policy_data

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
