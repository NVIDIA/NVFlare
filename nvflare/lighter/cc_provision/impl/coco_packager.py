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

"""Package one normalized CoCo deployment plan on the trusted provisioning node."""

import json
import os
import re
import shlex
import shutil
from pathlib import Path

from nvflare.lighter.cc_provision.impl.coco_release import COCO_STARTUP_PROLOGUE, coco_runtime_class
from nvflare.lighter.cc_provision.workload_security import read_pod, validate_workload_pod
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.utils import verify_folder_signature

COMMAND = ["/opt/nvflare/startup/sub_start.sh", "--once", "--verify"]


def private_write(path, text):
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as output:
        output.write(text)


def copy_private_tree(source, destination):
    """Do not follow symlinks out of a build context or a generated kit."""
    for path in [source, *source.rglob("*")]:
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Build input must contain only regular files/directories: {path}")
    shutil.copytree(source, destination)
    destination.chmod(0o700)


class CoCoPlanPackager:
    """Prepare and validate a CoCo release directly from ``CCDeploymentPlan``."""

    def __init__(self, build_timeout=3600):
        if type(build_timeout) is not int or build_timeout <= 0:
            raise ValueError("build_timeout must be a positive integer number of seconds")
        self.build_timeout = build_timeout

    def prepare(self, owner, plan):
        runtime_class = coco_runtime_class(plan)
        source = plan.workload_source.values
        mode = plan.mode_config
        context = Path(source["context"])
        dockerfile = (context / source["dockerfile"]).resolve()
        platform = Path(mode["platform_config_file"])
        runner = Path(plan.internal["build_tools"]["build_command"])
        if not context.is_dir() or not dockerfile.is_file() or not platform.is_file():
            raise ValueError("Missing build context, Dockerfile, or platform configuration")
        if owner.resolve().is_relative_to(context.resolve()):
            raise ValueError("Build context must not contain the provisioning workspace")
        if platform.name != "platform.env":
            raise ValueError("platform_config_file must point to the prepared admin kit's platform.env")
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise ValueError(f"Build command is not executable: {runner}")
        kit = owner / "startup-kit"
        key_name = {"client": "client.key", "server": "server.key"}[plan.participant_type]
        for name in ("startup/sub_start.sh", "startup/rootCA.pem", f"startup/{key_name}", "signature.json"):
            if not (kit / name).is_file():
                raise ValueError(
                    f"Missing signed {plan.participant_type} kit input: {name}; "
                    "order CertBuilder/SignatureBuilder correctly"
                )
        if not (kit / "startup/sub_start.sh").read_text().startswith(COCO_STARTUP_PROLOGUE):
            raise ValueError("CoCo startup must discard host-visible output before startup-kit signing")
        if not verify_folder_signature(
            str(kit), str(kit / "startup/rootCA.pem"), single_signer=True, signature_file=ProvFileName.SIGNATURE_JSON
        ):
            raise ValueError("CoCo startup kit signature verification failed; do not modify kits after signing")
        for name in (".nvflare-kit", "Dockerfile.coco", "Dockerfile.coco.dockerignore"):
            if (context / name).exists():
                raise ValueError(f"Reserved CoCo build-context name: {name}")
        build = owner / "build-context"
        copy_private_tree(context, build)
        copy_private_tree(kit, build / ".nvflare-kit")
        private_write(
            build / "Dockerfile.coco",
            dockerfile.read_text().rstrip()
            + "\n\n"
            + "COPY --chown=65532:65532 .nvflare-kit/ /opt/nvflare/\n"
            + "ENV NVFL_WORKSPACE=/opt/nvflare PYTHONDONTWRITEBYTECODE=1\n"
            + "WORKDIR /opt/nvflare\nUSER 65532:65532\n"
            + "ENTRYPOINT "
            + json.dumps(COMMAND)
            + "\nCMD []\n",
        )
        ignore = build / ".dockerignore"
        existing = ignore.read_text() if ignore.exists() else ""
        ignore.write_text(existing.rstrip() + "\n!.nvflare-kit\n!.nvflare-kit/**\n!Dockerfile.coco\n")
        workload = owner / "workload.env"
        values = {
            "RELEASE_NAME": mode["release_name"],
            "REGISTRY_REPOSITORY": mode["registry_repository"],
            "COCO_RUNTIME_CLASS": runtime_class,
            "COCO_GPU_COUNT": "1" if plan.gpu_tee.value == "nvidia_cc" else "0",
            "BUILD_CONTEXT": str(build),
            "DOCKERFILE": str(build / "Dockerfile.coco"),
            "APP_COMMAND_JSON": json.dumps(COMMAND),
            "APP_UID": "65532",
            "APP_GID": "65532",
            "APP_READ_ONLY_ROOT_FILESYSTEM": "false",
        }
        private_write(workload, "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
        request = owner / "build-request.json"
        private_write(
            request,
            json.dumps(
                {
                    "schema": "nvflare-coco-build-request/v1",
                    "workload_env": str(workload),
                    "admin_dir": str(platform.parent),
                    "result_file": str(owner / "result.json"),
                },
                indent=2,
            )
            + "\n",
        )
        return request, runner

    @staticmethod
    def validate_pod(path, plan):
        runtime_class = coco_runtime_class(plan)
        pod = read_pod(path)
        validate_workload_pod(
            pod,
            {
                "privileged": False,
                "allowPrivilegeEscalation": False,
                "runAsNonRoot": True,
                "runAsUser": 65532,
                "runAsGroup": 65532,
                "readOnlyRootFilesystem": False,
                "capabilities": {"drop": ["ALL"]},
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            COMMAND,
            runtime_class=runtime_class,
        )
        if not isinstance(pod, dict) or pod.get("kind") != "Pod" or pod.get("apiVersion") != "v1":
            raise ValueError("Build did not produce a v1 Pod")
        spec = pod.get("spec", {})
        containers = spec.get("containers", [])
        if spec.get("runtimeClassName") != runtime_class or len(containers) != 1:
            raise ValueError(f"Expected one CoCo container using {runtime_class}")
        container = containers[0]
        expected_resources = {"nvidia.com/pgpu": "1"} if plan.gpu_tee.value == "nvidia_cc" else {}
        resources = container.get("resources", {})
        if (
            container.get("command") != COMMAND
            or not isinstance(resources, dict)
            or set(resources) - {"limits", "requests"}
            or resources.get("limits", {}) != expected_resources
            or resources.get("requests", {}) not in ({}, expected_resources)
        ):
            raise ValueError("Unexpected CoCo command or resource allocation")
        repository = plan.mode_config["registry_repository"]
        if not re.fullmatch(r"[^\s]+/" + re.escape(repository) + r"@sha256:[0-9a-f]{64}", container.get("image", "")):
            raise ValueError("Pod image must be digest-pinned in the configured repository")
        if not pod.get("metadata", {}).get("annotations", {}).get("io.katacontainers.config.hypervisor.cc_init_data"):
            raise ValueError("Pod lacks measured init-data")
        if any(key in spec for key in ("volumes", "initContainers", "ephemeralContainers", "imagePullSecrets")):
            raise ValueError("Pod contains unapproved storage, containers, or credentials")
