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

"""Package signed client kits using the trusted provisioning-node CoCo workflow."""

import base64
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from nvflare.lighter.cc_provision.impl.coco import resolve_cc_config, validate_coco_config
from nvflare.lighter.constants import PropKey, ProvFileName
from nvflare.lighter.spec import Packager
from nvflare.lighter.utils import load_yaml

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


class CoCoPackager(Packager):
    def __init__(self, build_image_cmd="build_coco_image.sh"):
        if not isinstance(build_image_cmd, str) or not build_image_cmd:
            raise ValueError("build_image_cmd must name a trusted executable")
        self.build_image_cmd = build_image_cmd

    def package(self, project, ctx):
        result = Path(ctx.get_result_location()).resolve()
        selected = []
        for participant in project.get_all_participants():
            cc_path = participant.get_prop(PropKey.CC_CONFIG)
            if not cc_path:
                continue
            config_path = Path(resolve_cc_config(project, cc_path))
            config = load_yaml(config_path)
            validate_coco_config(config)
            if participant.type != "client":
                raise ValueError("CoCoPackager supports clients only")
            if not participant.get_prop(PropKey.CC_ENABLED) or participant.get_prop(PropKey.CC_CONFIG_DICT) != config:
                raise ValueError("CoCo client was not configured by CCBuilder, or configuration changed")
            selected.append((participant, config_path, config))
        if not selected:
            raise ValueError("CoCoPackager requires at least one CoCo client")

        # Keep every selected plaintext kit outside prod before any external
        # build starts. On failure no selected client's directory can be
        # mistaken for a handoff. Never delete a kit as the CVM packager does.
        private_root = Path(ctx.get_state_dir()) / "coco-private"
        private_root.mkdir(mode=0o700, exist_ok=True)
        private_root.chmod(0o700)
        private = private_root / result.name
        private.mkdir(mode=0o700)
        # The aggregate launcher assumes every client is a plaintext kit.
        aggregate = result / ProvFileName.START_ALL_SH
        if aggregate.exists():
            aggregate.rename(private / ProvFileName.START_ALL_SH)
        for participant, _, _ in selected:
            owner = private / participant.name
            owner.mkdir(mode=0o700)
            source = result / participant.name
            if source.is_symlink() or not source.is_dir():
                raise ValueError("Expected a generated participant startup kit")
            source.rename(owner / "startup-kit")
            (owner / "startup-kit").chmod(0o700)

        bindings = {}
        for participant, config_path, config in selected:
            owner = private / participant.name
            try:
                request, runner = self.prepare(owner, config_path, config)
                subprocess.run([str(runner), str(request)], cwd=config_path.parent, check=True)
                receipt = json.loads((owner / "result.json").read_text())
                if (
                    receipt.get("schema") != "nvflare-coco-build-result/v1"
                    or receipt.get("release_name") != config["release_name"]
                ):
                    raise ValueError("Invalid CoCo build result")
                pod = Path(receipt["pod_yaml"])
                if not pod.is_absolute() or pod.is_symlink() or not pod.is_file():
                    raise ValueError("Build result must name an absolute regular Pod YAML")
                self.validate_pod(pod, config)
                data = yaml.safe_load(pod.read_text())
                annotation = data["metadata"]["annotations"]["io.katacontainers.config.hypervisor.cc_init_data"]
                raw = gzip.decompress(base64.b64decode(annotation, validate=True))
                bindings[participant.name] = {
                    "init_data": hashlib.sha256(raw).hexdigest(),
                    "image": data["spec"]["containers"][0]["image"],
                    "args": COMMAND,
                }
                public = result / participant.name
                public.mkdir(mode=0o755)
                shutil.copyfile(pod, public / f'{config["release_name"]}-pod.yaml')
                ctx.info(f"CoCo IT handoff: {public}. Private build/receipt: {owner}. Do not distribute state/.")
            except Exception:
                ctx.error(f"CoCo packaging failed for {participant.name}; private recovery inputs retained at {owner}")
                raise
        # Only the trusted, non-confidential server needs all final image and
        # InitData pins. Embedding those in clients would create a hash cycle.
        target = result / project.get_server().name / "local/coco_authorizer__p_resources.json"
        configuration = json.loads(target.read_text())
        configuration["components"][0]["args"]["expected_workloads"] = bindings
        target.write_text(json.dumps(configuration, indent=2) + "\n")

    def prepare(self, owner, config_path, config):
        base = config_path.parent
        context = (base / config["image_build"]["context"]).resolve()
        dockerfile = (context / config["image_build"]["dockerfile"]).resolve()
        platform = (base / config["platform_config"]).resolve()
        runner = (base / self.build_image_cmd).resolve()
        if not context.is_dir() or not dockerfile.is_file() or not platform.is_file():
            raise ValueError("Missing build context, Dockerfile, or platform configuration")
        if owner.resolve().is_relative_to(context):
            raise ValueError("Build context must not contain the provisioning workspace")
        if platform.name != "platform.env":
            raise ValueError("platform_config must point to the prepared admin kit's platform.env")
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise ValueError(f"Build command is not executable: {runner}")
        kit = owner / "startup-kit"
        for name in ("startup/sub_start.sh", "startup/rootCA.pem", "startup/client.key", "signature.json"):
            if not (kit / name).is_file():
                raise ValueError(
                    f"Missing signed client kit input: {name}; order CertBuilder/SignatureBuilder correctly"
                )
        for name in (".nvflare-kit", "Dockerfile.coco", "Dockerfile.coco.dockerignore"):
            if (context / name).exists():
                raise ValueError(f"Reserved CoCo build-context name: {name}")
        build = owner / "build-context"
        copy_private_tree(context, build)
        copy_private_tree(kit, build / ".nvflare-kit")
        # Append to the final application stage: the operator supplies all
        # NVFlare/custom-code dependencies, we supply the freshly signed kit.
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
            "RELEASE_NAME": config["release_name"],
            "REGISTRY_REPOSITORY": config["registry_repository"],
            "BUILD_CONTEXT": str(build),
            "DOCKERFILE": str(build / "Dockerfile.coco"),
            "APP_COMMAND_JSON": json.dumps(COMMAND),
            "APP_UID": "65532",
            "APP_GID": "65532",
            # NVFlare writes logs/jobs into guest-local writable image storage.
            # No hostPath/volume is added; genpolicy binds this exact setting.
            "APP_READ_ONLY_ROOT_FILESYSTEM": "false",
        }
        private_write(workload, "".join(f"{k}={shlex.quote(v)}\n" for k, v in values.items()))
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
    def validate_pod(path, config):
        pod = yaml.safe_load(path.read_text())
        if not isinstance(pod, dict) or pod.get("kind") != "Pod" or pod.get("apiVersion") != "v1":
            raise ValueError("Build did not produce a v1 Pod")
        spec = pod.get("spec", {})
        containers = spec.get("containers", [])
        if spec.get("runtimeClassName") != "kata-qemu-nvidia-gpu-snp" or len(containers) != 1:
            raise ValueError("Expected one CoCo SNP/GPU container")
        c = containers[0]
        if c.get("command") != COMMAND or c.get("resources", {}).get("limits") != {"nvidia.com/pgpu": "1"}:
            raise ValueError("Unexpected CoCo command or GPU allocation")
        if not re.fullmatch(
            r"[^\s]+/" + re.escape(config["registry_repository"]) + r"@sha256:[0-9a-f]{64}", c.get("image", "")
        ):
            raise ValueError("Pod image must be digest-pinned in the configured repository")
        if not pod.get("metadata", {}).get("annotations", {}).get("io.katacontainers.config.hypervisor.cc_init_data"):
            raise ValueError("Pod lacks measured init-data")
        if any(key in spec for key in ("volumes", "initContainers", "ephemeralContainers", "imagePullSecrets")):
            raise ValueError("Pod contains unapproved storage, containers, or credentials")
