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

"""Common fail-closed packager for every CC deployment mode."""

import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization

from nvflare.lighter.cc_provision.deployment import CCArtifact, CCDeploymentResult, plain_data
from nvflare.lighter.cc_provision.impl.coco_packager import _CoCoReleasePackager
from nvflare.lighter.cc_provision.impl.coco_release import _coco_runtime_class
from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.spec import Packager
from nvflare.lighter.utils import verify_folder_signature


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coco_config(plan):
    source = plan.workload_source.values
    mode = plan.mode_config
    service = plan.attestation_service
    return {
        "compute_env": "confidential_containers",
        "cc_cpu_mechanism": plan.cpu_tee.value,
        "cc_gpu": "nvidia" if plan.gpu_tee.value == "nvidia_cc" else "none",
        "role": plan.participant_type,
        "image_build": {"context": str(source["context"]), "dockerfile": str(source["dockerfile"])},
        "release_name": mode["release_name"],
        "registry_repository": mode["registry_repository"],
        "platform_config": str(mode["platform_config_file"]),
        "class_allow_list": list(plan.class_allow_list),
        "cc_issuers": [
            {
                "id": "coco_authorizer",
                "path": "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer",
                "token_expiration": service.values["token_expiration_seconds"],
                "args": {
                    "trustee_public_key_file": service.values["attestation_signing_public_key_file"],
                    "token_url": service.values["attestation_token_endpoint"],
                    **(
                        {"proof_iat_leeway_seconds": service.values["proof_iat_leeway_seconds"]}
                        if "proof_iat_leeway_seconds" in service.values
                        else {}
                    ),
                    "workload_constraints": plain_data(plan.internal["workload_constraints"]),
                },
            }
        ],
        "cc_attestation": {"check_frequency": service.values["check_frequency_seconds"]},
    }


def package_coco_plan(plan, private_kit, public_output, ctx):
    """Run the existing reviewed CoCo release pipeline for one normalized plan."""

    tools = plan.internal["build_tools"]
    project_path = plan.attestation_service.config_path
    runner_path = Path(tools["build_command"]).expanduser()
    if not runner_path.is_absolute():
        runner_path = project_path.parent / runner_path
    packager = _CoCoReleasePackager(str(runner_path.resolve()), tools.get("build_timeout_seconds", 3600))
    config = _coco_config(plan)
    owner = private_kit.parent
    # The internal CoCo release worker expects this exact private layout.
    request, runner = packager.prepare(owner, plan.config_path, config)
    registry = plan.internal["registry"]
    workload = owner / "workload.env"
    with workload.open("a") as stream:
        stream.write(f"REGISTRY_ENDPOINT={shlex.quote(registry['endpoint'])}\n")
        for key, field in (
            ("REGISTRY_CA_FILE", "ca_cert_file"),
            ("REGISTRY_USERNAME_FILE", "publisher_username_file"),
            ("REGISTRY_PASSWORD_FILE", "publisher_password_file"),
        ):
            value = Path(registry[field]).expanduser()
            if not value.is_absolute():
                value = project_path.parent / value
            stream.write(f"{key}={shlex.quote(str(value.resolve()))}\n")
        stream.write(f"KBS_URL={shlex.quote(plan.attestation_service.values['kbs_endpoint'])}\n")
        kbs_ca = Path(plan.attestation_service.values["ca_cert_file"]).expanduser()
        if not kbs_ca.is_absolute():
            kbs_ca = project_path.parent / kbs_ca
        stream.write(f"KBS_CA_FILE={shlex.quote(str(kbs_ca.resolve()))}\n")
    subprocess.run([str(runner), str(request)], cwd=plan.config_path.parent, check=True, timeout=packager.build_timeout)
    receipt = json.loads((owner / "result.json").read_text())
    if receipt.get("schema") != "nvflare-coco-build-result/v1" or receipt.get("release_name") != config["release_name"]:
        raise ValueError("Invalid CoCo build result")
    pod = Path(receipt.get("pod_yaml", ""))
    if not pod.is_absolute() or pod.is_symlink() or not pod.is_file():
        raise ValueError("Build result must name an absolute regular Pod YAML")
    packager.validate_pod(pod, config)
    public_output.mkdir(mode=0o755)
    destination = public_output / f'{config["release_name"]}-pod.yaml'
    shutil.copyfile(pod, destination)
    pod_data = json.loads(json.dumps(yaml.safe_load(destination.read_text())))
    image = pod_data["spec"]["containers"][0]["image"]
    return CCDeploymentResult(
        participant_name=plan.participant_name,
        mode=plan.mode,
        cpu_tee=plan.cpu_tee,
        gpu_tee=plan.gpu_tee,
        attestation_service=plan.attestation_service.name,
        artifacts=(
            CCArtifact(
                artifact_type="coco_pod",
                path=destination.name,
                sha256=_sha256(destination),
                metadata={"image": image, "runtime_class": _coco_runtime_class(config)},
            ),
        ),
    )


def _result_dict(result):
    return {
        "schema": "nvflare-cc-delivery/v1",
        "participant": result.participant_name,
        "cc_deployment_mode": result.mode.value,
        "cpu_tee": result.cpu_tee.value,
        "gpu_tee": result.gpu_tee.value,
        "attestation_service": result.attestation_service,
        "artifacts": [
            {
                "type": artifact.artifact_type,
                "path": artifact.path,
                "sha256": artifact.sha256,
                **dict(artifact.metadata),
            }
            for artifact in result.artifacts
        ],
    }


class CCPackager(Packager):
    """Own the private/public handoff for a mixed-mode CC project."""

    def package(self, project, ctx):
        plans = ctx.get(CtxKey.CC_DEPLOYMENT_PLANS, {})
        if not plans:
            raise ValueError("CCPackager requires at least one confidential participant")
        result_root = Path(ctx.get_result_location()).resolve()
        root_cert = ctx.get(CtxKey.ROOT_CERT)
        if root_cert is None:
            raise ValueError("CCPackager requires the project root certificate")
        expected_root = root_cert.public_bytes(serialization.Encoding.PEM)

        # Verify all kits before hiding any of them or starting a privileged or
        # external build. A partial failure never publishes a plaintext kit.
        for participant in project.get_all_participants():
            if participant.name not in plans:
                continue
            kit = result_root / participant.name
            if kit.is_symlink() or not kit.is_dir():
                raise ValueError(f"Missing generated startup kit for {participant.name}")
            if (kit / "startup/rootCA.pem").read_bytes() != expected_root:
                raise ValueError(f"Startup kit for {participant.name} does not use the project root certificate")
            if not verify_folder_signature(
                str(kit),
                str(kit / "startup/rootCA.pem"),
                single_signer=True,
                signature_file=ProvFileName.SIGNATURE_JSON,
            ):
                raise ValueError(f"Invalid signed startup kit for {participant.name}")

        private_root = Path(ctx.get_state_dir()) / "cc-private"
        private_root.mkdir(mode=0o700, exist_ok=True)
        if private_root.is_symlink() or not private_root.is_dir():
            raise ValueError(f"Invalid private CC staging directory: {private_root}")
        private_root.chmod(0o700)
        private = private_root / result_root.name
        if private.exists():
            archive = Path(tempfile.mkdtemp(prefix=f"{result_root.name}.superseded-", dir=private_root))
            private.rename(archive / result_root.name)
        private.mkdir(mode=0o700)
        aggregate = result_root / ProvFileName.START_ALL_SH
        if aggregate.exists():
            aggregate.rename(private / ProvFileName.START_ALL_SH)
        sources = {}
        for name in plans:
            owner = private / name
            owner.mkdir(mode=0o700)
            source = result_root / name
            source.rename(owner / "startup-kit")
            sources[name] = owner / "startup-kit"

        results = []
        deployments = ctx["cc_deployments"]
        for name, plan in plans.items():
            try:
                mode_result = deployments[plan.mode].package(plan, sources[name], result_root / name, ctx)
                results.append(_result_dict(mode_result))
            except Exception:
                ctx.error(f"CC packaging failed for {name}; private recovery inputs retained at {private / name}")
                raise

        manifest_dir = result_root / "cc_manifests"
        manifest_dir.mkdir(mode=0o755)
        for result in results:
            (manifest_dir / f'{result["participant"]}.json').write_text(json.dumps(result, indent=2) + "\n")
        ctx[CtxKey.CC_DEPLOYMENT_RESULTS] = results
        ctx.info(
            f"CC delivery manifests: {manifest_dir}. Private recovery inputs: {private}. Do not distribute state/."
        )
