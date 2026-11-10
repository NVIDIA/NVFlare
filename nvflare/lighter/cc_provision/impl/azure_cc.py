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

"""Azure implementation of the unified CC deployment interface."""

import hashlib
import io
import shutil
import stat
import tarfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.app_opt.confidential_computing.aci_authorizer import ACI_NAMESPACE
from nvflare.app_opt.confidential_computing.az_cvm_authorizer import AZ_CVM_NAMESPACE
from nvflare.lighter.cc_provision.deployment import (
    CCArtifact,
    CCDeployment,
    CCDeploymentMode,
    CCDeploymentPlan,
    CCDeploymentResult,
    immutable_data,
)
from nvflare.lighter.constants import ParticipantType
from nvflare.lighter.utils import load_private_key_file, sign_content

MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 4096
POLICY_FILE = "approved_cce_policy.txt"
ARCHIVE_FILE = "startup-kit.tar"
SIGNATURE_FILE = "startup-kit.sig"


class AzureCCDeployment(CCDeployment):
    mode = CCDeploymentMode.AZURE_CC

    def validate_project_config(self, project_config: Mapping[str, Any]) -> None:
        services = project_config.get("attestation_services", {})
        if not any(service.get("type") == "azure_maa" for service in services.values()):
            raise ValueError("azure_cc requires an azure_maa attestation service")

    def create_plan(self, participant, cc_config, project_config):
        internal = {}
        if cc_config["mode_config"]["deployment_target"] == "confidential_container":
            if participant.type != ParticipantType.SERVER:
                raise ValueError("ACI confidential_container is supported only for the server")
            try:
                policy = Path(cc_config["mode_config"]["cce_policy_file"]).read_bytes()
            except OSError:
                raise ValueError("Unable to read azure_cc.cce_policy_file") from None
            if not policy.strip():
                raise ValueError("azure_cc.cce_policy_file must contain a nonempty decoded policy")
            if len(policy) > MAX_EXTRACTED_BYTES:
                raise ValueError("azure_cc.cce_policy_file exceeds the startup-kit size limit")
            try:
                key = load_private_key_file(cc_config["mode_config"]["startup_kit_signing_key_file"])
            except (OSError, ValueError, TypeError):
                raise ValueError(
                    "Unable to load azure_cc.startup_kit_signing_key_file as an unencrypted private key"
                ) from None
            if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
                raise ValueError(
                    "azure_cc.startup_kit_signing_key_file must contain an RSA private key of at least 2048 bits"
                )
            internal = {
                "cce_policy": policy,
                "expected_policy_hash": hashlib.sha256(policy).hexdigest(),
                "signing_key": key,
            }
        return CCDeploymentPlan(
            participant_name=participant.name,
            participant_type=participant.type,
            config_path=cc_config["config_path"],
            mode=self.mode,
            cpu_tee=cc_config["cpu_tee"],
            gpu_tee=cc_config["gpu_tee"],
            attestation_service=cc_config["attestation_service"],
            class_allow_list=cc_config["class_allow_list"],
            workload_source=cc_config["workload_source"],
            mode_config=cc_config["mode_config"],
            internal=immutable_data(internal),
        )

    def configure_startup_kit(self, plan, project, ctx):
        if plan.mode_config["deployment_target"] == "confidential_container":
            participant = project.get_server()
            local = Path(ctx.get_local_dir(participant))
            local.mkdir(parents=True, exist_ok=True)
            (local / POLICY_FILE).write_bytes(plan.internal["cce_policy"])

    @staticmethod
    def authorizer(plan, *, issuer):
        target = plan.mode_config["deployment_target"]
        endpoint = urlsplit(plan.attestation_service.values["endpoint"]).netloc.lower()
        suffix = hashlib.sha256(endpoint.encode()).hexdigest()[:12]
        if target == "confidential_vm":
            return {
                "id": f"az_cvm_authorizer_{suffix}",
                "path": "nvflare.app_opt.confidential_computing.az_cvm_authorizer.AZCVMAuthorizer",
                "args": {"maa_endpoint": endpoint, "namespace": f"{AZ_CVM_NAMESPACE}-{suffix}"},
                "token_expiration": plan.attestation_service.values["token_expiration_seconds"],
            }
        return {
            "id": f"aci_authorizer_{suffix}",
            "path": "nvflare.app_opt.confidential_computing.aci_authorizer.ACIAuthorizer",
            "args": {
                "maa_endpoint": endpoint,
                "namespace": f"{ACI_NAMESPACE}-{suffix}",
                "expected_policy_hash": plan.internal["expected_policy_hash"],
            },
            "token_expiration": plan.attestation_service.values["token_expiration_seconds"],
        }

    def package(self, plan, private_kit: Path, public_output: Path, ctx):
        if public_output.exists():
            raise ValueError(f"Azure CC public output already exists: {public_output}")
        if plan.mode_config["deployment_target"] == "confidential_container":
            return self._package_aci(plan, private_kit, public_output)
        shutil.copytree(private_kit, public_output)
        digest = hashlib.sha256()
        for path in sorted(item for item in public_output.rglob("*") if item.is_file()):
            digest.update(path.relative_to(public_output).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return CCDeploymentResult(
            participant_name=plan.participant_name,
            mode=plan.mode,
            cpu_tee=plan.cpu_tee,
            gpu_tee=plan.gpu_tee,
            attestation_service=plan.attestation_service.name,
            artifacts=(
                CCArtifact(
                    artifact_type="azure_startup_kit",
                    path=public_output.name,
                    sha256=digest.hexdigest(),
                    metadata={"deployment_target": plan.mode_config["deployment_target"]},
                ),
            ),
        )

    @staticmethod
    def _package_aci(plan, private_kit, public_output):
        # The common packager has already verified the complete signed kit.
        # Never read the policy/key source paths again after plan preparation.
        if (private_kit / "local" / POLICY_FILE).read_bytes() != plan.internal["cce_policy"]:
            raise ValueError("Signed server policy differs from the approved policy snapshot")
        paths = sorted(private_kit.rglob("*"))
        if len(paths) > MAX_ARCHIVE_ENTRIES:
            raise ValueError("ACI startup kit exceeds the archive entry limit")
        total_size = 0
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT, dereference=True) as archive:
            for path in paths:
                mode = path.lstat().st_mode
                if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    raise ValueError("ACI startup kit must contain only regular files and directories")
                info = archive.gettarinfo(str(path), arcname=path.relative_to(private_kit).as_posix())
                if info.isfile():
                    total_size += info.size
                    if total_size > MAX_EXTRACTED_BYTES:
                        raise ValueError("ACI startup kit exceeds the extracted size limit")
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)
        contents = buffer.getvalue()
        if len(contents) > MAX_ARCHIVE_BYTES:
            raise ValueError("ACI startup kit exceeds the archive size limit")
        signature = sign_content(contents, plan.internal["signing_key"], return_str=False)
        public_output.mkdir(mode=0o700)
        (public_output / ARCHIVE_FILE).write_bytes(contents)
        (public_output / SIGNATURE_FILE).write_bytes(signature)
        (public_output / POLICY_FILE).write_bytes(plan.internal["cce_policy"])
        return CCDeploymentResult(
            participant_name=plan.participant_name,
            mode=plan.mode,
            cpu_tee=plan.cpu_tee,
            gpu_tee=plan.gpu_tee,
            attestation_service=plan.attestation_service.name,
            artifacts=tuple(
                CCArtifact(
                    artifact_type=kind,
                    path=f"{public_output.name}/{name}",
                    sha256=hashlib.sha256(data).hexdigest(),
                    metadata={
                        "deployment_target": "confidential_container",
                        "policy_sha256": plan.internal["expected_policy_hash"],
                    },
                )
                for name, data, kind in (
                    (ARCHIVE_FILE, contents, "azure_aci_startup_archive"),
                    (SIGNATURE_FILE, signature, "azure_aci_startup_signature"),
                    (POLICY_FILE, plan.internal["cce_policy"], "azure_cce_policy"),
                )
            ),
        )
