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

"""Bare-metal CVM implementation of the unified CC deployment interface."""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from nvflare.lighter.cc.vault_adapter import VaultAdapter
from nvflare.lighter.cc_provision.deployment import (
    CCArtifact,
    CCDeployment,
    CCDeploymentMode,
    CCDeploymentPlan,
    CCDeploymentResult,
    plain_data,
)
from nvflare.lighter.constants import PropKey


def _declared_path(config_path, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config_path).parent / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BareMetalCVMDeployment(CCDeployment):
    mode = CCDeploymentMode.BARE_METAL_CVM

    def __init__(self):
        self.adapters = {}

    def validate_project_config(self, project_config: Mapping[str, Any]) -> None:
        if not project_config.get("build_tools", {}).get(self.mode.value):
            raise ValueError("bare_metal_cvm requires build_tools.bare_metal_cvm")
        if not project_config.get("approval"):
            raise ValueError("bare_metal_cvm requires approval.public_key_files")
        if not any(service.get("type") == "trustee" for service in project_config["attestation_services"].values()):
            raise ValueError("bare_metal_cvm requires a Trustee attestation service")

    def create_plan(self, participant, cc_config, project_config):
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
        )

    @staticmethod
    def authorizer(plan, *, issuer):
        service = plan.attestation_service
        base = service.config_path.parent
        key = (base / service.values["attestation_signing_public_key_file"]).resolve()
        ca = (base / service.values["ca_cert_file"]).resolve()
        retry = service.values.get("retry", {})
        args = {
            "trustee_public_key": key.read_text(),
            "audience": "nvflare-trustee:" + plan.internal["project_name"],
            "max_token_age_seconds": service.values["token_expiration_seconds"],
            "token_provider": "cvm" if issuer else "verifier",
            **({"site_name": plan.internal["site_name"]} if issuer else {}),
            **(
                {
                    "kbs_url": service.values["kbs_endpoint"],
                    "kbs_ca": ca.read_text(),
                }
                if issuer
                else {}
            ),
            **({"retry_max_attempts": retry["max_attempts"]} if "max_attempts" in retry else {}),
            **({"retry_initial_delay": retry["initial_delay_seconds"]} if "initial_delay_seconds" in retry else {}),
            **({"retry_max_delay": retry["max_delay_seconds"]} if "max_delay_seconds" in retry else {}),
            **({"retry_backoff_multiplier": retry["backoff_multiplier"]} if "backoff_multiplier" in retry else {}),
            **({"retry_jitter_ratio": retry["jitter_ratio"]} if "jitter_ratio" in retry else {}),
            **(
                {"proof_iat_leeway_seconds": service.values["proof_iat_leeway_seconds"]}
                if "proof_iat_leeway_seconds" in service.values
                else {}
            ),
            "workload_constraints": plain_data(plan.internal["workload_constraints"]),
        }
        return {
            "id": "trustee_authorizer",
            "path": "nvflare.app_opt.confidential_computing.trustee_authorizer.TrusteeAuthorizer",
            "args": args,
            "token_expiration": service.values["token_expiration_seconds"],
        }

    def bind(self, plan, project_config, project, ctx):
        """Prepare the legacy vault worker before a new prod directory exists."""

        service = plan.attestation_service
        base = service.config_path.parent
        private = Path(ctx.get_state_dir()) / "cc-private-config"
        private.mkdir(mode=0o700, exist_ok=True)
        if private.is_symlink() or not private.is_dir():
            raise ValueError(f"Invalid private CVM configuration directory: {private}")
        private.chmod(0o700)
        approval = project_config["approval"]
        value = {
            "trustee": {
                "url": service.values["kbs_endpoint"],
                "ca": str((base / service.values["ca_cert_file"]).resolve()),
                "admin_token_file": str((base / service.values["admin_token_file"]).resolve()),
            },
            "approval": {
                "public_keys": [
                    str((Path(project_config["_config_path"]).parent / item).resolve())
                    for item in approval["public_key_files"]
                ]
            },
        }
        project_text = yaml.safe_dump(value, sort_keys=False)
        config_id = hashlib.sha256(project_text.encode()).hexdigest()[:16]
        project_file = private / f"cvm_project_{config_id}.yml"
        if project_file.exists():
            if project_file.is_symlink() or not project_file.is_file() or project_file.read_text() != project_text:
                raise ValueError(f"Invalid cached private CVM project configuration: {project_file}")
        else:
            fd = os.open(project_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(project_text)
        mode = plan.mode_config
        storage, network = mode["storage"], mode["network"]
        cvm_image = mode["cvm_image"]
        local_image = Path(cvm_image).expanduser()
        if "://" not in cvm_image and "@sha256:" not in cvm_image:
            cvm_image = str(
                local_image.resolve()
                if local_image.is_absolute()
                else (plan.config_path.parent / local_image).resolve()
            )
        tools = project_config["build_tools"][self.mode.value]
        settings = {
            "cvm_builder_dir": str(_declared_path(project_config["_config_path"], tools["cvm_builder_dir"])),
            "project_config": str(project_file),
            "participants": [plan.participant_name],
            "cvm_image": cvm_image,
            "docker_archive": str(plan.workload_source.values["path"]),
            "platforms": [plan.cpu_tee.value],
            "requires_gpu": plan.gpu_tee.value == "nvidia_cc",
            "vault_drive_size": storage["vault_size_gib"],
            "applog_drive_size": storage["applog_size_gib"],
            "user_config_drive_size": storage["user_config_size_gib"],
            "user_data_drive_size": storage["user_data_size_gib"],
            "allowed_ports": list(network["allowed_in_ports"]),
            "allowed_out_ports": list(network["allowed_out_ports"]),
            "allowed_in_cidrs": list(network["allowed_in_cidrs"]),
            "allowed_out_cidrs": list(network["allowed_out_cidrs"]),
            "hosts_entries": dict(mode.get("hosts_entries", {})),
            # Peer proof generation needs the measured kbs-client and TEE device.
            "host_bin": True,
            "tee_device": True,
        }
        for field in ("user_config", "user_data"):
            if field in mode:
                settings[field] = str(mode[field])
        if "output_root" in tools:
            settings["output_root"] = str(_declared_path(project_config["_config_path"], tools["output_root"]))
        project_file_path = project.get_prop(PropKey.PROJECT_FILE)
        adapter = VaultAdapter(settings, project_file_path, Path(ctx.get_workspace()).parent, project)
        self.adapters[plan.participant_name] = adapter

    def package(self, plan, private_kit, public_output, ctx):
        legacy = self.adapters[plan.participant_name].build(ctx, source_dirs={plan.participant_name: private_kit})[0]
        if public_output.is_symlink():
            raise ValueError(f"Invalid CVM public output directory: {public_output}")
        public_output.mkdir(mode=0o755, exist_ok=True)
        if not public_output.is_dir():
            raise ValueError(f"Invalid CVM public output directory: {public_output}")
        artifacts = []
        for item in legacy["artifacts"]:
            declared_source = Path(item["path"])
            if declared_source.is_symlink():
                raise ValueError(f"Invalid CVM OCI artifact: {declared_source}")
            source = declared_source.resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"Invalid CVM OCI artifact: {source}")
            destination = public_output / source.name
            if source != destination.resolve():
                if destination.exists() or destination.is_symlink():
                    raise ValueError(f"CVM public artifact already exists: {destination}")
                shutil.copyfile(source, destination)
            digest = _sha256(destination)
            if digest != item["archive_sha256"]:
                destination.unlink(missing_ok=True)
                raise ValueError(f"CVM public artifact checksum mismatch: {destination}")
            artifacts.append(
                CCArtifact(
                    artifact_type="cvm_oci",
                    path=destination.name,
                    sha256=digest,
                    metadata={
                        "platform": item["platform"],
                        "manifest_digest": item["manifest_digest"],
                        "cvm_build_id": item["cvm_build_id"],
                        "resource": item["resource"],
                    },
                )
            )
        return CCDeploymentResult(
            participant_name=plan.participant_name,
            mode=plan.mode,
            cpu_tee=plan.cpu_tee,
            gpu_tee=plan.gpu_tee,
            attestation_service=plan.attestation_service.name,
            artifacts=tuple(artifacts),
        )
