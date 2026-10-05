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
import shutil
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from nvflare.lighter.cc_provision.deployment import (
    CCArtifact,
    CCDeployment,
    CCDeploymentMode,
    CCDeploymentPlan,
    CCDeploymentResult,
)


class AzureCCDeployment(CCDeployment):
    mode = CCDeploymentMode.AZURE_CC

    def validate_project_config(self, project_config: Mapping[str, Any]) -> None:
        services = project_config.get("attestation_services", {})
        if not any(service.get("type") == "azure_maa" for service in services.values()):
            raise ValueError("azure_cc requires an azure_maa attestation service")

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
        target = plan.mode_config["deployment_target"]
        endpoint = urlsplit(plan.attestation_service.values["endpoint"]).netloc
        if target == "confidential_vm":
            return {
                "id": "az_cvm_authorizer",
                "path": "nvflare.app_opt.confidential_computing.az_cvm_authorizer.AZCVMAuthorizer",
                "args": {"maa_endpoint": endpoint},
                "token_expiration": plan.attestation_service.values["token_expiration_seconds"],
            }
        return {
            "id": "aci_authorizer",
            "path": "nvflare.app_opt.confidential_computing.aci_authorizer.ACIAuthorizer",
            "args": {"maa_endpoint": endpoint},
            "token_expiration": plan.attestation_service.values["token_expiration_seconds"],
        }

    def package(self, plan, private_kit: Path, public_output: Path, ctx):
        if public_output.exists():
            raise ValueError(f"Azure CC public output already exists: {public_output}")
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
