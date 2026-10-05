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

"""Backend-independent confidential-computing deployment contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Tuple

from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project


class CCDeploymentMode(str, Enum):
    BARE_METAL_CVM = "bare_metal_cvm"
    COCO = "coco"
    AZURE_CC = "azure_cc"


class CPUTEE(str, Enum):
    INTEL_TDX = "intel_tdx"
    AMD_SEV_SNP = "amd_sev_snp"


class GPUTEE(str, Enum):
    NONE = "none"
    NVIDIA_CC = "nvidia_cc"


class WorkloadSourceType(str, Enum):
    DOCKER_ARCHIVE = "docker_archive"
    DOCKER_BUILD = "docker_build"
    EXTERNAL = "external"


def plain_data(value):
    """Copy an immutable normalized value into JSON/YAML-compatible data."""

    if isinstance(value, Mapping):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_data(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedAttestationService:
    name: str
    service_type: str
    config_path: Path
    values: Mapping[str, Any]


@dataclass(frozen=True)
class WorkloadSource:
    source_type: WorkloadSourceType
    values: Mapping[str, Any]


@dataclass(frozen=True)
class CCDeploymentPlan:
    participant_name: str
    participant_type: str
    config_path: Path
    mode: CCDeploymentMode
    cpu_tee: CPUTEE
    gpu_tee: GPUTEE
    attestation_service: ResolvedAttestationService
    class_allow_list: Tuple[str, ...]
    workload_source: WorkloadSource
    mode_config: Mapping[str, Any]
    internal: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class CCArtifact:
    artifact_type: str
    path: str
    sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CCDeploymentResult:
    participant_name: str
    mode: CCDeploymentMode
    cpu_tee: CPUTEE
    gpu_tee: GPUTEE
    attestation_service: str
    artifacts: Tuple[CCArtifact, ...]


class CCDeployment(ABC):
    """Implement one deployment mode behind the common provisioning lifecycle."""

    mode: CCDeploymentMode

    @abstractmethod
    def validate_project_config(self, project_config: Mapping[str, Any]) -> None:
        """Validate the project settings consumed by this mode."""

    @abstractmethod
    def create_plan(
        self,
        participant: Participant,
        cc_config: Mapping[str, Any],
        project_config: Mapping[str, Any],
    ) -> CCDeploymentPlan:
        """Create an immutable plan from already schema-validated input."""

    def configure_startup_kit(
        self,
        plan: CCDeploymentPlan,
        project: Project,
        ctx: ProvisionContext,
    ) -> None:
        """Add mode-specific content before the common signature step."""

        return None

    @abstractmethod
    def package(
        self,
        plan: CCDeploymentPlan,
        private_kit: Path,
        public_output: Path,
        ctx: ProvisionContext,
    ) -> CCDeploymentResult:
        """Turn one verified signed kit into the mode's public deliverable."""
