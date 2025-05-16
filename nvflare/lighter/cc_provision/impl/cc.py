# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, Optional, Type

from nvflare.app_opt.confidential_computing.cc_manager import SHUTDOWN_JOB, SHUTDOWN_SYSTEM
from nvflare.lighter import utils
from nvflare.lighter.constants import PropKey, TemplateSectionKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Builder

from ..cc_constants import CC_AUTHORIZERS_KEY, CCAuthConfig, CCConfigKey, CCConfigValue, CCManagerArgs
from .mock import MockBuilder
from .onprem_cvm import OnPremCVMBuilder

JOB_RETURN_CODE_MAPPING = {
    "stop_system": SHUTDOWN_SYSTEM,
    "stop_job": SHUTDOWN_JOB,
}

CC_MGR_PATH = "nvflare.app_opt.confidential_computing.cc_manager.CCManager"


# (deploy_env, CPU_CC_MECHANISM, GPU_CC_MECHANISM)
VALID_COMPUTE_ENVS = [
    (
        CCConfigValue.ONPREM_CVM,
        CCConfigValue.AMD_SEV_SNP,
        CCConfigValue.NVIDIA_CC,
    ),
    (
        CCConfigValue.MOCK,
        None,
        None,
    ),
]


BUILDER_CLASSES = {
    CCConfigValue.ONPREM_CVM: OnPremCVMBuilder,
    CCConfigValue.MOCK: MockBuilder,
}


class CCBuilder(Builder):
    """Builder that coordinates different CC implementations (AzureCVM, OnPremCVM, etc.).

    Each CC implementation builder handles all participants that use its implementation.
    This builder also sets up the CCManager component for each participant.
    """

    def __init__(
        self,
        cc_mgr_id="cc_manager",
    ):
        self.project_name: Optional[str] = None
        self.project: Optional[Project] = None
        self.cc_config: Optional[Dict[str, Any]] = None
        # CC Manager specific
        self._cc_mgr_id = cc_mgr_id
        self._cc_enabled_sites = []
        # Map of compute environment to its builder class
        self._cc_builders: Dict[str, Type[Builder]] = {}

    def initialize(self, project: Project, ctx: ProvisionContext):
        """Initialize all CC builders needed for the project."""
        self.project_name = project.name
        self.project = project
        for participant in project.get_all_participants():
            if participant.get_prop(PropKey.CC_CONFIG):
                cc_config = self._load_cc_config(participant.get_prop(PropKey.CC_CONFIG))
                if self._validate_cc_env(cc_config):
                    self._cc_enabled_sites.append(participant)
                    participant.set_prop(PropKey.CC_ENABLED, True)
                    participant.set_prop(PropKey.CC_CONFIG_DICT, cc_config)
                    participant.set_prop(PropKey.AUTHZ_SECTION_KEY, TemplateSectionKey.CC_AUTHZ)
                    compute_env = cc_config.get(CCConfigKey.COMPUTE_ENV)
                    if compute_env not in self._cc_builders:
                        builder_class = self._get_builder_class(cc_config)
                        token_expiration = cc_config.get(CCConfigKey.CC_ATTESTATION_CONFIG, {}).get(
                            "token_expiration", 3600
                        )
                        builder = builder_class(token_expiration)
                        self._cc_builders[compute_env] = builder
                else:
                    print(f"CC is not enabled due to invalid compute environment for participant {participant.name}")
            else:
                print(f"CC is not enabled due to missing cc_config for participant {participant.name}")

        # Initialize each builder type once
        for builder in self._cc_builders.values():
            builder.initialize(project, ctx)

    def _load_cc_config(self, config_path: str) -> Dict[str, Any]:
        """Load CC configuration from YAML file."""
        if not os.path.exists(config_path):
            raise ValueError(f"CC config file not found: {config_path}")
        return utils.load_yaml(config_path)

    def _validate_cc_env(self, cc_config: Dict[str, Any]):
        compute_env = cc_config.get(CCConfigKey.COMPUTE_ENV)
        if compute_env == CCConfigValue.MOCK:
            return True

        cc_cpu_mechanism = cc_config.get(CCConfigKey.CC_CPU_MECHANISM)
        cc_gpu_mechanism = cc_config.get(CCConfigKey.CC_GPU_MECHANISM)
        env_tuple = (compute_env, cc_cpu_mechanism, cc_gpu_mechanism)

        if env_tuple not in VALID_COMPUTE_ENVS:
            return False
        return True

    def _get_builder_class(self, cc_config: Dict[str, Any]) -> Type[Builder]:
        """Get the appropriate builder class based on CC config."""
        compute_env = cc_config.get(CCConfigKey.COMPUTE_ENV)

        if compute_env in BUILDER_CLASSES:
            return BUILDER_CLASSES[compute_env]

        raise ValueError(f"Unsupported compute environment: {compute_env}")

    def _build_cc_manager_component(self, participant: Participant, ctx: ProvisionContext):
        """Build CCManager component for a participant."""
        cc_authorizers = ctx.get(CC_AUTHORIZERS_KEY)
        cc_mgr_args = {CCManagerArgs.CC_ISSUERS_CONF: [], CCManagerArgs.CC_VERIFIER_IDS: []}
        cc_enabled = participant.get_prop(PropKey.CC_ENABLED, False)
        if not cc_enabled:
            return

        cc_config = participant.get_prop(PropKey.CC_CONFIG_DICT, {})
        if cc_config == {}:
            return

        cc_verifier_ids = set([])

        for item in cc_authorizers:
            if item.get(CCAuthConfig.AUTHORIZER_ID):
                if item.get(CCAuthConfig.AUTHORIZER_ID) == cc_config.get(CCConfigKey.CC_ISSUER, ""):
                    cc_mgr_args[CCManagerArgs.CC_ISSUERS_CONF].append(
                        {
                            "issuer_id": item.get(CCAuthConfig.AUTHORIZER_ID),
                            "token_expiration": item.get(CCAuthConfig.TOKEN_EXPIRATION),
                        }
                    )
                    attestation_config = cc_config.get(CCConfigKey.CC_ATTESTATION_CONFIG)
                    if attestation_config:
                        cc_mgr_args[CCManagerArgs.VERIFY_FREQUENCY] = attestation_config.get("check_frequency", 600)
                        failure_action = attestation_config.get("failure_action", "stop_system")
                        cc_mgr_args[CCManagerArgs.CRITICAL_LEVEL] = JOB_RETURN_CODE_MAPPING.get(
                            failure_action, SHUTDOWN_SYSTEM
                        )

                cc_verifier_ids.add(item.get(CCAuthConfig.AUTHORIZER_ID))

        cc_mgr_args[CCManagerArgs.CC_ENABLED_SITES] = [e.name for e in self._cc_enabled_sites]
        cc_mgr_args[CCManagerArgs.CC_VERIFIER_IDS] = list(cc_verifier_ids)

        component = {
            "id": self._cc_mgr_id,
            "path": CC_MGR_PATH,
            "args": cc_mgr_args,
        }

        dest_dir = ctx.get_local_dir(participant)
        resources_file = os.path.join(dest_dir, f"{self._cc_mgr_id}__p_resources.json")
        utils.add_component_to_resources(resources_file, component)

    def build(self, project: Project, ctx: ProvisionContext):
        """Build CC configuration for all participants."""
        # Build CC implementation for each participant
        for builder in self._cc_builders.values():
            builder.build(project, ctx)

        # Build CCManager for each participant
        server = project.get_server()
        if server:
            self._build_cc_manager_component(server, ctx)

        for client in project.get_clients():
            self._build_cc_manager_component(client, ctx)

    def finalize(self, project: Project, ctx: ProvisionContext):
        """Finalize all CC builders."""
        for builder in self._cc_builders.values():
            builder.finalize(project, ctx)
