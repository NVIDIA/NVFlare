# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unified confidential-computing provisioning dispatcher."""

import json
import os
from dataclasses import replace
from pathlib import Path

from nvflare.apis.fl_constant import SiteType
from nvflare.app_opt.confidential_computing.cc_manager import CC_ISSUER_ID, TOKEN_EXPIRATION
from nvflare.app_opt.confidential_computing.cc_timeouts import resolve_token_timeouts
from nvflare.lighter.cc_provision.config import load_participant_config, load_project_config
from nvflare.lighter.cc_provision.deployment import GPUTEE, CCDeploymentMode, immutable_data, plain_data
from nvflare.lighter.cc_provision.impl.azure_cc import AzureCCDeployment
from nvflare.lighter.cc_provision.impl.bare_metal_cvm import BareMetalCVMDeployment
from nvflare.lighter.cc_provision.impl.coco import CoCoDeployment
from nvflare.lighter.cc_provision.utils import resolve_cc_config
from nvflare.lighter.constants import CtxKey, ParticipantType, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.spec import Builder

CC_MGR_PATH = "nvflare.app_opt.confidential_computing.cc_manager.CCManager"
CC_PACKAGER_PATH = "nvflare.lighter.cc_provision.impl.cc_packager.CCPackager"

DEPLOYMENTS = {
    CCDeploymentMode.BARE_METAL_CVM: BareMetalCVMDeployment,
    CCDeploymentMode.COCO: CoCoDeployment,
    CCDeploymentMode.AZURE_CC: AzureCCDeployment,
}


def _site_name(participant):
    return SiteType.SERVER if participant.type == ParticipantType.SERVER else participant.name


class CCBuilder(Builder):
    """Validate and provision all three CC deployment modes through one contract."""

    def __init__(self, cc_mgr_id="cc_manager"):
        self._cc_mgr_id = cc_mgr_id
        self.project_config = None
        self.plans = {}
        self.deployments = {}

    @staticmethod
    def _configure_client_gpu_capacity(participant, gpu_tee):
        """Keep FL resource scheduling consistent with the declared GPU TEE."""
        if participant.type != ParticipantType.CLIENT:
            return

        capacity = participant.get_prop(PropKey.CAPACITY)
        if capacity is None:
            if gpu_tee is GPUTEE.NVIDIA_CC:
                participant.set_prop(PropKey.CAPACITY, {PropKey.NUM_GPUS: 1})
            return
        if not isinstance(capacity, dict):
            raise ValueError("capacity must be a mapping")

        num_gpus = capacity.get(PropKey.NUM_GPUS)
        if num_gpus is not None and type(num_gpus) is not int:
            raise ValueError("capacity.num_of_gpus must be an integer")
        if gpu_tee is GPUTEE.NVIDIA_CC:
            if num_gpus is None:
                capacity = dict(capacity)
                capacity[PropKey.NUM_GPUS] = 1
                participant.set_prop(PropKey.CAPACITY, capacity)
            elif num_gpus < 1:
                raise ValueError("capacity.num_of_gpus must be positive when gpu_tee is nvidia_cc")
        elif num_gpus not in (None, 0):
            raise ValueError("capacity.num_of_gpus must be zero or omitted when gpu_tee is none")

    def initialize(self, project, ctx):
        # Builders may be reused by programmatic callers. Never carry plans,
        # adapters, or project credentials into a later provisioning run.
        self.project_config = None
        self.plans = {}
        self.deployments = {}
        if project.get_prop("cvm_vault") is not None:
            raise ValueError(
                "Legacy top-level cvm_vault is not supported; use participant cc_config with "
                "cc_deployment_mode: bare_metal_cvm"
            )
        selected = [
            participant for participant in project.get_all_participants() if participant.get_prop(PropKey.CC_CONFIG)
        ]
        if not selected:
            ctx[CtxKey.CC_DEPLOYMENT_PLANS] = {}
            return
        if any(client.name == SiteType.SERVER for client in project.get_clients()):
            raise ValueError("CC projects reserve the client name 'server' for the logical root-server identity")
        invalid = [
            participant.name
            for participant in selected
            if participant.type not in (ParticipantType.SERVER, ParticipantType.CLIENT)
        ]
        if invalid:
            raise ValueError(f"cc_config is supported only for server/client participants: {', '.join(invalid)}")
        packager = project.get_prop("packager", {})
        if not isinstance(packager, dict) or packager.get("path") != CC_PACKAGER_PATH:
            raise ValueError(f"Confidential participants require the common packager {CC_PACKAGER_PATH}")
        project_ref = project.get_prop(PropKey.CC_PROJECT_CONFIG)
        if not isinstance(project_ref, str) or not project_ref:
            raise ValueError("cc_project_config is required when any participant has cc_config")
        project_path = Path(resolve_cc_config(project, project_ref))
        self.project_config = load_project_config(project_path)
        self.project_config["_project_name"] = project.name

        normalized = {}
        for participant in selected:
            try:
                path = Path(resolve_cc_config(project, participant.get_prop(PropKey.CC_CONFIG)))
                participant_config = load_participant_config(path, self.project_config)
                self._configure_client_gpu_capacity(participant, participant_config["gpu_tee"])
                normalized[participant.name] = participant_config
            except Exception as exc:
                raise ValueError(f"Invalid CC configuration for {participant.name}: {exc}") from exc

        used_modes = {value["mode"] for value in normalized.values()}
        supplied_blocks = {
            CCDeploymentMode(name)
            for name in self.project_config.get("build_tools", {})
            if name in (CCDeploymentMode.BARE_METAL_CVM.value, CCDeploymentMode.COCO.value)
        }
        for mode in used_modes | supplied_blocks:
            deployment = self.deployments.setdefault(mode, DEPLOYMENTS[mode]())
            deployment.validate_project_config(self.project_config)

        trustee_services = {
            value["attestation_service"].name
            for value in normalized.values()
            if value["attestation_service"].service_type == "trustee"
        }
        if len(trustee_services) > 1:
            raise ValueError("Bare-metal CVM and CoCo participants must resolve the same named Trustee service")
        derived_constraints = {}
        for service_name in trustee_services:
            service = self.project_config["attestation_services"][service_name]
            constraints = service.get("workload_constraints")
            participants = [
                participant
                for participant in selected
                if normalized[participant.name]["attestation_service"].name == service_name
            ]
            expected_sites = {_site_name(participant) for participant in participants}
            if constraints is not None:
                if set(constraints) != expected_sites:
                    raise ValueError(
                        f"attestation_services.{service_name}.workload_constraints must contain exactly "
                        f"the protected sites: {', '.join(sorted(expected_sites))}"
                    )
            derived_constraints[service_name] = {
                _site_name(participant): {
                    **plain_data((constraints or {}).get(_site_name(participant), {})),
                    "gpu_required": normalized[participant.name]["gpu_tee"] is GPUTEE.NVIDIA_CC,
                }
                for participant in participants
            }
        releases = [
            value["mode_config"]["release_name"]
            for value in normalized.values()
            if value["mode"] is CCDeploymentMode.COCO
        ]
        if len(releases) != len(set(releases)):
            raise ValueError("Each CoCo participant requires a distinct coco.release_name")

        for participant in selected:
            value = normalized[participant.name]
            deployment = self.deployments[value["mode"]]
            plan = deployment.create_plan(participant, value, self.project_config)
            # Internal values are not serialized. They let every mode use the
            # same logical server identity and project audience.
            plan = replace(
                plan,
                internal=immutable_data(
                    {
                        **plan.internal,
                        "project_name": project.name,
                        "site_name": _site_name(participant),
                        **(
                            {"workload_constraints": derived_constraints[plan.attestation_service.name]}
                            if plan.attestation_service.service_type == "trustee"
                            else {}
                        ),
                    }
                ),
            )
            self.plans[participant.name] = plan
            participant.set_prop(PropKey.CC_ENABLED, True)
            participant.set_prop(PropKey.CC_CONFIG_DICT, value["raw"])
            participant.set_prop(PropKey.CC_DEPLOYMENT_PLAN, plan)
            participant.set_prop(PropKey.AUTHZ_SECTION_KEY, TemplateSectionKey.CC_AUTHZ)
            if hasattr(deployment, "bind"):
                deployment.bind(plan, self.project_config, project, ctx)

        ctx[CtxKey.CC_PROJECT_CONFIG] = self.project_config
        ctx[CtxKey.CC_DEPLOYMENT_PLANS] = self.plans
        ctx["cc_deployments"] = self.deployments

    @staticmethod
    def _write_component(ctx, participant, component):
        target = Path(ctx.get_local_dir(participant)) / f'{component["id"]}__p_resources.json'
        target.write_text(json.dumps({"components": [component]}, indent=2) + "\n")

    def _authorizer(self, plan, issuer):
        deployment = self.deployments[plan.mode]
        if not hasattr(deployment, "authorizer"):
            raise ValueError(f"{plan.mode.value} cannot supply a peer authorizer")
        return deployment.authorizer(plan, issuer=issuer)

    def _build_peer_matrix(self, project, ctx):
        protected = {
            _site_name(participant): self.plans[participant.name]
            for participant in project.get_all_participants()
            if participant.name in self.plans
        }
        verifier_plans = {}
        for plan in self.plans.values():
            spec = self._authorizer(plan, issuer=False)
            verifier_plans.setdefault(spec["id"], plan)
        required = {site: [self._authorizer(plan, issuer=False)["id"]] for site, plan in protected.items()}
        enabled = list(protected)
        frequencies = [plan.attestation_service.values["check_frequency_seconds"] for plan in self.plans.values()]

        for participant in [project.get_server(), *project.get_clients()]:
            if participant is None:
                continue
            own = self.plans.get(participant.name)
            components = {
                authorizer_id: self._authorizer(plan, issuer=False) for authorizer_id, plan in verifier_plans.items()
            }
            issuers = []
            if own:
                own_spec = self._authorizer(own, issuer=True)
                components[own_spec["id"]] = own_spec
                issuers.append(
                    {
                        CC_ISSUER_ID: own_spec["id"],
                        TOKEN_EXPIRATION: own_spec["token_expiration"],
                    }
                )
            for spec in components.values():
                component = {key: spec[key] for key in ("id", "path", "args")}
                self._write_component(ctx, participant, component)

            manager_args = {
                "cc_issuers_conf": issuers,
                "cc_verifier_ids": list(components),
                "verify_frequency": min(frequencies),
                "cc_enabled_sites": enabled,
                "required_site_verifier_ids": required,
                # Trustee proofs bind their signed subject to the mTLS peer.
                # The existing 2.9 Azure authorizers retain their MAA semantics.
                "require_site_binding": all(plan.mode is not CCDeploymentMode.AZURE_CC for plan in self.plans.values()),
            }
            trustee = next(
                (
                    plan.attestation_service.values
                    for plan in self.plans.values()
                    if plan.attestation_service.service_type == "trustee"
                ),
                None,
            )
            if trustee:
                manager_args.update(
                    resolve_token_timeouts(
                        registration_token_timeout=trustee["registration_token_timeout_seconds"],
                        refresh_token_timeout=trustee["refresh_token_timeout_seconds"],
                        get_token_request_timeout=trustee["get_token_request_timeout_seconds"],
                    )
                )
            self._write_component(
                ctx,
                participant,
                {"id": self._cc_mgr_id, "path": CC_MGR_PATH, "args": manager_args},
            )
            participant.set_prop(PropKey.AUTHZ_SECTION_KEY, TemplateSectionKey.CC_AUTHZ)

    @staticmethod
    def _extend_class_allow_list(participant, plan, ctx):
        if not plan.class_allow_list:
            return
        resources_file = Path(ctx.get_local_dir(participant)) / ProvFileName.RESOURCES_JSON_DEFAULT
        if not resources_file.is_file():
            raise RuntimeError("CCBuilder requires StaticFileBuilder before class_allow_list is extended")
        resources = json.loads(resources_file.read_text())
        configured = resources.get("class_allow_list", [])
        if not isinstance(configured, list):
            raise RuntimeError(f"{resources_file} contains a non-list class_allow_list")
        for class_path in plan.class_allow_list:
            if class_path not in configured:
                configured.append(class_path)
        resources["class_allow_list"] = configured
        temporary = resources_file.with_name(resources_file.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(resources, indent=2) + "\n")
        os.replace(temporary, resources_file)

    def build(self, project, ctx):
        if not self.plans:
            return
        for participant in project.get_all_participants():
            plan = self.plans.get(participant.name)
            if not plan:
                continue
            self.deployments[plan.mode].configure_startup_kit(plan, project, ctx)
            self._extend_class_allow_list(participant, plan, ctx)
        self._build_peer_matrix(project, ctx)
