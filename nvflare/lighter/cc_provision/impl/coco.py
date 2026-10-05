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

"""CoCo implementation of the unified CC deployment interface."""

from nvflare.lighter.cc_provision.deployment import CCDeployment, CCDeploymentMode, CCDeploymentPlan, plain_data
from nvflare.lighter.cc_provision.impl.coco_release import _silence_coco_startup


class CoCoDeployment(CCDeployment):
    """CoCo implementation selected by the unified CC dispatcher."""

    mode = CCDeploymentMode.COCO

    def validate_project_config(self, project_config):
        if not project_config.get("build_tools", {}).get(self.mode.value):
            raise ValueError("coco requires build_tools.coco")
        if not project_config.get("container_registries"):
            raise ValueError("coco requires a named container registry")
        if not any(service.get("type") == "trustee" for service in project_config["attestation_services"].values()):
            raise ValueError("coco requires a Trustee attestation service")

    def create_plan(self, participant, cc_config, project_config):
        mode_config = dict(cc_config["mode_config"])
        registry = project_config["container_registries"][mode_config["registry"]]
        internal = {
            "registry": registry,
            "build_tools": project_config["build_tools"][self.mode.value],
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
            internal=internal,
        )

    @staticmethod
    def authorizer(plan, *, issuer):
        service = plan.attestation_service
        key = (service.config_path.parent / service.values["attestation_signing_public_key_file"]).resolve()
        retry = service.values.get("retry", {})
        args = {
            "trustee_public_key": key.read_text(),
            "audience": "nvflare-trustee:" + plan.internal["project_name"],
            "max_token_age_seconds": service.values["token_expiration_seconds"],
            "token_provider": "coco" if issuer else "verifier",
            **(
                {"site_name": plan.internal["site_name"], "token_url": service.values["attestation_token_endpoint"]}
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

    def configure_startup_kit(self, plan, project, ctx):
        participant = next(item for item in project.get_all_participants() if item.name == plan.participant_name)
        _silence_coco_startup(ctx, participant)

    def package(self, plan, private_kit, public_output, ctx):
        from nvflare.lighter.cc_provision.impl.cc_packager import package_coco_plan

        return package_coco_plan(plan, private_kit, public_output, ctx)
