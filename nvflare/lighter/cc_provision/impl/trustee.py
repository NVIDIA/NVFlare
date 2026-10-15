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

"""Shared Trustee authorizer construction for CVM and CoCo deployments."""

from pathlib import Path

from nvflare.lighter.cc_provision.deployment import plain_data


def trustee_authorizer(plan, *, issuer, token_provider, issuer_args=None):
    """Build the common Trustee component, adding only provider-specific issuer inputs."""
    service = plan.attestation_service
    retry = service.values.get("retry", {})
    args = {
        "trustee_public_key": Path(service.values["attestation_signing_public_key_file"]).read_text(),
        "audience": "nvflare-trustee:" + plan.internal["project_name"],
        "max_token_age_seconds": service.values["token_expiration_seconds"],
        "token_provider": token_provider if issuer else "verifier",
        **({"site_name": plan.internal["site_name"], **(issuer_args or {})} if issuer else {}),
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
