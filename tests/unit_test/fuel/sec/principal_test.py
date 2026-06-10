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

from nvflare.fuel.sec.principal import AUTH_METHOD_CERT, AUTH_METHOD_OIDC, Principal


def test_legacy_admin_principal_uses_current_policy_fields():
    principal = Principal.from_legacy_admin(
        username="admin@nvidia.com",
        org="nvidia",
        role="project_admin",
    )

    assert principal.subject == "admin@nvidia.com"
    assert principal.policy_name() == "admin@nvidia.com"
    assert principal.policy_org() == "nvidia"
    assert principal.policy_role() == "project_admin"
    assert principal.raw_roles == ("project_admin",)
    assert principal.auth_method == AUTH_METHOD_CERT


def test_principal_round_trip_preserves_oidc_metadata():
    principal = Principal(
        subject="keycloak-subject",
        username="admin@nvidia.com",
        email="admin@nvidia.com",
        org="nvidia",
        raw_roles=["flare_project_admin", "flare_member"],
        groups=["/flare/project-admins"],
        issuer="https://keycloak.example.com/realms/nvflare",
        token_id="token-id",
        auth_time=123.0,
        token_exp=456.0,
        effective_role="project_admin",
        auth_method=AUTH_METHOD_OIDC,
    )

    restored = Principal.from_dict(principal.to_dict())

    assert restored == principal
    assert restored.raw_roles == ("flare_project_admin", "flare_member")
    assert restored.groups == ("/flare/project-admins",)
    assert restored.token_exp == 456.0


def test_submitter_dict_excludes_raw_idp_claims_and_token_metadata():
    principal = Principal(
        subject="keycloak-subject",
        username="admin@nvidia.com",
        email="admin@nvidia.com",
        org="nvidia",
        raw_roles=["flare_project_admin"],
        groups=["/flare/project-admins"],
        issuer="https://keycloak.example.com/realms/nvflare",
        token_id="token-id",
        auth_time=123.0,
        token_exp=456.0,
        effective_role="project_admin",
        auth_method=AUTH_METHOD_OIDC,
    )

    data = principal.to_submitter_dict()

    assert data == {
        "subject": "keycloak-subject",
        "username": "admin@nvidia.com",
        "email": "admin@nvidia.com",
        "org": "nvidia",
        "effective_role": "project_admin",
        "auth_method": "oidc",
        "issuer": "https://keycloak.example.com/realms/nvflare",
    }


def test_submitter_dict_omits_empty_fields():
    principal = Principal(subject="keycloak-subject")

    assert principal.to_submitter_dict() == {"subject": "keycloak-subject", "auth_method": "unknown"}


def test_principal_coerces_auth_time_to_float():
    principal = Principal.from_dict({"subject": "sub", "auth_time": "123.5"})

    assert principal.auth_time == 123.5
