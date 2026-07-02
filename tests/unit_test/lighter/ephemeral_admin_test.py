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

import pytest

from nvflare.lighter.constants import ParticipantType
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.ephemeral_admin import get_admin_ephemeral_cert_config


def _project(props=None, admin_props=None):
    server = Participant(type=ParticipantType.SERVER, name="server", org="nvidia")
    admin_props = admin_props or {}
    if "ephemeral_admin_cert" in admin_props:
        participant_props = admin_props
        org = None
    else:
        participant_props = {"role": "project_admin", **admin_props}
        org = "nvidia"
    admin = Participant(
        type=ParticipantType.ADMIN,
        name="admin@example.com",
        org=org,
        props=participant_props,
    )
    project = Project(name="project", description="desc", participants=[server, admin], props=props or {})
    return project, admin


def _ephemeral_cert_config(ca_url="https://step-ca.example.com", provisioner="nvflare-admin-oidc"):
    provider_config = {"ca_url": ca_url}
    if provisioner:
        provider_config["provisioner"] = provisioner
    return {
        "provider": "step_ca",
        "renewal_window": 60,
        "provider_config": provider_config,
    }


def test_admin_without_ephemeral_cert_config_has_no_ephemeral_cert_config():
    project, admin = _project()

    assert get_admin_ephemeral_cert_config(admin) is None


def test_per_admin_ephemeral_cert_config_supplies_admin_config():
    _project_obj, admin = _project(
        admin_props={
            "ephemeral_admin_cert": {
                "provider": "step_ca",
                "renewal_window": 60,
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                    "cert_ttl": "1h",
                },
            }
        },
    )

    assert get_admin_ephemeral_cert_config(admin)["provider_config"]["ca_url"] == "https://step-ca.example.com"
    assert get_admin_ephemeral_cert_config(admin)["provider_config"]["cert_ttl"] == "1h"
    assert "subject" not in get_admin_ephemeral_cert_config(admin)


def test_per_admin_ephemeral_cert_config_allows_admin_kit_name():
    admin = Participant(
        type=ParticipantType.ADMIN,
        name="sso-admin-kit",
        org=None,
        props={"ephemeral_admin_cert": _ephemeral_cert_config()},
    )

    assert admin.name == "sso-admin-kit"
    assert get_admin_ephemeral_cert_config(admin)["provider"] == "step_ca"


def test_per_admin_ephemeral_cert_config_is_used_directly():
    _project_obj, admin = _project(
        admin_props={
            "ephemeral_admin_cert": _ephemeral_cert_config(ca_url="https://admin-step-ca.example.com"),
        },
    )

    assert get_admin_ephemeral_cert_config(admin)["provider_config"]["ca_url"] == "https://admin-step-ca.example.com"


def test_ephemeral_cert_config_requires_provider():
    project, admin = _project(
        admin_props={"ephemeral_admin_cert": {"provider_config": {"ca_url": "https://step-ca.example.com"}}}
    )

    with pytest.raises(ValueError, match="provider"):
        get_admin_ephemeral_cert_config(admin)


def test_provider_specific_config_is_not_validated_by_generic_provisioning_helper():
    project, admin = _project(
        admin_props={
            "ephemeral_admin_cert": {
                "provider": "step_ca",
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                },
            }
        }
    )

    assert get_admin_ephemeral_cert_config(admin)["provider_config"]["ca_url"] == "https://step-ca.example.com"


def test_ephemeral_cert_config_rejects_unknown_provider_name_shape():
    _project_obj, admin = _project(
        admin_props={
            "ephemeral_admin_cert": {
                "provider": "step-ca",
                "provider_config": {},
            }
        }
    )

    with pytest.raises(ValueError, match="built-in provider name or module:function path"):
        get_admin_ephemeral_cert_config(admin)


def test_ephemeral_cert_config_accepts_custom_provider_path_without_importing_it():
    _project_obj, admin = _project(
        admin_props={
            "ephemeral_admin_cert": {
                "provider": "customer.cert_provider:obtain_certificate",
                "provider_config": {},
            }
        }
    )

    assert get_admin_ephemeral_cert_config(admin)["provider"] == "customer.cert_provider:obtain_certificate"
