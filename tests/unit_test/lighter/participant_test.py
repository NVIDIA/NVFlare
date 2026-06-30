# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.lighter.entity import Participant


class TestParticipant:
    @pytest.mark.parametrize(
        "type,invalid_name",
        [
            ("server", "server_"),
            ("server", "server@"),
            ("server", "-server"),
            ("client", "client!"),
            ("client", "client@"),
            ("admin", "admin"),
            ("admin", "admin@example_1.com"),
        ],
    )
    def test_invalid_name(self, type, invalid_name):
        with pytest.raises(ValueError):
            _ = Participant(name=invalid_name, org="org", type=type)

    def test_ephemeral_admin_allows_kit_name(self):
        participant = Participant(
            name="sso-admin-kit",
            org=None,
            type="admin",
            props={
                "ephemeral_admin_cert": {
                    "provider": "step_ca",
                    "provider_config": {
                        "ca_url": "https://step-ca.example.com",
                        "provisioner": "nvflare-admin-oidc",
                    },
                },
            },
        )

        assert participant.name == "sso-admin-kit"
        assert not participant.org

    @pytest.mark.parametrize(
        "props,org,match",
        [
            ({"role": "project_admin", "ephemeral_admin_cert": {"provider": "step_ca"}}, None, "must not define role"),
            ({"ephemeral_admin_cert": {"provider": "step_ca"}}, "org", "must not define org"),
        ],
    )
    def test_ephemeral_admin_rejects_project_time_identity(self, props, org, match):
        with pytest.raises(ValueError, match=match):
            _ = Participant(name="sso-admin-kit", org=org, type="admin", props=props)

    def test_static_admin_rejects_kit_name(self):
        with pytest.raises(ValueError):
            _ = Participant(name="sso-admin-kit", org="org", type="admin", props={"role": "project_admin"})

    @pytest.mark.parametrize(
        "invalid_org",
        [("org-"), ("org@"), ("org!"), ("org~")],
    )
    def test_invalid_org(self, invalid_org):
        with pytest.raises(ValueError):
            _ = Participant(name="server", type="server", org=invalid_org)

    @pytest.mark.parametrize(
        "invalid_type",
        [("type@"), ("type!"), ("type~")],
    )
    def test_invalid_type(self, invalid_type):
        with pytest.raises(ValueError):
            _ = Participant(name="server", type=invalid_type, org="org")
