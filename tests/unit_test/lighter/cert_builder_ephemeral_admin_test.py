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

from nvflare.lighter.constants import ParticipantType
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.cert import CertBuilder


class _CertBuilderCtx(dict):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.state_dir = root_dir / "state"
        self.state_dir.mkdir()

    def get_state_dir(self):
        return str(self.state_dir)

    def get_kit_dir(self, participant):
        kit_dir = self.root_dir / participant.name / "startup"
        kit_dir.mkdir(parents=True, exist_ok=True)
        return str(kit_dir)


def test_cert_builder_omits_static_admin_cert_for_ephemeral_cert(tmp_path):
    server = Participant(type=ParticipantType.SERVER, name="server", org="nvidia")
    admin = Participant(
        type=ParticipantType.ADMIN,
        name="sso-admin-kit",
        org=None,
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
    project = Project(name="project", description="desc", participants=[server, admin])
    ctx = _CertBuilderCtx(tmp_path)
    builder = CertBuilder()

    builder.initialize(project, ctx)
    builder.build(project, ctx)

    admin_startup = tmp_path / "sso-admin-kit" / "startup"
    assert (admin_startup / "rootCA.pem").is_file()
    assert not (admin_startup / "client.key").exists()
    assert not (admin_startup / "client.crt").exists()
