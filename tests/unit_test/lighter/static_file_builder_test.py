# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json

import pytest

from nvflare.lighter.constants import CtxKey
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.static_file import StaticFileBuilder


class _FakeCtx:
    def __init__(self, project, root_dir):
        self._project = project
        self._root_dir = root_dir
        self.calls = []

    def get_project(self):
        return self._project

    def get(self, key):
        return {
            CtxKey.ADMIN_PORT: 8002,
            CtxKey.FED_LEARN_PORT: 8003,
        }.get(key)

    def get_kit_dir(self, entity):
        return self._root_dir / entity.name / "startup"

    def get_local_dir(self, entity):
        return self._root_dir / entity.name / "local"

    def get_ws_dir(self, entity):
        return self._root_dir / entity.name

    def build_from_template(
        self, dest_dir, temp_section, file_name, replacement=None, mode="t", exe=False, content_modify_cb=None, **kwargs
    ):
        self.calls.append((str(dest_dir), file_name))


class TestStaticFileBuilder:
    @pytest.mark.parametrize(
        "scheme",
        [("grpc"), ("http"), ("tcp")],
    )
    def test_scheme(self, scheme):
        builder = StaticFileBuilder(scheme=scheme)
        assert builder.scheme == scheme

    def test_build_server_emits_study_registry_when_studies_exist(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
                "studies": {
                    "study_a": {
                        "site_orgs": {"org": ["client1"]},
                        "admins": ["admin1@org.com"],
                    }
                },
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert registry_path.exists()
        assert json.loads(registry_path.read_text()) == {
            "format_version": "1.0",
            "studies": project.get_prop("studies"),
        }

    def test_build_server_does_not_emit_study_registry_without_studies(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert not registry_path.exists()

    def test_master_template_includes_server_predeployed_flwr_right(self):
        """master_template.yml's default_authz contains server-predeployed-flwr right."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        assert os.path.exists(template_path)

        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        assert "default_authz" in template
        authz = json.loads(template["default_authz"])

        assert "permissions" in authz
        org_admin = authz["permissions"]["org_admin"]
        assert "server-predeployed-flwr" in org_admin
        assert org_admin["server-predeployed-flwr"] == "none"

        lead = authz["permissions"]["lead"]
        assert "server-predeployed-flwr" in lead
        assert lead["server-predeployed-flwr"] == "none"

    def test_master_template_moves_user_config_runtime_workspace(self):
        """CC startup kits live in plaintext /user_config, so runtime artifacts must not default there."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        assert os.path.exists(template_path)

        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        sub_start = template["sub_start_sh"]
        stop_fl = template["stop_fl_sh"]
        copy_cmd = 'cp -a "$SOURCE_WORKSPACE" "$KIT_WORKSPACE"'
        verify_cmd = 'verify_startup_kits -f "$KIT_WORKSPACE" -c "$KIT_WORKSPACE/startup/rootCA.pem"'

        assert 'SOURCE_WORKSPACE" == "/user_config"' in sub_start
        assert 'SOURCE_WORKSPACE" == /user_config/*' in sub_start
        assert 'SOURCE_WORKSPACE" == "/user_config"' in stop_fl
        assert 'SOURCE_WORKSPACE" == /user_config/*' in stop_fl
        assert 'WORKSPACE="/vault/workspace"' in sub_start
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in sub_start
        assert 'WORKSPACE="/vault/workspace"' in stop_fl
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in stop_fl
        assert 'KIT_WORKSPACE="$WORKSPACE/kit"' in sub_start
        assert copy_cmd in sub_start
        assert 'rm -rf "$KIT_WORKSPACE"' in sub_start
        assert 'rm -rf "$WORKSPACE"' not in sub_start
        assert 'ln -s "$KIT_WORKSPACE/startup" "$WORKSPACE/startup"' in sub_start
        assert 'ln -s "$KIT_WORKSPACE/local" "$WORKSPACE/local"' in sub_start
        assert '-m "$WORKSPACE"' in sub_start
        assert verify_cmd in sub_start
        assert sub_start.index(copy_cmd) < sub_start.index(verify_cmd)
        assert sub_start.index(verify_cmd) < sub_start.index('mkdir -p "$WORKSPACE/transfer"')
        assert 'touch "$WORKSPACE/shutdown.fl"' in stop_fl

    def test_master_template_includes_fedopt_default_optimizer_allow_list(self):
        """FedOpt default optimizer configs must pass protected non-BYOC component authorization."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        assert os.path.exists(template_path)

        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        expected_paths = {
            "tensorflow.keras.optimizers.SGD",
            "tensorflow.keras.optimizers.schedules.CosineDecay",
            "torch.optim.SGD",
            "torch.optim.lr_scheduler.CosineAnnealingLR",
        }
        for resource_key in ("local_client_resources", "local_server_resources"):
            resource_template = template[resource_key]
            for expected_path in expected_paths:
                assert f'"{expected_path}"' in resource_template
