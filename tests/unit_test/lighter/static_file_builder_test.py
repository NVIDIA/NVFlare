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
                        "sites": ["client1"],
                        "admins": {"admin1@org.com": "project_admin"},
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
