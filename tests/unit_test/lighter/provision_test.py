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

import pytest

from nvflare.lighter.provision import add_extra_clients, add_extra_users, prepare_project, provision_for_edge


class TestProvision:
    def test_prepare_project(self):
        for api_version in [2, 5]:
            project_config = {"api_version": api_version}
            with pytest.raises(ValueError, match=f"API version expected 3 or 4 but found {api_version}"):
                prepare_project(project_dict=project_config)

        project_config = {
            "api_version": 3,
            "name": "mytest",
            "description": "test",
            "participants": [
                {"type": "server", "name": "server1", "org": "org"},
                {"type": "server", "name": "server2", "org": "org"},
                {"type": "server", "name": "server3", "org": "org"},
            ],
        }

        with pytest.raises(ValueError, match=".* server already exists"):
            prepare_project(project_dict=project_config)

    def _base_project(self, *, api_version=4, studies=None):
        project_config = {
            "api_version": api_version,
            "name": "mytest",
            "description": "test",
            "participants": [
                {"type": "server", "name": "server1", "org": "org"},
                {"type": "client", "name": "client1", "org": "org"},
                {"type": "admin", "name": "admin1@org.com", "org": "org", "role": "project_admin"},
            ],
        }
        if studies is not None:
            project_config["studies"] = studies
        return project_config

    def test_prepare_project_accepts_api_version_4(self):
        project = prepare_project(project_dict=self._base_project())

        assert project.name == "mytest"
        assert project.get_server().name == "server1"
        assert [p.name for p in project.get_clients()] == ["client1"]
        assert [p.name for p in project.get_admins()] == ["admin1@org.com"]

    def test_prepare_project_requires_api_version_4_for_studies(self):
        project_config = self._base_project(api_version=3, studies={"study-a": {"sites": ["client1"], "admins": {}}})

        with pytest.raises(ValueError, match="studies: requires api_version: 4"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_rejects_reserved_default_study_name(self):
        project_config = self._base_project(
            studies={"default": {"sites": ["client1"], "admins": {"admin1@org.com": "project_admin"}}}
        )

        with pytest.raises(ValueError, match="study name 'default' is reserved"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_rejects_invalid_study_name(self):
        project_config = self._base_project(
            studies={"Study_A": {"sites": ["client1"], "admins": {"admin1@org.com": "project_admin"}}}
        )

        with pytest.raises(ValueError, match="invalid study name 'Study_A'"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_accepts_underscore_in_study_name(self):
        project_config = self._base_project(
            studies={"study_a": {"sites": ["client1"], "admins": {"admin1@org.com": "project_admin"}}}
        )

        project = prepare_project(project_dict=project_config)

        assert "study_a" in project.get_prop("studies")

    def test_prepare_project_accepts_null_study_definition_as_empty_mapping(self):
        project_config = self._base_project(studies={"study-a": None})
        project = prepare_project(project_dict=project_config)

        assert project.name == "mytest"
        assert project.get_prop("studies")["study-a"] == {}

    def test_prepare_project_rejects_non_mapping_study_definition(self):
        project_config = self._base_project(studies={"study-a": 123})

        with pytest.raises(ValueError, match="study 'study-a' must be a mapping"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_rejects_unknown_client_reference(self):
        project_config = self._base_project(
            studies={"study-a": {"sites": ["client2"], "admins": {"admin1@org.com": "project_admin"}}}
        )

        with pytest.raises(ValueError, match="study 'study-a' references unknown client 'client2'"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_rejects_unknown_admin_reference(self):
        project_config = self._base_project(
            studies={"study-a": {"sites": ["client1"], "admins": {"admin2@org.com": "project_admin"}}}
        )

        with pytest.raises(ValueError, match="study 'study-a' references unknown admin 'admin2@org.com'"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_rejects_invalid_role_in_study_mapping(self):
        project_config = self._base_project(
            studies={"study-a": {"sites": ["client1"], "admins": {"admin1@org.com": "captain"}}}
        )

        with pytest.raises(ValueError, match="study 'study-a' assigns unknown role 'captain' to 'admin1@org.com'"):
            prepare_project(project_dict=project_config)

    def test_prepare_project_requires_project_name(self):
        project_config = self._base_project()
        project_config["name"] = None

        with pytest.raises(ValueError, match="missing project name"):
            prepare_project(project_dict=project_config)

    def test_add_extra_client_exits_when_output_error_is_mocked(self, tmp_path):
        bad_yaml = tmp_path / "bad_client.yml"
        bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
        participant_defs = []

        with pytest.raises(SystemExit) as exc_info:
            add_extra_clients(str(bad_yaml), participant_defs)

        assert exc_info.value.code == 4
        assert participant_defs == []

    def test_add_extra_user_exits_when_output_error_is_mocked(self, tmp_path):
        bad_yaml = tmp_path / "bad_user.yml"
        bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
        participant_defs = []

        with pytest.raises(SystemExit) as exc_info:
            add_extra_users(str(bad_yaml), participant_defs)

        assert exc_info.value.code == 4
        assert participant_defs == []

    def test_add_extra_client_empty_yaml_exits(self, tmp_path):
        bad_yaml = tmp_path / "bad_client_empty.yml"
        bad_yaml.write_text("", encoding="utf-8")
        participant_defs = []

        with pytest.raises(SystemExit) as exc_info:
            add_extra_clients(str(bad_yaml), participant_defs)

        assert exc_info.value.code == 4
        assert participant_defs == []

    def test_add_extra_user_empty_yaml_exits(self, tmp_path):
        bad_yaml = tmp_path / "bad_user_empty.yml"
        bad_yaml.write_text("", encoding="utf-8")
        participant_defs = []

        with pytest.raises(SystemExit) as exc_info:
            add_extra_users(str(bad_yaml), participant_defs)

        assert exc_info.value.code == 4
        assert participant_defs == []

    def test_provision_for_edge_requires_participants(self):
        with pytest.raises(ValueError, match="missing 'participants' in project config"):
            provision_for_edge({}, {"name": "proj"})

    def test_prepare_project_requires_participants(self):
        project_config = {
            "api_version": 4,
            "name": "mytest",
            "description": "test",
        }

        with pytest.raises(ValueError, match="missing 'participants' in project config"):
            prepare_project(project_dict=project_config)
