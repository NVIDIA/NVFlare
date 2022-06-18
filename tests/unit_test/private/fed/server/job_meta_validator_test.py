# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import io
import os
import sys
import zipfile
from zipfile import ZipFile

import pytest

from nvflare.fuel.hci.zip_utils import convert_legacy_zip, get_all_file_paths, normpath_for_zip, split_path
from nvflare.private.fed.server.job_meta_validator import META, JobMetaValidator


def _zip_directory_with_meta(root_dir: str, folder_name: str, meta: str, writer: io.BytesIO):
    dir_name = normpath_for_zip(os.path.join(root_dir, folder_name))
    assert os.path.exists(dir_name), 'directory "{}" does not exist'.format(dir_name)
    assert os.path.isdir(dir_name), '"{}" is not a valid directory'.format(dir_name)

    file_paths = get_all_file_paths(dir_name)
    if folder_name:
        prefix_len = len(split_path(dir_name)[0]) + 1
    else:
        prefix_len = len(dir_name) + 1

    with ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # writing each file one by one
        for full_path in file_paths:
            rel_path = full_path[prefix_len:]
            if len(meta) > 0 and rel_path.endswith(META):
                z.writestr(rel_path, meta)
            else:
                z.write(full_path, arcname=rel_path)


def _zip_job_with_meta(folder_name: str, meta: str) -> bytes:
    job_path = os.path.join(os.path.dirname(__file__), "../../../data/jobs")
    bio = io.BytesIO()
    _zip_directory_with_meta(job_path, folder_name, meta, bio)
    zip_data = bio.getvalue()
    return convert_legacy_zip(zip_data)


META_WITH_VALID_DEPLOY_MAP = [
    pytest.param({"deploy_map": {"app1": ["@ALL"]}}, id="all"),
    pytest.param({"deploy_map": {"app1": ["@ALL"], "app2": []}}, id="all_idle"),
    pytest.param({"deploy_map": {"app1": ["server", "site-1", "site-2"], "app2": []}}, id="idle_app"),
    pytest.param({"deploy_map": {"app1": ["server", "site-1", "site-2"]}}, id="one_app"),
    pytest.param({"deploy_map": {"app1": ["server", "site-1"], "app2": ["site-2"]}}, id="two_app"),
]


META_WITH_INVALID_DEPLOY_MAP = [
    pytest.param({"deploy_map": {"app1": ["@ALL", "server"]}}, id="all_other"),
    pytest.param({"deploy_map": {"app1": ["@ALL"], "app2": ["@all"]}}, id="dup_all"),
    pytest.param({"deploy_map": {"app1": ["server", "site-1", "site-2"], "app2": ["site-2"]}}, id="dup_client"),
    pytest.param({"deploy_map": {"app1": ["server", "site-1"], "app2": ["server", "site-2"]}}, id="dup_server"),
    pytest.param({"deploy_map": {}}, id="empty_deploy_map"),
    pytest.param({"deploy_map": {"app1": []}}, id="no_deployment"),
    pytest.param({"deploy_map": {"app1": [], "app2": []}}, id="no_deployment_two_apps"),
]


VALID_JOBS = [
    pytest.param("valid_job", id="valid_job"),
    pytest.param("valid_job_deployment_all_idle", id="valid_job_deployment_all_idle"),
    pytest.param("valid_app_as_job", id="valid_app_wo_meta"),
]


INVALID_JOBS = [
    pytest.param("duplicate_clients", id="duplicate_clients"),
    pytest.param("duplicate_server", id="duplicate_server"),
    pytest.param("invalid_resource_spec_data_type", id="invalid_resource_spec_data_type"),
    pytest.param("mandatory_not_met", id="mandatory_not_met"),
    pytest.param("missing_app", id="missing_app"),
    pytest.param("missing_client_config", id="missing_client_config"),
    pytest.param("missing_server_config", id="missing_server_config"),
    pytest.param("no_deployment", id="no_deployment"),
    pytest.param("not_enough_clients", id="not_enough_clients"),
]


class TestJobMetaValidator:
    @classmethod
    def setup_class(cls):
        cls.validator = JobMetaValidator()

    @pytest.mark.parametrize("meta", META_WITH_VALID_DEPLOY_MAP)
    def test_validate_valid_deploy_map(self, meta):
        site_list = JobMetaValidator._validate_deploy_map("unit_test", meta)
        assert site_list

    @pytest.mark.parametrize("meta", META_WITH_INVALID_DEPLOY_MAP)
    def test_validate_invalid_deploy_map(self, meta):
        with pytest.raises(ValueError):
            JobMetaValidator._validate_deploy_map("unit_test", meta)

    @pytest.mark.parametrize("job_name", VALID_JOBS)
    def test_validate_valid_jobs(self, job_name):
        self._assert_valid(job_name)

    @pytest.mark.parametrize("job_name", INVALID_JOBS)
    def test_validate_invalid_jobs(self, job_name):
        self._assert_invalid(job_name)

    @pytest.mark.parametrize(
        "min_clients",
        [
            pytest.param(-1, id="negative value"),
            pytest.param(0, id="zero value"),
            pytest.param(sys.maxsize + 1, id="sys.maxsize + 1 value"),
        ],
    )
    def test_invalid_min_clients_value_range(self, min_clients):
        job_name = "min_clients_value_range"
        meta = f"""
        {{
            "name": "sag", 
            "resource_spec": {{ "site-a": {{"gpu": 1}}, "site-b": {{"gpu": 1}} }}, 
            "deploy_map": {{"min_clients_value_range": ["server","site-a", "site-b"]}},
            "min_clients" : {min_clients}
        }}
        """
        self._assert_invalid(job_name, meta)

    def test_deploy_map_config_non_exists_app(self):
        job_name = "valid_job"
        # valid_job folder contains sag app, not hello-pt app
        meta = """
            { 
              "resource_spec": { "site-a": {"gpu": 1}, "site-b": {"gpu": 1}}, 
              "deploy_map": {"hello-pt": ["server","site-a", "site-b"]}
            }
        """
        self._assert_invalid(job_name, meta)

    def test_meta_missing_job_folder_name(self):
        job_name = "valid_job"
        meta = """
            { 
              "resource_spec": { "site-a": {"gpu": 1}, "site-b": {"gpu": 1}}, 
              "deploy_map": {"sag": ["server","site-a", "site-b"]}
            }
        """
        self._assert_valid(job_name, meta)

    def _assert_valid(self, job_name: str, meta: str = ""):
        data = _zip_job_with_meta(job_name, meta)
        valid, error, meta = self.validator.validate(job_name, data)
        assert valid
        assert error == ""

    def _assert_invalid(self, job_name: str, meta: str = ""):
        data = _zip_job_with_meta(job_name, meta)
        valid, error, meta = self.validator.validate(job_name, data)
        assert not valid
        assert error
