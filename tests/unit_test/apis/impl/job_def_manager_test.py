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

import os
import shutil
import tempfile
import unittest
from unittest import mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import JobDataKey, JobMetaKey
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage
from nvflare.fuel.hci.zip_utils import zip_directory_to_bytes
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator


class TestJobManager(unittest.TestCase):
    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uri_root = tempfile.mkdtemp()
        self.data_folder = os.path.join(dir_path, "../../data/jobs")
        self.job_manager = SimpleJobDefManager(uri_root=self.uri_root)
        self.fl_ctx = FLContext()

    def tearDown(self) -> None:
        shutil.rmtree(self.uri_root)

    def test_create_job(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            content = self.job_manager.get_content(meta.get(JobMetaKey.JOB_ID), self.fl_ctx)
            assert content == data

    def _create_job(self):
        data = zip_directory_to_bytes(self.data_folder, "valid_job")
        folder_name = "valid_job"
        job_validator = JobMetaValidator()
        valid, error, meta = job_validator.validate(folder_name, data)
        meta = self.job_manager.create(meta, data, self.fl_ctx)
        return data, meta

    def test_save_workspace(self):
        with mock.patch("nvflare.apis.impl.job_def_manager.SimpleJobDefManager._get_job_store") as mock_store:
            mock_store.return_value = FilesystemStorage()

            data, meta = self._create_job()
            job_id = meta.get(JobMetaKey.JOB_ID)
            self.job_manager.save_workspace(job_id, data, self.fl_ctx)
            result = self.job_manager.get_job_data(job_id, self.fl_ctx)
            assert result.get(JobDataKey.WORKSPACE_DATA.value) == data
