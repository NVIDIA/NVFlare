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

from unittest import mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.app_common.storages.filesystem_storage import FilesystemStorage


def _submitter():
    return {"name": "submitter@nvidia.com", "org": "nvidia", "role": "lead"}


def _record(job_id="job-1", state="created"):
    return {
        "schema_version": 1,
        "state": state,
        "submit_token": "retry-1",
        "job_id": job_id,
        "study": "study-a",
        "submitter_name": "submitter@nvidia.com",
        "submitter_org": "nvidia",
        "submitter_role": "lead",
        "job_name": "hello",
        "job_folder_name": "hello",
        "job_content_hash": "sha256:abc",
        "submit_time": "2026-04-29T10:00:00-07:00",
    }


def test_submit_record_persists_across_manager_restart(tmp_path):
    storage = FilesystemStorage(root_dir=str(tmp_path / "store"), uri_root="/")
    fl_ctx = FLContext()

    with mock.patch.object(SimpleJobDefManager, "_get_job_store", return_value=storage):
        manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        manager.create_submit_record(_record(), fl_ctx)

        restarted = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        record = restarted.get_submit_record("study-a", _submitter(), "retry-1", fl_ctx)

    assert record["job_id"] == "job-1"
    assert record["job_content_hash"] == "sha256:abc"
    assert record["submit_token"] == "retry-1"


def test_creating_record_survives_restart_for_retry_recovery(tmp_path):
    storage = FilesystemStorage(root_dir=str(tmp_path / "store"), uri_root="/")
    fl_ctx = FLContext()

    with mock.patch.object(SimpleJobDefManager, "_get_job_store", return_value=storage):
        manager = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        manager.create_submit_record(_record(job_id="pre-generated-job", state="creating"), fl_ctx)

        restarted = SimpleJobDefManager(uri_root=str(tmp_path / "jobs"))
        record = restarted.get_submit_record("study-a", _submitter(), "retry-1", fl_ctx)

    assert record["state"] == "creating"
    assert record["job_id"] == "pre-generated-job"
