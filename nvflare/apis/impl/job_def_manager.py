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

import datetime
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import Job, JobDataKey, JobMetaKey, job_from_meta
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.storage import StorageException, StorageSpec
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes
from nvflare.fuel.utils import fobs


class _JobFilter(ABC):
    @abstractmethod
    def filter_job(self, meta: dict) -> bool:
        pass


class _StatusFilter(_JobFilter):
    def __init__(self, status_to_check):
        self.result = []
        self.status_to_check = status_to_check

    def filter_job(self, meta: dict):
        if meta[JobMetaKey.STATUS] == self.status_to_check:
            self.result.append(job_from_meta(meta))
        return True


class _AllJobsFilter(_JobFilter):
    def __init__(self):
        self.result = []

    def filter_job(self, meta: dict):
        self.result.append(job_from_meta(meta))
        return True


class _ReviewerFilter(_JobFilter):
    def __init__(self, reviewer_name, fl_ctx: FLContext):
        """Not used yet, for use in future implementations."""
        self.result = []
        self.reviewer_name = reviewer_name

    def filter_job(self, meta: dict):
        approvals = meta.get(JobMetaKey.APPROVALS)
        if not approvals or self.reviewer_name not in approvals:
            self.result.append(job_from_meta(meta))
        return True


# TODO:: use try block around storage calls


class SimpleJobDefManager(JobDefManagerSpec):
    def __init__(self, uri_root: str = "jobs", job_store_id: str = "job_store"):
        super().__init__()
        self.uri_root = uri_root
        os.makedirs(uri_root, exist_ok=True)
        self.job_store_id = job_store_id

    def _get_job_store(self, fl_ctx):
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise TypeError(f"engine should be of type ServerEngineSpec, but got {type(engine)}")
        store = engine.get_component(self.job_store_id)
        if not isinstance(store, StorageSpec):
            raise TypeError(f"engine should have a job store component of type StorageSpec, but got {type(store)}")
        return store

    def job_uri(self, jid: str):
        return os.path.join(self.uri_root, jid)

    def create(self, meta: dict, uploaded_content: bytes, fl_ctx: FLContext) -> Dict[str, Any]:
        # validate meta to make sure it has:

        jid = str(uuid.uuid4())
        meta[JobMetaKey.JOB_ID.value] = jid
        meta[JobMetaKey.SUBMIT_TIME.value] = time.time()
        meta[JobMetaKey.SUBMIT_TIME_ISO.value] = (
            datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
        )
        meta[JobMetaKey.START_TIME.value] = ""
        meta[JobMetaKey.DURATION.value] = "N/A"
        meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value

        # write it to the store
        stored_data = {JobDataKey.JOB_DATA.value: uploaded_content, JobDataKey.WORKSPACE_DATA.value: None}
        store = self._get_job_store(fl_ctx)
        store.create_object(self.job_uri(jid), fobs.dumps(stored_data), meta, overwrite_existing=True)
        return meta

    def delete(self, jid: str, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        store.delete_object(self.job_uri(jid))

    def _validate_meta(self, meta):
        """Validate meta

        Args:
            meta: meta to validate

        Returns:

        """
        pass

    def _validate_uploaded_content(self, uploaded_content) -> bool:
        """Validate uploaded content for creating a run config. (THIS NEEDS TO HAPPEN BEFORE CONTENT IS PROVIDED NOW)

        Internally used by create and update.

        1. check all sites in deployment are in resources
        2. each site in deployment need to have resources (each site in resource need to be in deployment ???)
        """
        pass

    def get_job(self, jid: str, fl_ctx: FLContext) -> Optional[Job]:
        store = self._get_job_store(fl_ctx)
        try:
            job_meta = store.get_meta(self.job_uri(jid))
            return job_from_meta(job_meta)
        except StorageException:
            return None

    def set_results_uri(self, jid: str, result_uri: str, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        updated_meta = {JobMetaKey.RESULT_LOCATION.value: result_uri}
        store.update_meta(self.job_uri(jid), updated_meta, replace=False)
        return self.get_job(jid, fl_ctx)

    def get_app(self, job: Job, app_name: str, fl_ctx: FLContext) -> bytes:
        temp_dir = tempfile.mkdtemp()
        job_id_dir = self._load_job_data_from_store(job.job_id, temp_dir, fl_ctx)
        job_folder = os.path.join(job_id_dir, job.meta[JobMetaKey.JOB_FOLDER_NAME.value])
        fullpath_src = os.path.join(job_folder, app_name)
        result = zip_directory_to_bytes(fullpath_src, "")
        shutil.rmtree(temp_dir)
        return result

    def get_apps(self, job: Job, fl_ctx: FLContext) -> Dict[str, bytes]:
        temp_dir = tempfile.mkdtemp()
        job_id_dir = self._load_job_data_from_store(job.job_id, temp_dir, fl_ctx)
        job_folder = os.path.join(job_id_dir, job.meta[JobMetaKey.JOB_FOLDER_NAME.value])
        result_dict = {}
        for app in job.get_deployment():
            fullpath_src = os.path.join(job_folder, app)
            result_dict[app] = zip_directory_to_bytes(fullpath_src, "")
        shutil.rmtree(temp_dir)
        return result_dict

    def _load_job_data_from_store(self, jid: str, temp_dir: str, fl_ctx: FLContext):
        data_bytes = self.get_content(jid, fl_ctx)
        job_id_dir = os.path.join(temp_dir, jid)
        if os.path.exists(job_id_dir):
            shutil.rmtree(job_id_dir)
        os.mkdir(job_id_dir)
        unzip_all_from_bytes(data_bytes, job_id_dir)
        return job_id_dir

    def get_content(self, jid: str, fl_ctx: FLContext) -> Optional[bytes]:
        store = self._get_job_store(fl_ctx)
        try:
            stored_data = store.get_data(self.job_uri(jid))
        except StorageException:
            return None
        return fobs.loads(stored_data).get(JobDataKey.JOB_DATA.value)

    def get_job_data(self, jid: str, fl_ctx: FLContext) -> dict:
        store = self._get_job_store(fl_ctx)
        stored_data = store.get_data(self.job_uri(jid))
        return fobs.loads(stored_data)

    def set_status(self, jid: str, status: RunStatus, fl_ctx: FLContext):
        meta = {JobMetaKey.STATUS.value: status.value}
        store = self._get_job_store(fl_ctx)
        if status == RunStatus.RUNNING.value:
            meta[JobMetaKey.START_TIME.value] = str(datetime.datetime.now())
        elif status in [
            RunStatus.FINISHED_ABORTED.value,
            RunStatus.FINISHED_COMPLETED.value,
            RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
        ]:
            job_meta = store.get_meta(self.job_uri(jid))
            if job_meta[JobMetaKey.START_TIME.value]:
                start_time = datetime.datetime.strptime(job_meta.get(JobMetaKey.START_TIME), "%Y-%m-%d %H:%M:%S.%f")
                meta[JobMetaKey.DURATION.value] = str(datetime.datetime.now() - start_time)
        store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def update_meta(self, jid: str, meta, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def get_all_jobs(self, fl_ctx: FLContext) -> List[Job]:
        job_filter = _AllJobsFilter()
        self._scan(job_filter, fl_ctx)
        return job_filter.result

    def _scan(self, job_filter: _JobFilter, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        jid_paths = store.list_objects(self.uri_root)
        if not jid_paths:
            return

        for jid_path in jid_paths:
            jid = pathlib.PurePath(jid_path).name
            meta = store.get_meta(self.job_uri(jid))
            if meta:
                ok = job_filter.filter_job(meta)
                if not ok:
                    break

    def get_jobs_by_status(self, status, fl_ctx: FLContext) -> List[Job]:
        job_filter = _StatusFilter(status)
        self._scan(job_filter, fl_ctx)
        return job_filter.result

    def get_jobs_waiting_for_review(self, reviewer_name: str, fl_ctx: FLContext) -> List[Job]:
        job_filter = _ReviewerFilter(reviewer_name, fl_ctx)
        self._scan(job_filter, fl_ctx)
        return job_filter.result

    def set_approval(
        self, jid: str, reviewer_name: str, approved: bool, note: str, fl_ctx: FLContext
    ) -> Dict[str, Any]:
        meta = self.get_job(jid, fl_ctx).meta
        if meta:
            approvals = meta.get(JobMetaKey.APPROVALS)
            if not approvals:
                approvals = {}
                meta[JobMetaKey.APPROVALS.value] = approvals
            approvals[reviewer_name] = (approved, note)
            updated_meta = {JobMetaKey.APPROVALS.value: approvals}
            store = self._get_job_store(fl_ctx)
            store.update_meta(self.job_uri(jid), updated_meta, replace=False)
        return meta

    def save_workspace(self, jid: str, data: bytes, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        stored_data = store.get_data(self.job_uri(jid))
        job_data = fobs.loads(stored_data)
        job_data[JobDataKey.WORKSPACE_DATA.value] = data
        store.update_data(self.job_uri(jid), fobs.dumps(job_data))
