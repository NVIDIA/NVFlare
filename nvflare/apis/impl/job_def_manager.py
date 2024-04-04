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

import datetime
import os
import pathlib
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import Job, JobDataKey, JobMetaKey, job_from_meta, new_job_id
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.storage import WORKSPACE, StorageException, StorageSpec
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes

_OBJ_TAG_SCHEDULED = "scheduled"


class JobInfo:
    def __init__(self, meta: dict, job_id: str, uri: str):
        self.meta = meta
        self.job_id = job_id
        self.uri = uri


class _JobFilter(ABC):
    @abstractmethod
    def filter_job(self, info: JobInfo) -> bool:
        pass


class _StatusFilter(_JobFilter):
    def __init__(self, status_to_check):
        self.result = []
        if not isinstance(status_to_check, list):
            # turning to list
            status_to_check = [status_to_check]
        self.status_to_check = status_to_check

    def filter_job(self, info: JobInfo):
        status = info.meta.get(JobMetaKey.STATUS.value)
        if status in self.status_to_check:
            self.result.append(job_from_meta(info.meta))
        return True


class _AllJobsFilter(_JobFilter):
    def __init__(self):
        self.result = []

    def filter_job(self, info: JobInfo):
        self.result.append(job_from_meta(info.meta))
        return True


class _ReviewerFilter(_JobFilter):
    def __init__(self, reviewer_name):
        """Not used yet, for use in future implementations."""
        self.result = []
        self.reviewer_name = reviewer_name

    def filter_job(self, info: JobInfo):
        approvals = info.meta.get(JobMetaKey.APPROVALS)
        if not approvals or self.reviewer_name not in approvals:
            self.result.append(job_from_meta(info.meta))
        return True


class _ScheduleJobFilter(_JobFilter):

    """
    This filter is optimized for selecting jobs to schedule since it is used so frequently (every 1 sec).
    """

    def __init__(self, store):
        self.store = store
        self.result = []

    def filter_job(self, info: JobInfo):
        status = info.meta.get(JobMetaKey.STATUS.value)
        if status == RunStatus.SUBMITTED.value:
            self.result.append(job_from_meta(info.meta))
        elif status:
            # skip this job in all future calls (so the meta file of this job won't be read)
            self.store.tag_object(uri=info.uri, tag=_OBJ_TAG_SCHEDULED)
        return True


class SimpleJobDefManager(JobDefManagerSpec):
    def __init__(self, uri_root: str = "jobs", job_store_id: str = "job_store"):
        super().__init__()
        self.uri_root = uri_root
        os.makedirs(uri_root, exist_ok=True)
        self.job_store_id = job_store_id

    def _get_job_store(self, fl_ctx):
        engine = fl_ctx.get_engine()

        if not (isinstance(engine, ServerEngineSpec) or isinstance(engine, ClientEngineSpec)):
            raise TypeError(f"engine should be of type ServerEngineSpec or ClientEngineSpec, but got {type(engine)}")
        store = engine.get_component(self.job_store_id)
        if not isinstance(store, StorageSpec):
            raise TypeError(f"engine should have a job store component of type StorageSpec, but got {type(store)}")
        return store

    def job_uri(self, jid: str):
        return os.path.join(self.uri_root, jid)

    def create(self, meta: dict, uploaded_content: Union[str, bytes], fl_ctx: FLContext) -> Dict[str, Any]:
        # validate meta to make sure it has:
        jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not jid:
            jid = new_job_id()
            meta[JobMetaKey.JOB_ID.value] = jid

        now = time.time()
        meta[JobMetaKey.SUBMIT_TIME.value] = now
        meta[JobMetaKey.SUBMIT_TIME_ISO.value] = datetime.datetime.fromtimestamp(now).astimezone().isoformat()
        meta[JobMetaKey.START_TIME.value] = ""
        meta[JobMetaKey.DURATION.value] = "N/A"
        meta[JobMetaKey.DATA_STORAGE_FORMAT.value] = 2
        meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value

        # write it to the store
        store = self._get_job_store(fl_ctx)
        store.create_object(self.job_uri(jid), uploaded_content, meta, overwrite_existing=True)
        return meta

    def clone(self, from_jid: str, meta: dict, fl_ctx: FLContext) -> Dict[str, Any]:
        jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not jid:
            jid = new_job_id()
            meta[JobMetaKey.JOB_ID.value] = jid

        now = time.time()
        meta[JobMetaKey.SUBMIT_TIME.value] = now
        meta[JobMetaKey.SUBMIT_TIME_ISO.value] = datetime.datetime.fromtimestamp(now).astimezone().isoformat()
        meta[JobMetaKey.START_TIME.value] = ""
        meta[JobMetaKey.DURATION.value] = "N/A"
        meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value

        # write it to the store
        store = self._get_job_store(fl_ctx)
        store.clone_object(
            from_uri=self.job_uri(from_jid), to_uri=self.job_uri(jid), meta=meta, overwrite_existing=True
        )
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
        job_id_dir = self._load_job_data_from_store(job, temp_dir, fl_ctx)
        job_folder = os.path.join(job_id_dir, job.meta[JobMetaKey.JOB_FOLDER_NAME.value])
        fullpath_src = os.path.join(job_folder, app_name)
        result = zip_directory_to_bytes(fullpath_src, "")
        shutil.rmtree(temp_dir)
        return result

    def _load_job_data_from_store(self, job: Job, temp_dir: str, fl_ctx: FLContext):
        data_bytes = self.get_content(job.meta, fl_ctx)
        job_id_dir = os.path.join(temp_dir, job.job_id)
        if os.path.exists(job_id_dir):
            shutil.rmtree(job_id_dir)
        os.mkdir(job_id_dir)
        unzip_all_from_bytes(data_bytes, job_id_dir)
        return job_id_dir

    def get_content(self, meta: dict, fl_ctx: FLContext) -> Optional[bytes]:
        store = self._get_job_store(fl_ctx)
        jid = meta.get(JobMetaKey.JOB_ID.value)
        if not jid:
            raise RuntimeError("no Job ID in meta")

        try:
            stored_data = store.get_data(self.job_uri(jid))
            storage_format = meta.get(JobMetaKey.DATA_STORAGE_FORMAT.value)
            if storage_format:
                # new format
                return stored_data
            else:
                # old format
                return fobs.loads(stored_data).get(JobDataKey.JOB_DATA.value)
        except StorageException:
            return None

    def set_status(self, jid: str, status: RunStatus, fl_ctx: FLContext):
        meta = {JobMetaKey.STATUS.value: status.value}
        store = self._get_job_store(fl_ctx)
        if status == RunStatus.RUNNING.value:
            meta[JobMetaKey.START_TIME.value] = str(datetime.datetime.now())
        elif status in [
            RunStatus.FINISHED_ABORTED.value,
            RunStatus.FINISHED_COMPLETED.value,
            RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
            RunStatus.FINISHED_CANT_SCHEDULE.value,
        ]:
            job_meta = store.get_meta(self.job_uri(jid))
            if job_meta[JobMetaKey.START_TIME.value]:
                start_time = datetime.datetime.strptime(
                    job_meta.get(JobMetaKey.START_TIME.value), "%Y-%m-%d %H:%M:%S.%f"
                )
                meta[JobMetaKey.DURATION.value] = str(datetime.datetime.now() - start_time)
        store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def update_meta(self, jid: str, meta, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def refresh_meta(self, job: Job, meta_keys: list, fl_ctx: FLContext):
        """Refresh meta of the job as specified in the meta keys
        Save the values of the specified keys into job store

        Args:
            job: job object
            meta_keys: meta keys need to updated
            fl_ctx: FLContext

        """
        if meta_keys:
            meta = {}
            for k in meta_keys:
                if k in job.meta:
                    meta[k] = job.meta[k]
        else:
            meta = job.meta
        if meta:
            self.update_meta(job.job_id, meta, fl_ctx)

    def get_all_jobs(self, fl_ctx: FLContext) -> List[Job]:
        job_filter = _AllJobsFilter()
        self._scan(job_filter, fl_ctx)
        return job_filter.result

    def get_jobs_to_schedule(self, fl_ctx: FLContext) -> List[Job]:
        job_filter = _ScheduleJobFilter(self._get_job_store(fl_ctx))
        self._scan(job_filter, fl_ctx, skip_tag=_OBJ_TAG_SCHEDULED)
        return job_filter.result

    def _scan(self, job_filter: _JobFilter, fl_ctx: FLContext, skip_tag=None):
        store = self._get_job_store(fl_ctx)
        obj_uris = store.list_objects(self.uri_root, without_tag=skip_tag)
        self.log_debug(fl_ctx, f"objects to scan: {len(obj_uris)}")
        if not obj_uris:
            return

        for uri in obj_uris:
            jid = pathlib.PurePath(uri).name
            job_uri = self.job_uri(jid)
            meta = store.get_meta(job_uri)
            if meta:
                ok = job_filter.filter_job(JobInfo(meta, jid, job_uri))
                if not ok:
                    break

    def get_jobs_by_status(self, status: Union[RunStatus, List[RunStatus]], fl_ctx: FLContext) -> List[Job]:
        """Get jobs that are in the specified status

        Args:
            status: a single status value or a list of status values
            fl_ctx: the FL context

        Returns: list of jobs that are in specified status

        """
        job_filter = _StatusFilter(status)
        self._scan(job_filter, fl_ctx)
        return job_filter.result

    def get_jobs_waiting_for_review(self, reviewer_name: str, fl_ctx: FLContext) -> List[Job]:
        job_filter = _ReviewerFilter(reviewer_name)
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

    def save_workspace(self, jid: str, data: Union[bytes, str], fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        store.update_object(self.job_uri(jid), data, WORKSPACE)

    def get_storage_component(self, jid: str, component: str, fl_ctx: FLContext):
        store = self._get_job_store(fl_ctx)
        return store.get_data(self.job_uri(jid), component)

    def get_storage_for_download(
        self, jid: str, download_dir: str, component: str, download_file: str, fl_ctx: FLContext
    ):
        store = self._get_job_store(fl_ctx)
        os.makedirs(os.path.join(download_dir, jid), exist_ok=True)
        destination_file = os.path.join(download_dir, jid, download_file)
        store.get_data_for_download(self.job_uri(jid), component, destination_file)
