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
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from nvflare.apis.job_def import Job, JobMetaKey, job_from_meta
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec, RunStatus
from nvflare.apis.storage import StorageSpec
from nvflare.apis.study_manager_spec import StudyManagerSpec
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes


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
    def __init__(self, reviewer_name, study_manager: StudyManagerSpec):
        """Not used yet, for use in future implementations."""
        self.result = []
        self.reviewer_name = reviewer_name
        self.study_manager = study_manager

    def filter_job(self, meta: dict):
        study = self.study_manager.get_study(meta[JobMetaKey.STUDY_NAME])
        if study and study.reviewers and self.reviewer_name in study.reviewers:
            # this job requires review from this reviewer
            # has it been reviewed already?
            approvals = meta.get(JobMetaKey.APPROVALS)
            if approvals and self.reviewer_name in approvals:
                # already reviewed
                return True
            else:
                self.result.append(job_from_meta(meta))
        return True


class SimpleJobDefManager(JobDefManagerSpec):
    def __init__(
        self, study_manager: StudyManagerSpec, store: StorageSpec, uri_root: str = "jobs", temp_dir: str = "/tmp"
    ):
        super().__init__()
        self.store = store
        self.uri_root = uri_root
        self.result_uri_root = "results"
        self.study_manager = study_manager
        if not os.path.isdir(temp_dir):
            raise ValueError("temp_dir {} is not a valid dir".format(temp_dir))
        self.temp_dir = temp_dir

    def job_uri(self, jid: str):
        return os.path.join(self.uri_root, jid)

    def create(self, meta: dict, uploaded_content: bytes) -> Dict[str, Any]:
        # validate meta to make sure it has:
        # - valid study ...

        jid = str(uuid.uuid4())
        meta[JobMetaKey.JOB_ID] = jid
        meta[JobMetaKey.SUBMIT_TIME] = time.time()
        meta[JobMetaKey.SUBMIT_TIME_ISO] = (
            datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
        )
        meta[JobMetaKey.STATUS] = RunStatus.SUBMITTED

        # write it to the store
        self.store.create_object(self.job_uri(jid), uploaded_content, meta, overwrite_existing=True)
        return meta

    def delete(self, jid: str):
        return self.store.delete_object(self.job_uri(jid))

    def _validate_meta(self, meta):
        """Validate meta against study.

        Args:
            meta: meta to validate

        Returns:

        """
        pass

    def _validate_uploaded_content(self, uploaded_content) -> bool:
        """Validate uploaded content for creating a run config. (THIS NEEDS TO HAPPEN BEFORE CONTENT IS PROVIDED NOW)

        Internally used by create and update. Use study_manager to get study info to validate.

        1. check all sites in deployment are in study
        2. check all sites in resources are in study
        3. check all sites in deployment are in resources
        4. each site in deployment need to have resources (each site in resource need to be in deployment ???)
        """
        pass

    def get_job(self, jid: str) -> Job:
        job_meta = self.store.get_meta(self.job_uri(jid))
        return job_from_meta(job_meta)

    def set_results_uri(self, jid: str, result_uri: str):
        updated_meta = {JobMetaKey.RESULT_LOCATION: result_uri}
        self.store.update_meta(self.job_uri(jid), updated_meta, replace=False)
        return self.get_job(jid)

    def get_apps(self, job: Job) -> Dict[str, bytes]:
        data_bytes = self.store.get_data(self.job_uri(job.job_id))
        job_id_dir = os.path.join(self.temp_dir, job.job_id)
        if os.path.exists(job_id_dir):
            shutil.rmtree(job_id_dir)
        os.mkdir(job_id_dir)
        unzip_all_from_bytes(data_bytes, job_id_dir)
        result_dict = {}
        for app in job.get_deployment():
            result_dict[app] = zip_directory_to_bytes(job_id_dir, app)
        return result_dict

    def get_content(self, jid: str) -> bytes:
        return self.store.get_data(self.job_uri(jid))

    def set_status(self, jid: str, status):
        meta = {JobMetaKey.STATUS: status}
        self.store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def update_meta(self, jid: str, meta):
        self.store.update_meta(uri=self.job_uri(jid), meta=meta, replace=False)

    def list_all(self) -> List[Job]:
        job_filter = _AllJobsFilter()
        self._scan(job_filter)
        return job_filter.result

    def _scan(self, job_filter: _JobFilter):
        jid_paths = self.store.list_objects(self.uri_root)
        if not jid_paths:
            return

        for jid_path in jid_paths:
            jid = pathlib.PurePath(jid_path).name
            meta = self.get_job(jid).meta
            if meta:
                ok = job_filter.filter_job(meta)
                if not ok:
                    break

    def get_jobs_by_status(self, status) -> List[Job]:
        job_filter = _StatusFilter(status)
        self._scan(job_filter)
        return job_filter.result

    def get_jobs_waiting_for_review(self, reviewer_name: str) -> List[Dict[str, Any]]:
        # scan the store!
        job_filter = _ReviewerFilter(reviewer_name, self.study_manager)
        self._scan(job_filter)
        return job_filter.result

    def set_approval(self, jid: str, reviewer_name: str, approved: bool, note: str) -> Dict[str, Any]:
        meta = self.get_job(jid).meta
        if meta:
            approvals = meta.get(JobMetaKey.APPROVALS)
            if not approvals:
                approvals = {}
                meta[JobMetaKey.APPROVALS] = approvals
            approvals[reviewer_name] = (approved, note)
            updated_meta = {JobMetaKey.APPROVALS: approvals}
            self.store.update_meta(self.job_uri(jid), updated_meta, replace=False)
        return meta
