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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import Job, RunStatus


class JobDefManagerSpec(FLComponent, ABC):
    """Job Definition Management API."""

    @abstractmethod
    def create(self, meta: dict, uploaded_content: bytes, fl_ctx: FLContext) -> Dict[str, Any]:
        """Create a new job permanently.

        The caller must have validated the content already and created initial meta. Receives bytes of uploaded folder,
        uploading to permanent store, create unique Job ID (jid) and return meta.

        Args:
            meta: caller-provided meta info
            uploaded_content: data of the job definition
            fl_ctx (FLContext): FLContext information

        Returns:
            A dict containing meta info. Additional meta info are added, especially
            a unique Job ID (jid) which has been created.
        """
        pass

    @abstractmethod
    def get_job(self, jid: str, fl_ctx: FLContext) -> Job:
        """Gets the Job object through the job ID.

        Args:
            jid (str): Job ID
            fl_ctx (FLContext): FLContext information

        Returns:
            A Job object
        """
        pass

    @abstractmethod
    def get_app(self, job: Job, app_name: str, fl_ctx: FLContext) -> bytes:
        """Get the contents of the specified app in bytes.

        Args:
            job: Job object
            app_name: name of the app to get
            fl_ctx (FLContext): FLContext information

        Returns:
            Content of the specified app in bytes
        """
        pass

    @abstractmethod
    def get_apps(self, job: Job, fl_ctx: FLContext) -> Dict[str, bytes]:
        """Get the all the apps of a Job.

        Args:
            job: Job object
            fl_ctx (FLContext): FLContext information

        Returns:
            A dictionary of app names with the content of the corresponding app encoded in bytes
        """
        pass

    @abstractmethod
    def get_content(self, jid: str, fl_ctx: FLContext) -> Optional[bytes]:
        """Gets the entire uploaded content for a Job.

        Args:
            jid (str): Job ID
            fl_ctx (FLContext): FLContext information

        Returns:
            Uploaded content of the job in bytes
        """
        pass

    @abstractmethod
    def get_job_data(self, jid: str, fl_ctx: FLContext) -> dict:
        """Gets the entire uploaded content and workspace for a job.

        Args:
            jid (str): Job ID
            fl_ctx (FLContext): FLContext information

        Returns:
            a dict to hold the job data and workspace. With the format: {JobDataKey.JOB_DATA.value: stored_data, JobDataKey.WORKSPACE_DATA: workspace_data}

        """
        pass

    @abstractmethod
    def update_meta(self, jid: str, meta, fl_ctx: FLContext):
        """Update the meta of an existing Job.

        Args:
            jid (str): Job ID
            meta: dictionary of metadata for the job
            fl_ctx (FLContext): FLContext information

        """
        pass

    @abstractmethod
    def refresh_meta(self, job: Job, meta_keys: list, fl_ctx: FLContext):
        """Refresh meta of the job as specified in the meta keys
        Save the values of the specified keys into job store

        Args:
            job: job object
            meta_keys: meta keys need to updated
            fl_ctx: FLContext

        """
        pass

    @abstractmethod
    def set_status(self, jid: str, status: RunStatus, fl_ctx: FLContext):
        """Set status of an existing Job.

        Args:
            jid (str): Job ID
            status (RunStatus): status to set
            fl_ctx (FLContext): FLContext information

        """
        pass

    @abstractmethod
    def get_all_jobs(self, fl_ctx: FLContext) -> List[Job]:
        """Gets all Jobs in the system.

        Args:
            fl_ctx (FLContext): FLContext information

        Returns:
            A list of all jobs
        """
        pass

    @abstractmethod
    def get_jobs_by_status(self, run_status: RunStatus, fl_ctx: FLContext) -> List[Job]:
        """Gets Jobs of a specified status.

        Args:
            run_status (RunStatus): status to filter for
            fl_ctx (FLContext): FLContext information

        Returns:
            A list of Jobs of the specified status
        """
        pass

    @abstractmethod
    def get_jobs_waiting_for_review(self, reviewer_name: str, fl_ctx: FLContext) -> List[Job]:
        """Gets Jobs waiting for review for the specified user.

        Args:
            reviewer_name (str): reviewer name
            fl_ctx (FLContext): FLContext information

        Returns:
            A list of Jobs waiting for review for the specified user.
        """
        pass

    @abstractmethod
    def set_approval(
        self, jid: str, reviewer_name: str, approved: bool, note: str, fl_ctx: FLContext
    ) -> Dict[str, Any]:
        """Sets the approval for the specified user for a certain Job.

        Args:
            jid (str): job id
            reviewer_name (str): reviewer name
            approved (bool): whether job is approved
            note (str): any note message
            fl_ctx (FLContext): FLContext information

        Returns:
            A dictionary of Job metadata.
        """
        pass

    @abstractmethod
    def delete(self, jid: str, fl_ctx: FLContext):
        """Deletes the specified Job.

        Args:
            jid (str): Job ID
            fl_ctx (FLContext): FLContext information

        """
        pass

    @abstractmethod
    def save_workspace(self, jid: str, data: bytes, fl_ctx: FLContext):
        """Save the job workspace to the job storage.

        Args:
            jid (str): Job ID
            data: Job workspace data
            fl_ctx (FLContext): FLContext information

        """
        pass
