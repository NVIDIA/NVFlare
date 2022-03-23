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

from abc import ABC, abstractmethod
from typing import List

from nvflare.mt.job_def import Job, RunStatus


class JobDefManagerSpec(ABC):
    """Job Definition Management API."""

    @abstractmethod
    def create(self, meta: dict, uploaded_content: bytes) -> Job:
        """Create a new job permanently.

        The caller must have validated the content already and created initial meta. Receives bytes of uploaded folder,
        uploading to permanent store, create unique Job ID (jid) and return meta.

        Args:
            meta: caller-provided meta info
            uploaded_content: data of the job definition

        Returns: a dict containing meta info. Additional meta info are added, especially
        a unique Job ID (jid) which has been created.

        """
        pass

    @abstractmethod
    def get_job(self, jid: str) -> Job:
        """Get the info for a Job."""
        pass

    @abstractmethod
    def get_content(self, jid: str) -> bytes:
        """Gets the uploaded content for a Job."""
        pass

    @abstractmethod
    def set_status(self, jid: str, status):
        """Set status of an existing Job."""
        pass

    @abstractmethod
    def list_all(self) -> List[Job]:
        """Return a list of all Jobs."""
        pass

    @abstractmethod
    def get_jobs_by_status(self, run_status: RunStatus) -> List[Job]:
        """Return a list of Jobs of a specified status."""
        pass

    @abstractmethod
    def get_jobs_waiting_for_review(self, reviewer_name: str) -> List[Job]:
        """Return a list of Jobs waiting for review for the specified user."""
        pass

    @abstractmethod
    def set_approval(self, jid: str, reviewer_name: str, approved: bool, note: str) -> Job:
        """Sets the approval for the specified user for a certain Job."""
        pass

    @abstractmethod
    def delete(self, jid: str):
        """Deletes the specified Job."""
        pass
