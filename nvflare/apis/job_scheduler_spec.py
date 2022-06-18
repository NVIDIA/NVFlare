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
from typing import Dict, List, Optional

from .fl_context import FLContext
from .job_def import Job


class DispatchInfo:
    """Information needed for dispatch"""

    def __init__(self, app_name: str, resource_requirements: dict, token: Optional[str]):
        self.app_name = app_name
        self.resource_requirements = resource_requirements
        self.token = token

    def __eq__(self, other):
        return (
            self.app_name == other.app_name
            and self.resource_requirements == other.resource_requirements
            and self.token == other.token
        )

    def __repr__(self):
        return f"{self.__class__.__name__}: app_name: {self.app_name}, resource_requirements: {self.resource_requirements}, token: {self.token}"


class JobSchedulerSpec(ABC):
    @abstractmethod
    def schedule_job(
        self, job_candidates: List[Job], fl_ctx: FLContext
    ) -> (Optional[Job], Optional[Dict[str, DispatchInfo]]):
        """Try to schedule a Job.

        Args:
            job_candidates: The candidate to choose from.
            fl_ctx: FLContext.

        Returns:
            A tuple of (job, sites_dispatch_info), if there is a Job that satisfy the criteria of the scheduler.
            sites_dispatch_info is a dict of {site name: DispatchInfo}.
            Otherwise, return (None, None).
        """
        pass
