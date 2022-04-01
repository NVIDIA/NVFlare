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
from enum import Enum
from typing import Dict, List

from .fl_context import FLContext
from .job_def import Job


class DispatchStatus(str, Enum):
    DEPLOY_FAILED = "DEPLOY_FAILED"
    START_FAILED = "START_FAILED"
    SUCCESS = "SUCCESS"


class JobDispatcherSpec(ABC):
    """API specification for job dispatcher.

    Q: Say client 1, 2, 3, 4 are participating and originally only client 1, 2, 3 are up
      when client 4 join after the Job is dispatched and running on client 1, 2, 3, how can we handle this?

    Q: Is site and client the same concept?
       => let's assume they are 1 to 1 right now, so that means we have the resource specification for each client.
    """

    @abstractmethod
    def dispatch(self, job: Job, sites: List[str], fl_ctx: FLContext) -> Dict[str, DispatchStatus]:
        """Dispatch a job.


        #1 Need to check if each of the clients is dispatched successful...
        #2 Create a run_id in the manager that calls this method...

        Args:
            job (Job): The job.
            sites (List[str]): A list of site to deploy to.
            fl_ctx: system context

        Returns:
            A dictionary of {site_name: DispatchStatus}
        """
        pass

    @abstractmethod
    def stop(self, job: Job, sites: List[str], fl_ctx: FLContext) -> Dict[str, bool]:
        """Stop a job.

        Args:
            job (Job): The job to be stopped.
            sites (List[str]): A list of site to stop.
            fl_ctx: system context

        Returns:
            A dictionary of {site_name: boolean}
            The boolean specify whether stop success or not.
        """
        pass
