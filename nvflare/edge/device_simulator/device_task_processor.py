# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo


class DeviceTaskProcessor(ABC):
    """
    The spec for a task processor that handles tasks on edge devices
    """

    @abstractmethod
    def setup(self, device_info: DeviceInfo, user_info: UserInfo, job: JobResponse) -> None:
        """
        Setup for a new job

        Args
            device_info: Device information from request header
            user_info: User information from request header
            job: Job information returned by server
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean-up the resources allocated for the job and get ready for next
        """
        pass

    def cancel(self) -> None:
        """
        Cancel the task processing
        """
        pass

    @abstractmethod
    def process_task(self, task: TaskResponse) -> dict:
        """
        Process a task and return the result. This method is repeated for
        each task until all tasks are done

        Args:
            task: The task information from server

        Returns:
            The result as a dict
        """
        pass
