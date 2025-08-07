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

from nvflare.edge.simulation.simulated_device import SimulatedDevice
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.utils.log_utils import get_obj_logger


class DeviceTaskProcessor(ABC):
    """
    The spec for a task processor that handles tasks on edge devices
    """

    def __init__(self):
        self.device: SimulatedDevice = None
        self.logger = get_obj_logger(self)

    @property
    def device_info(self):
        return self.device.get_device_info() if self.device else None

    @property
    def user_info(self):
        return self.device.get_user_info() if self.device else None

    @property
    def job_id(self):
        return self.device.job_id

    @property
    def job_name(self):
        return self.device.job_name

    @property
    def job_data(self):
        return self.device.job_data

    @property
    def job_method(self):
        return self.device.job_method

    @abstractmethod
    def setup(self, job: JobResponse) -> None:
        """
        Setup for a new job

        Args
            device: the SimulatedDevice object
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
