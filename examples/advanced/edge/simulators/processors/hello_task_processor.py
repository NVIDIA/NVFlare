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
import logging
from typing import Optional

from nvflare.edge.device_simulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


class HelloTaskProcessor(DeviceTaskProcessor):
    def __init__(self, parameters: Optional[dict], data_file: str = "test.data"):
        self.parameters = parameters
        self.job_id = None
        self.job_name = None
        self.data_file = data_file

    def setup(self, device_info: DeviceInfo, user_info: UserInfo, job: JobResponse) -> None:
        self.job_id = job.job_id
        self.job_name = job.job_name

    def shutdown(self) -> None:
        pass

    def process_task(self, task: TaskResponse) -> dict:
        log.info(f"Processing task {task.task_name}")

        result = None
        if task.task_name == "train":
            weights = task.task_data.get("weights")
            if weights:
                w = [x + 0.1 for x in weights]
            else:
                w = [0, 0, 0, 0]
            result = {"weights": w}
        elif task.task_name == "validate":
            result = {"accuracy": [0.01, 0.02, 0.03, 0.04]}
        else:
            log.error(f"Received unknown task: {task.task_name}")

        return result
