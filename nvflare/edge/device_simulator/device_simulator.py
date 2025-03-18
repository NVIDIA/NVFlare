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
import time
from typing import Optional

from nvflare.edge.constants import EdgeApiKey, EdgeApiStatus
from nvflare.edge.device_simulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.device_simulator.feg_api import FegApi
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


class DeviceSimulator:
    def __init__(
        self,
        endpoint: str,
        device_info: DeviceInfo,
        user_info: UserInfo,
        capabilities: Optional[dict],
        processor: DeviceTaskProcessor,
    ):
        self.device_info = device_info
        self.device_id = device_info.device_id
        self.user_info = user_info
        self.capabilities = capabilities
        self.processor = processor
        self.feg_api = FegApi(endpoint, device_info, user_info)

    def run(self):
        try:
            job = self.fetch_job()
            self.processor.setup(self.device_info, self.user_info, job)
            log.info(f"Received job: {job}")

            while True:
                task = self.fetch_task(job)
                if not task:
                    log.info("Job is done")
                    break
                log.info(f"Device:{self.device_id} Received task.")

                # Catch exception
                result = self.processor.process_task(task)
                log.info(f"Device:{self.device_id} Task processed.")
                # Check result
                result_response = self.feg_api.report_result(task, result)
                log.info(f"Device:{self.device_id} Received result response: {result_response}")
                task_done = task.get("task_done", False)
                if task_done or result_response.status == EdgeApiStatus.DONE:
                    log.info(f"Job {job.job_id} {job.job_name} is done")
                    break
                elif result_response.status == EdgeApiStatus.NO_JOB:
                    log.info(f"Job {job.job_id} {job.job_name} is gone")
                    break
                elif result_response.status == EdgeApiStatus.RETRY:
                    continue
                elif result_response.status != EdgeApiStatus.OK:
                    log.error(
                        f"Device:{self.device_id} Result report for task {task.task_name}"
                        f" is invalid. Status: {result_response.status}"
                    )
                    continue

                log.info(f"Device:{self.device_id} Task {task.task_name} result reported successfully")

            self.processor.shutdown()
            log.info(f"Device:{self.device_id} Job {job.job_name} run ended")

        except ApiError as error:
            log.error(f"Status: {error.status}\nMessage: {str(error)}\nDetails: {error.details}")

    def fetch_job(self) -> JobResponse:
        while True:
            try:
                job = self.feg_api.get_job(self.capabilities)
                if job.status == "OK":
                    return job
                elif job.status in {"DONE", "NO_JOB"}:
                    job[EdgeApiKey.JOB_DONE] = True
                    return job
                if job.status == "RETRY":
                    wait = job.retry_wait if job.retry_wait else 5
                    log.info(f"Device:{self.device_id} Retrying getting job in {wait} seconds")
                    time.sleep(wait)
            except ApiError as error:
                log.error(f"Job request error. Status: {error.status}\nMessage: {str(error)}\nDetails: {error.details}")
                time.sleep(5)

    def fetch_task(self, job: JobResponse) -> Optional[TaskResponse]:
        while True:
            try:
                task = self.feg_api.get_task(job)
                if task.status == "OK":
                    return task
                elif task.status == "DONE":
                    task["task_done"] = True
                    return task
                elif task.status == "NO_TASK":
                    return None
                elif task.status == "RETRY":
                    wait = task.retry_wait if task.retry_wait else 5
                    log.info(f"Device:{self.device_id} Retrying getting task in {wait} seconds")
                    time.sleep(wait)
                else:
                    raise ApiError(500, f"wrong status: {task.status}")
            except ApiError as error:
                log.error(
                    f"Task request error. Status: {error.status}\nMessage: {str(error)}\nDetails: {error.details}"
                )
                time.sleep(5)
