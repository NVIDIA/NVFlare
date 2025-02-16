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

from nvflare.edge.emulator.sample_task_processor import SampleTaskProcessor
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.emulator.eta_api import EtaApi
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


class DeviceEmulator:
    def __init__(self, endpoint: str, device_info: DeviceInfo, user_info: UserInfo,
                 capabilities: Optional[dict], processor: DeviceTaskProcessor):
        self.device_info = device_info
        self.user_info = user_info
        self.capabilities = capabilities
        self.processor = processor
        self.eta_api = EtaApi(endpoint, device_info, user_info)

    def run(self):

        try:

            job = self.fetch_job()
            self.processor.setup(job)
            log.info(f"Received job: {job}")

            while True:
                task = self.eta_api.get_task(job)
                log.info(f"Received task: {task}")
                if task.task_name == "end_run":
                    log.info(f"Job {job.job_id} {job.job_name} ended")
                    break

                result = self.processor.process_task(task)
                log.info(f"Task processed. Result: {result}")

                result_response = self.eta_api.report_result(task, result)
                log.info(f"Received result response: {result_response}")
                if result_response.status == "DONE":
                    log.info(f"Job {job.job_id} {job.job_name} is done")
                    break
                elif result_response.status != "OK":
                    log.error(f"Result report for task {task.task_name} is invalid")
                    break

                log.info(f"Task {task.task_name} result reported successfully")

            self.processor.shutdown()
            log.info(f"Job {job.job_name} run ended")

        except ApiError as error:
            log.error(f"Status: {error.status}\nMessage: {str(error)}\nDetails: {error.details}")

    def fetch_job(self) -> JobResponse:

        while True:
            job = self.eta_api.get_job(self.capabilities)
            if job.status == "OK":
                return job
            elif job.status == "DONE":
                job["job_done"] = True
                return job
            if job.status == "RETRY":
                wait = job.retry_wait if job.retry_wait else 5
                log.info(f"Retrying getting job in {wait} seconds")
                time.sleep(wait)
