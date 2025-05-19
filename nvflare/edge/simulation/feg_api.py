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
from urllib.parse import urlencode, urljoin

import requests

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo


class FegApi:
    def __init__(self, endpoint: str, device_info: DeviceInfo, user_info: UserInfo):
        self.endpoint = endpoint
        self.device_info = device_info
        self.user_info = user_info
        temp = device_info.copy()
        del temp["device_id"]
        device_qs = urlencode(temp)
        user_qs = urlencode(user_info)

        self.common_headers = {
            "X-Flare-Device-ID": device_info.device_id,
            "X-Flare-Device-Info": device_qs,
            "X-Flare-User-Info": user_qs,
        }

    def get_job(self, request: JobRequest) -> JobResponse:
        url = urljoin(self.endpoint, "job")
        body = {"capabilities": request.capabilities}
        headers = {"Content-Type": "application/json"}
        headers.update(self.common_headers)
        response = requests.post(url, json=body, headers=headers)

        code = response.status_code
        if code == 200:
            return JobResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())

    def get_task(self, request: TaskRequest) -> TaskResponse:
        url = urljoin(self.endpoint, "task")
        params = {
            "job_id": request.job_id,
        }
        response = requests.get(url, params=params, headers=self.common_headers)
        code = response.status_code
        if code == 200:
            return TaskResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())

    def report_result(self, report: ResultReport) -> ResultResponse:
        url = urljoin(self.endpoint, "result")
        body = {"result": report.result}
        headers = {"Content-Type": "application/json"}
        headers.update(self.common_headers)
        params = {
            "job_id": report.job_id,
            "task_name": report.task_name,
            "task_id": report.task_id,
        }
        response = requests.post(url, json=body, params=params, headers=headers)

        code = response.status_code
        if code == 200:
            return ResultResponse(**response.json())

        details = {"response": response.text}
        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", details)
