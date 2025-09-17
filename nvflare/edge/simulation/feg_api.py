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

from nvflare.edge.constants import HttpHeaderKey
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.base_model import EdgeProtoKey
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
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
            "Content-Type": "application/json",
            HttpHeaderKey.DEVICE_ID: device_info.device_id,
            HttpHeaderKey.DEVICE_INFO: device_qs,
            HttpHeaderKey.USER_INFO: user_qs,
        }

    def get_job(self, request: JobRequest) -> JobResponse:
        return self._do_post(
            clazz=JobResponse,
            url=urljoin(self.endpoint, "job"),
            params={},
            body={EdgeProtoKey.JOB_NAME: request.job_name, EdgeProtoKey.CAPABILITIES: request.capabilities},
        )

    def get_task(self, request: TaskRequest) -> TaskResponse:
        return self._do_post(
            clazz=TaskResponse,
            url=urljoin(self.endpoint, "task"),
            params={EdgeProtoKey.JOB_ID: request.job_id},
            body={EdgeProtoKey.COOKIE: request.cookie} if request.cookie else {},
        )

    def report_result(self, report: ResultReport) -> ResultResponse:
        body = {
            EdgeProtoKey.STATUS: report.status,
            EdgeProtoKey.TASK_NAME: report.task_name,
            EdgeProtoKey.RESULT: report.result,
        }
        if report.cookie:
            body[EdgeProtoKey.COOKIE] = report.cookie

        return self._do_post(
            clazz=ResultResponse,
            url=urljoin(self.endpoint, "result"),
            params={
                EdgeProtoKey.JOB_ID: report.job_id,
                EdgeProtoKey.TASK_ID: report.task_id,
            },
            body=body,
        )

    def get_selection(self, request: SelectionRequest) -> SelectionResponse:
        return self._do_post(
            clazz=SelectionResponse,
            url=urljoin(self.endpoint, "selection"),
            params={EdgeProtoKey.JOB_ID: request.job_id},
            body={},
        )

    def _do_post(self, clazz, url, params, body):
        response = requests.post(url, params=params, json=body, headers=self.common_headers)
        code = response.status_code
        if code == 200:
            return clazz(**response.json())
        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())
