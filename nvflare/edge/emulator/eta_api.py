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
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo


class EtaApi:

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

    def get_job(self, capabilities: dict = None) -> JobResponse:
        url = urljoin(self.endpoint, "job")
        body = {"capabilities": capabilities}
        headers = {"Content-Type": "application/json"}
        headers.update(self.common_headers)
        response = requests.post(url, json=body, headers=headers)

        code = response.status_code
        if code == 200:
            return JobResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())

    def get_task(self, job: JobResponse) -> TaskResponse:
        url = urljoin(self.endpoint, "task")
        params = {
            "session_id": job.session_id,
            "job_id": job.job_id,
        }
        response = requests.get(url, params=params, headers=self.common_headers)
        code = response.status_code
        if code == 200:
            return TaskResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())

    def report_result(self, task: TaskResponse, result: dict) -> ResultResponse:
        url = urljoin(self.endpoint, "result")
        body = {"result": result}
        headers = {"Content-Type": "application/json"}
        headers.update(self.common_headers)
        params = {
            "session_id": task.session_id,
            "task_name": task.task_name,
            "task_id": task.task_id,
        }
        response = requests.post(url, json=body, params=params, headers=headers)

        code = response.status_code
        if code == 200:
            return ResultResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}", response.json())


def run_test():
    device_info = DeviceInfo("1234", "flare_mobile", "1.0")
    user_info = UserInfo("demo_id", "demo_user")

    try:
        api = EtaApi("http://localhost:4321", device_info, user_info)

        job = api.get_job({})
        print(job)

        while True:
            task = api.get_task(job)
            print(task)
            if task.task_name == "end_run":
                break

            test_result = {"test": f"Result for task_id: {task.task_id}"}
            result_response = api.report_result(task, test_result)
            print(result_response)

            if result_response.status == "DONE":
                break

        print("Test run ended")
    except ApiError as error:
        print(f"Status: {error.status}\nMessage: {str(error)}\nDetails: {error.details}")


if __name__ == "__main__":
    run_test()
