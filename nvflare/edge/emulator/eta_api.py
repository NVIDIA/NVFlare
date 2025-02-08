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
from urllib.parse import urlencode

import requests

from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
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

    def get_job(self, capabilities: dict = None):
        url = self.endpoint + "/job"
        body = {"capabilities": capabilities}
        headers = {"Content-Type": "application/json"}
        headers.update(self.common_headers)
        response = requests.post(url, json=body, headers=headers)

        code = response.status_code
        if code == 200:
            return JobResponse(**response.json())

        raise ApiError(code, "ERROR", f"API Call failed with status code {code}")


device_info = DeviceInfo("1234", "flare_mobile", "1.0")
user_info = UserInfo("demo_id", "demo_user")

api = EtaApi("http://localhost:4321", device_info, user_info)

job_response = api.get_job({})
print(job_response)


