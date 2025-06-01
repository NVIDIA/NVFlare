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
from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.models.base_model import BaseModel, EdgeProtoKey
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class ResultReport(BaseModel):
    def __init__(
        self,
        device_info: DeviceInfo,
        user_info: UserInfo,
        job_id: str,
        task_id: str,
        task_name: str = None,
        status: str = EdgeApiStatus.OK,
        result: dict = None,
        cookie: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.device_info = device_info
        self.user_info = user_info
        self.job_id = job_id
        self.task_id = task_id
        self.task_name = task_name
        self.result = result
        self.cookie = cookie
        self.status = status

        if kwargs:
            self.update(kwargs)

    @classmethod
    def validate(cls, d: dict) -> str:
        return cls.check_keys(
            d,
            [
                EdgeProtoKey.JOB_ID,
                EdgeProtoKey.DEVICE_INFO,
                EdgeProtoKey.TASK_ID,
                EdgeProtoKey.TASK_NAME,
                EdgeProtoKey.STATUS,
                EdgeProtoKey.RESULT,
                EdgeProtoKey.COOKIE,
            ],
        )

    @classmethod
    def from_dict(cls, d: dict):
        error = cls.validate(d)
        if error:
            return error, None

        error, device_info = DeviceInfo.extract_from_dict(d)
        if error:
            return error, None

        _, user_info = UserInfo.extract_from_dict(d)

        report = ResultReport(
            device_info,
            user_info,
            job_id=d.pop(EdgeProtoKey.JOB_ID),
            task_id=d.pop(EdgeProtoKey.TASK_ID),
            task_name=d.pop(EdgeProtoKey.TASK_NAME),
            status=d.pop(EdgeProtoKey.STATUS),
            result=d.pop(EdgeProtoKey.RESULT),
            cookie=d.pop(EdgeProtoKey.COOKIE),
        )
        report.update(d)
        return "", report
