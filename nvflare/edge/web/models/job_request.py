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
from nvflare.edge.web.models.base_model import BaseModel, EdgeProtoKey
from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo


class JobRequest(BaseModel):
    def __init__(
        self,
        job_name: str,
        device_info: DeviceInfo,
        user_info: UserInfo,
        capabilities: Capabilities = None,
        **kwargs,
    ):
        super().__init__()
        self.job_name = job_name
        self.device_info = device_info
        self.user_info = user_info
        self.capabilities = capabilities

        if kwargs:
            self.update(kwargs)

    @classmethod
    def validate(cls, d: dict) -> str:
        return cls.check_keys(d, [EdgeProtoKey.CAPABILITIES, EdgeProtoKey.DEVICE_INFO, EdgeProtoKey.JOB_NAME])

    @classmethod
    def from_dict(cls, d: dict):
        error = cls.validate(d)
        if error:
            return error, None

        job_name = d.pop(EdgeProtoKey.JOB_NAME, None)
        error, device_info = DeviceInfo.extract_from_dict(d)
        if error:
            return error, None

        _, user_info = UserInfo.extract_from_dict(d)

        error, caps = Capabilities.extract_from_dict(d)
        if error:
            return error, None

        job_req = JobRequest(job_name, device_info, user_info, caps, **d)
        return "", job_req
