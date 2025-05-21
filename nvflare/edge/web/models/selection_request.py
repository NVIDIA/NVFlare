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
from nvflare.edge.web.models.device_info import DeviceInfo


class SelectionRequest(BaseModel):
    def __init__(
        self,
        device_info: DeviceInfo,
        job_id: str,
        **kwargs,
    ):
        super().__init__()
        self.device_info = device_info
        self.job_id = job_id

        if kwargs:
            self.update(kwargs)

    @classmethod
    def validate(cls, d: dict) -> str:
        return cls.check_keys(d, [EdgeProtoKey.JOB_ID, EdgeProtoKey.DEVICE_INFO])

    @classmethod
    def from_dict(cls, d: dict):
        error = cls.validate(d)
        if error:
            return error, None

        error, device_info = DeviceInfo.extract_from_dict(d)
        if error:
            return error, None

        req = SelectionRequest(device_info, d.get(EdgeProtoKey.JOB_ID))
        req.update(d)
        return "", req
