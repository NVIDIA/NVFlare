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


class DeviceInfo(BaseModel):
    """Device information"""

    def __init__(
        self,
        device_id: str,
        app_name: str = None,
        app_version: str = None,
        platform: str = None,
        platform_version: str = None,
        **kwargs,
    ):
        super().__init__()
        self.device_id = device_id
        self.app_name = app_name
        self.app_version = app_version
        self.platform = platform
        self.platform_version = platform_version

        if kwargs:
            self.update(kwargs)

    @staticmethod
    def extract_from_dict(d: dict):
        device_info_dict = d.pop(EdgeProtoKey.DEVICE_INFO, None)
        if not device_info_dict:
            return "missing device_info", None

        device_id = device_info_dict.pop(EdgeProtoKey.DEVICE_ID, None)
        if not device_id:
            return "missing device_id", None

        device_info = DeviceInfo(device_id)
        device_info.update(device_info_dict)
        return "", device_info
