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
from typing import List, Optional, Union
from urllib.parse import parse_qs, urlencode


class EdgeProtoKey:
    STATUS = "status"
    JOB_NAME = "job_name"
    JOB_ID = "job_id"
    TASK_ID = "task_id"
    CAPABILITIES = "capabilities"
    TASK_NAME = "task_name"
    RESULT = "result"
    COOKIE = "cookie"
    DEVICE_INFO = "device_info"
    DEVICE_ID = "device_id"
    USER_INFO = "user_info"
    METHODS = "methods"


class BaseModel(dict):
    """Dictionary based model"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def from_query_string(self, qs: str):
        params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(qs).items()}
        self.update(params)

    def to_query_string(self) -> str:
        return urlencode(self, doseq=True)

    def get_device_id(self) -> Optional[str]:
        device_info = self.get("device_info")
        if not device_info:
            return None

        return device_info.get("device_id")

    @staticmethod
    def check_keys(d: dict, keys: Union[str, List[str]]) -> str:
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in d:
                return f"missing {key}"
        return ""
