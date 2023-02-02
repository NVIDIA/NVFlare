# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.fuel.f3.headers import Headers


class Message:

    def __init__(self, headers: Headers, payload: Any):
        """Construct an FCI message

         Raises:
             CommError: If any error encountered while starting up
         """

        self.headers = headers
        self.payload = payload

    def set_header(self, key: str, value):
        self.headers[key] = value

    def add_headers(self, headers: dict):
        self.headers.update(headers)

    def get_header(self, key: str, default=None):
        return self.headers.get(key, default)
