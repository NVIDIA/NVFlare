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
import traceback
from typing import Any


class ApiError(Exception):
    def __init__(self, status_code: int, status: str, message=None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.status = status
        if isinstance(details, Exception):
            tb = traceback.format_tb(details.__traceback__)
            self.details = {"traceback": tb}
        else:
            self.details = details

    def to_dict(self):
        return {
            "status": self.status,
            "message": str(self),
            "details": self.details,
        }
