# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class CommError(Exception):

    # Error codes
    ERROR = "ERROR"
    NOT_READY = "NOT_READY"
    BAD_DATA = "BAD_DATA"
    BAD_CONFIG = "BAD_CONFIG"
    CLOSED = "CLOSED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    TIMEOUT = "TIMEOUT"

    def __init__(self, code: str, message=None):
        self.code = code
        self.message = message

    def __str__(self):
        if self.message:
            return f"Code: {self.code} Error: {self.message}"
        else:
            return f"Code: {self.code}"
