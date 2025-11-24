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
class CallOpt:

    def __init__(
        self,
        expect_result: bool = True,
        blocking: bool = True,
        timeout: float = 5.0,
        secure: bool = False,
        optional: bool = False,
    ):
        self.expect_result = expect_result
        self.blocking = blocking
        self.timeout = timeout
        self.secure = secure
        self.optional = optional

    def __str__(self):
        return (
            f"expect_result={self.expect_result} blocking={self.blocking} timeout={self.timeout} "
            f"secure={self.secure} optional={self.optional}"
        )
