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
class CallOption:

    def __init__(
        self,
        expect_result: bool = True,
        blocking: bool = True,
        timeout: float = 5.0,
        secure: bool = False,
        optional: bool = False,
        collab_obj_name=None,
    ):
        """CallOption defines behavior of a collab call.

        Args:
            expect_result: whether result is expected from the remote object.
            blocking: whether rhe call is blocking. Only for group calls.
            timeout: when expecting result, the max number of secs to wait for result.
            secure: whether to use P2P secure messaging.
            optional: whether the call is optional.
            collab_obj_name: name of the collab object to be called.
        """
        self.expect_result = expect_result
        self.blocking = blocking
        self.timeout = timeout
        self.secure = secure
        self.optional = optional
        self.collab_obj_name = collab_obj_name

    def __str__(self):
        return (
            f"expect_result={self.expect_result} blocking={self.blocking} timeout={self.timeout} "
            f"secure={self.secure} optional={self.optional} collab_obj={self.collab_obj_name}"
        )
