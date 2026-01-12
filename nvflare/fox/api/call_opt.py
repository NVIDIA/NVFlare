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
        timeout: float = 60.0,
        secure: bool = False,
        optional: bool = False,
        target=None,
        parallel=0,
    ):
        """CallOption defines behavior of a collab call.

        Args:
            expect_result: whether result is expected from the remote object.
            blocking: whether the call is blocking. Only for group calls.
            timeout: when expecting result, the max number of secs to wait for result. Default is 60 seconds.
            secure: whether to use P2P secure messaging.
            optional: whether the call is optional.
            target: name of the collab object to be called.
            parallel: number of parallel outgoing messages.
        """
        self.expect_result = expect_result
        self.blocking = blocking
        self.timeout = timeout
        self.secure = secure
        self.optional = optional
        self.target = target
        self.parallel = parallel

        if not self.expect_result:
            # fire and forget - no need to control parallel
            self.parallel = 0

    def __str__(self):
        return (
            f"expect_result={self.expect_result} blocking={self.blocking} timeout={self.timeout} "
            f"secure={self.secure} optional={self.optional} target={self.target} parallel={self.parallel}"
        )
