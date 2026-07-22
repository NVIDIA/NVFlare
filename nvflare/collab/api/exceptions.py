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


class RunAborted(Exception):
    """Raised when a Collab run is aborted."""

    pass


class CollabCallError(Exception):
    """Raised when a call to a Collab function fails."""

    def __init__(
        self,
        site: str,
        func_name: str,
        cause,
        cause_type: str = None,
        remote_traceback: str = None,
    ):
        self.site = site.split(".", 1)[0]
        self.func_name = func_name
        self.cause = cause
        self.cause_type = cause_type or type(cause).__name__
        self.remote_traceback = remote_traceback
        super().__init__(f"call to {self.site}.{func_name} failed: {cause}")
