# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from .fl_context import FLContext


class Security:
    def authenticate(self, fl_ctx: FLContext) -> (bool, str):
        """Check the authentication of the operations.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean of the authentication result,
            reason if failed authentication
        """
        pass

    def authorize(self, fl_ctx: FLContext) -> (bool, str):
        """Check the authorization of the operations.

        Args:
            fl_ctx: FLContext

        Returns:
            boolean of the authentication result,
            reason if failed authorization
        """
        pass
