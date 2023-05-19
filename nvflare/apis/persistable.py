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


class StatePersistable:
    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        """Generate data from state to be persisted.

        Args:
            fl_ctx: FLContext

        Returns:
            A dict serializable persist data
        """
        return {}

    def restore(self, state_data: dict, fl_ctx: FLContext):
        """Restore the state from persisted data.

        Args:
            state_data: serialized persist data
            fl_ctx: FLContext
        """
        pass
