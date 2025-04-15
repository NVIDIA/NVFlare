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
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.edge.constants import EdgeTaskHeaderKey


class AddUpdateHeader(Filter):
    def __init__(self):
        Filter.__init__(self)

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        rc = shareable.get_return_code()
        if rc != ReturnCode.OK:
            return shareable

        has_update = shareable.get_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA)
        if has_update is None:
            # add the header only if it's not present.
            shareable.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, True)
            self.log_info(fl_ctx, f"added {EdgeTaskHeaderKey.HAS_UPDATE_DATA} header to result")
        return shareable
