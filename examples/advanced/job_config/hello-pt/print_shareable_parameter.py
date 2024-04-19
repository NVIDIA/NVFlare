# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import from_shareable
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class PrintShareable(Filter):
    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        dxo = from_shareable(shareable)
        model_weights = dxo.data

        count = 0
        keys = ""
        for item in model_weights.keys():
            keys += item + ";   "
            count += 1
        print(f"{fl_ctx.get_identity_name()} -----  Total parameters in the Shareable: {count}")

        return shareable
