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


from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class DummyFilter(DXOFilter):
    def __init__(self):
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Filtering DXO: data={dxo.data}, meta={dxo.meta}")

        return dxo
