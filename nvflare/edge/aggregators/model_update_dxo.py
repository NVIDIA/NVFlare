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
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


class ModelUpdateDXOAggregator(Aggregator):

    def __init__(self):
        Aggregator.__init__(self)
        self.dict = None
        self.count = 0

    def _aggregate(self, weight_base, weight_to_add):
        # aggregates the dict on corresponding keys
        for key, sub_object in weight_base.items():
            if isinstance(sub_object, dict):
                sub_to_add = weight_to_add.get(key)
                self._aggregate(sub_object, sub_to_add)
            weight_base[key] = np.add(weight_base[key], weight_to_add[key])
        return weight_base

    def reset(self, fl_ctx: FLContext):
        self.dict = None
        self.count = 0

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        # check data_kind
        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            raise ValueError(f"DXO data_kind must be {DataKind.WEIGHT_DIFF}, but got {dxo.data_kind}")

        # get weights and add to base
        weight_to_add = dxo.data.get("dict")
        # convert to numpy arrays if they are lists
        if weight_to_add is not None:
            for key, value in weight_to_add.items():
                if isinstance(value, list):
                    weight_to_add[key] = np.array(value)
        if weight_to_add is None:
            raise ValueError("Model dict is empty, please check the message")
        if self.dict is None:
            self.dict = weight_to_add
        else:
            self.dict = self._aggregate(self.dict, weight_to_add)

        # get count and add to base
        count = dxo.data.get("count", 1)

        # print the count
        self.log_info(fl_ctx, f"Aggregator got {count} updates")
        self.count += count

        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": self.dict, "count": self.count})
        # once returned to upper layer, reset the aggregator
        self.reset(fl_ctx)
        return dxo.to_shareable()
