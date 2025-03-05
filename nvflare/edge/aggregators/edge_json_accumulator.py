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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.constants import MsgKey


class EdgeJsonAccumulator(Aggregator):
    def __init__(self, aggr_key: str):
        Aggregator.__init__(self)
        self.weights = None
        self.num_devices = 0
        self.aggr_key = aggr_key

    def _aggregate(self, weight_base, weight_to_add):
        # aggregates the dict on items with the aggregation key
        # iteratively find the key and add the values
        for key, sub_object in weight_base.items():
            if isinstance(sub_object, dict):
                sub_to_add = weight_to_add.get(key)
                self._aggregate(sub_object, sub_to_add)
        if self.aggr_key in weight_base:
            weight_base[self.aggr_key] = np.add(weight_base[self.aggr_key], weight_to_add[self.aggr_key])
        return weight_base

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        weight_to_add = shareable.get(MsgKey.RESULT)
        if weight_to_add is None:
            return True

        # bottom level does not have num_devices
        # in which case num_devices_to_add is 1
        num_devices_to_add = shareable.get("num_devices")
        if num_devices_to_add is None:
            num_devices_to_add = 1
        self.num_devices += num_devices_to_add
        self.log_info(fl_ctx, f"Accepting result with {num_devices_to_add}")

        # add new weights to the existing weights
        if self.weights is None:
            self.weights = weight_to_add
        else:
            self.weights = self._aggregate(self.weights, weight_to_add)

        return True

    def reset(self, fl_ctx: FLContext):
        self.weights = None
        self.num_devices = 0

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        return Shareable({MsgKey.RESULT: self.weights, "num_devices": self.num_devices})
