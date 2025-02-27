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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator

import numpy as np


class EdgeResultAccumulator(Aggregator):
    def __init__(self):
        Aggregator.__init__(self)
        self.weights = None
        self.num_devices = 0

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        self.log_info(fl_ctx, f"Accepting: {shareable}")
        self.num_devices += 1
        w = shareable.get("weights")
        if self.weights is None:
            self.weights = w
        else:
            self.weights = np.add(self.weights, w)
        return True

    def reset(self, fl_ctx: FLContext):
        self.weights = None
        self.num_devices = 0

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        return Shareable({"weights": self.weights, "num_devices": self.num_devices})
