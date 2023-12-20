# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import copy
from typing import Any

from nvflare.app_common.app_constant import AppConstants

from .simple_aggregator import SimpleAggregator


class AVAggregator(SimpleAggregator):
    def __init__(self):
        SimpleAggregator.__init__(self)
        self.current_result = {}
        self.current_meta = {}
        self.num_clients = 0

    def processing_training_result(self, client_name: str, trained_weights: Any, trained_meta: dict) -> bool:
        if not isinstance(trained_weights, dict):
            self.log_error(
                self.fl_ctx, f"invalid result from client {client_name}: expect dict but got {type(trained_weights)}"
            )
            return False

        current_round = self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        print(f"====== Round {current_round}: received result from client {client_name}: {trained_weights}")
        print(f"====== Round {current_round}: current result: {self.current_result}")
        for k, v in trained_weights.items():
            if k in self.current_result:
                for i, w in enumerate(v):
                    self.current_result[k][i] += w
            else:
                self.current_result[k] = copy.deepcopy(trained_weights[k])

        self.current_meta.update(trained_meta)
        self.num_clients += 1
        print(f"accumulated after client {self.num_clients}: {self.current_result}")
        return True

    def reset(self):
        self.current_result = {}
        self.current_meta = {}
        self.num_clients = 0

    def aggregate_training_result(self) -> (Any, dict):
        result = self.current_result
        meta = self.current_meta
        if self.num_clients > 0:
            for k, v in result.items():
                for i, w in enumerate(v):
                    v[i] = w / self.num_clients
        print(f"after aggregated: {result}")
        return result, meta
